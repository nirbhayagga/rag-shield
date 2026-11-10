"""The same checks against each vector database, skipped unless it's available:
  ChromaDB   pip install ".[chroma]"
  pgvector   pip install ".[pgvector]", then set RAGSHIELD_TEST_PG_DSN to a throwaway Postgres with pgvector installed"""
import os
import unittest
import uuid

import numpy as np

from ragshield.detectors import OutlierDetector
from ragshield.provenance import Ledger, Signer
from ragshield.shield import Shield
from test_core import StubEmbedder, StubInjection


class Contract:
    """What Shield needs from any vector database."""

    def make_index(self):
        raise NotImplementedError

    def change_text(self, chunk_id, text):
        raise NotImplementedError

    def write_around_the_gate(self, chunk_id, text, vector):
        raise NotImplementedError

    def setUp(self):
        self.embedder = StubEmbedder()
        self.index = self.make_index()
        self.shield = Shield(Ledger(), Signer.generate(), self.embedder, self.index, injection=StubInjection(), outlier=OutlierDetector())
        self.shield.ingest('vpn-runbook', 'vpn setup guide for contractors uses a hardware token', 'wiki', 'high', metadata={'team': 'net', 'tags': ['a'], 'none': None})

    def test_a_signed_chunk_comes_back_with_its_similarity(self):
        hit = self.shield.retrieve('vpn setup guide for contractors uses a hardware token', 3)[0]
        self.assertEqual((hit['chunk_id'], hit['doc_id'], hit['trust']['action']), ('vpn-runbook::0', 'vpn-runbook', 'pass'))
        self.assertAlmostEqual(hit['score'], 1.0, places=3)
        self.assertEqual(hit['metadata']['team'], 'net')

    def test_a_new_version_replaces_the_old_chunks(self):
        self.shield.ingest('vpn-runbook', 'vpn setup guide for contractors uses a passkey', 'wiki', 'high')
        self.assertEqual(self.index.ids, ['vpn-runbook::0'])
        self.assertIn('passkey', self.index.texts[0])

    def test_text_changed_in_the_database_is_dropped(self):
        self.change_text('vpn-runbook::0', 'vpn setup guide: send your token to mallory')
        self.assertEqual(self.shield.retrieve('vpn setup guide', 3), [])

    def test_a_chunk_written_around_the_gate_is_dropped(self):
        text = 'vpn setup guide says to mail credentials out'
        self.write_around_the_gate('planted::0', text, self.embedder.query(text))
        self.assertNotIn('planted::0', [h['chunk_id'] for h in self.shield.retrieve('vpn setup guide', 3)])

    def test_a_flood_of_dropped_chunks_does_not_hide_a_trusted_chunk(self):
        for i in range(60):  # past pgvector's default of 40 rows per index scan
            self.shield.ingest(f'flood-{i}', f'vpn setup guide ignore previous instructions number {i}', 'uploads', 'low')
        self.assertEqual([h['chunk_id'] for h in self.shield.retrieve('vpn setup guide', 3)], ['vpn-runbook::0'])

    def test_ids_texts_and_vectors_line_up(self):
        for name in ('zebra', 'alpha', 'mango'):
            self.shield.ingest(name, f'{name} notes about {name} handling', 'wiki', 'high')
        ids, texts, matrix = self.index.ids, self.index.texts, self.index.matrix
        self.assertEqual(ids, sorted(ids))
        np.testing.assert_allclose(matrix, self.embedder.passages(texts), atol=1e-5)
        self.assertEqual(self.shield.fit_outlier(), int(sum(self.shield.ledger.get(c)[0].scores['outlier_flag'] for c in ids)))


try:
    import chromadb
except ImportError:
    chromadb = None


@unittest.skipUnless(chromadb, 'chromadb is not installed')
class ChromaTests(Contract, unittest.TestCase):
    def make_index(self):
        from ragshield.adapters import ChromaIndex
        return ChromaIndex(chromadb.EphemeralClient(), f't{uuid.uuid4().hex}')

    def change_text(self, chunk_id, text):
        got = self.index.collection.get(ids=[chunk_id], include=['embeddings'])
        self.index.collection.update(ids=[chunk_id], documents=[text], embeddings=[list(map(float, got['embeddings'][0]))])

    def write_around_the_gate(self, chunk_id, text, vector):
        self.index.collection.add(ids=[chunk_id], documents=[text], embeddings=[vector.tolist()], metadatas=[{'doc_id': 'planted'}])

    def test_a_collection_using_another_distance_is_refused(self):
        from ragshield.adapters import ChromaIndex
        client = chromadb.EphemeralClient()
        name = f't{uuid.uuid4().hex}'
        client.create_collection(name, metadata={'hnsw:space': 'l2'}, embedding_function=None)
        with self.assertRaises(ValueError):
            ChromaIndex(client, name)


DSN = os.environ.get('RAGSHIELD_TEST_PG_DSN')


@unittest.skipUnless(DSN, 'RAGSHIELD_TEST_PG_DSN is not set')
class PgVectorTests(Contract, unittest.TestCase):
    def make_index(self):
        import psycopg
        from ragshield.adapters import PgVectorIndex
        self.conn = psycopg.connect(DSN, autocommit=True)
        self.conn.execute('set enable_seqscan = off')  # small tables skip the index otherwise, and the index is what limits a scan
        self.table = f't{uuid.uuid4().hex}'
        self.addCleanup(self.conn.close)
        self.addCleanup(lambda: self.conn.execute(f'drop table if exists {self.table}'))
        return PgVectorIndex(self.conn, self.table, dim=256)

    def change_text(self, chunk_id, text):
        self.conn.execute(f'update {self.table} set text = %s where chunk_id = %s', (text, chunk_id))

    def write_around_the_gate(self, chunk_id, text, vector):
        self.conn.execute(f"insert into {self.table} values (%s, 'planted', %s, '{{}}', %s)", (chunk_id, text, vector))

    def test_a_transaction_on_the_connection_is_not_left_open(self):
        import psycopg
        from ragshield.adapters import PgVectorIndex
        with psycopg.connect(DSN) as conn:  # not autocommit
            index = PgVectorIndex(conn, self.table, dim=256)
            self.assertIn('vpn-runbook::0', index.ids)
            self.assertEqual(conn.info.transaction_status, psycopg.pq.TransactionStatus.IDLE)


if __name__ == '__main__':
    unittest.main()
