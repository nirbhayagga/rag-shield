"""Keep chunks in a vector database you already run. Each class has the same methods as index.VectorIndex, so pass one to
Shield in its place.

The database isn't trusted: Shield still checks every chunk's text against the signed ledger, so text written or changed
there directly is dropped at retrieval. Existing content has to go through Shield.ingest once, so that it gets signed.

Install with: pip install "rag-shield[chroma]" or "rag-shield[pgvector]"."""
import json

import numpy as np


def _flat(metadata):
    """ChromaDB only stores scalar metadata: drop None and keep anything else as a JSON string."""
    return {k: v if isinstance(v, (str, int, float, bool)) else json.dumps(v) for k, v in metadata.items() if v is not None}


class ChromaIndex:
    """A ChromaDB collection using cosine distance: ChromaIndex(chromadb.PersistentClient('db'), 'docs')."""

    def __init__(self, client, name='ragshield'):
        self.collection = client.get_or_create_collection(name, metadata={'hnsw:space': 'cosine'}, embedding_function=None)
        space = (self.collection.metadata or {}).get('hnsw:space')
        if space != 'cosine':
            raise ValueError(f'collection {name!r} uses {space} distance; RAG Shield needs cosine')

    def add(self, chunk_id, text, vector, metadata):
        self.collection.upsert(ids=[chunk_id], embeddings=[np.asarray(vector, dtype=float).tolist()], documents=[text], metadatas=[_flat(metadata)])

    def search(self, vector, k):
        n = min(k, self.collection.count())
        if not n:
            return []
        got = self.collection.query(query_embeddings=[np.asarray(vector, dtype=float).tolist()], n_results=n,
                                    include=['documents', 'metadatas', 'distances'])
        return [{'chunk_id': i, 'doc_id': m['doc_id'], 'text': t, 'metadata': m, 'score': 1.0 - float(d)}
                for i, t, m, d in zip(got['ids'][0], got['documents'][0], got['metadatas'][0], got['distances'][0])]

    def remove_doc(self, doc_id):
        self.collection.delete(where={'doc_id': doc_id})

    # Sorted by chunk id, so ids, texts and matrix line up across separate calls.
    @property
    def ids(self):
        return sorted(self.collection.get(include=[])['ids'])

    @property
    def texts(self):
        got = self.collection.get(include=['documents'])
        return [got['documents'][i] for i in np.argsort(got['ids'])]

    @property
    def matrix(self):
        got = self.collection.get(include=['embeddings'])
        if not len(got['ids']):
            return np.zeros((0, 1), np.float32)
        return np.asarray(got['embeddings'], dtype=np.float32)[np.argsort(got['ids'])]


class PgVectorIndex:
    """A Postgres table using pgvector. conn is a psycopg 3 connection; dim must match the embedder (768 for the default)."""

    # hnsw.ef_search caps how many rows an index scan returns (40 by default, 1000 at most). Left alone, a fetch of more
    # than 40 looks like the end of the table, and Shield.retrieve would stop looking past a flood of dropped chunks.
    MAX_EF_SEARCH = 1000

    def __init__(self, conn, table='ragshield_chunks', dim=768):
        from pgvector.psycopg import register_vector
        from psycopg import sql
        self.conn, self.sql = conn, sql
        self.table = sql.Identifier(table)
        with conn.transaction():
            conn.execute('create extension if not exists vector')
        register_vector(conn)
        with conn.transaction():
            conn.execute(sql.SQL('create table if not exists {t} (chunk_id text primary key, doc_id text not null, text text not null, '
                                 'metadata jsonb not null, embedding vector({d}) not null)').format(t=self.table, d=sql.Literal(dim)))
            conn.execute(sql.SQL('create index if not exists {i} on {t} (doc_id)').format(i=sql.Identifier(f'{table}_doc_idx'), t=self.table))
            conn.execute(sql.SQL('create index if not exists {i} on {t} using hnsw (embedding vector_cosine_ops)').format(
                i=sql.Identifier(f'{table}_embedding_idx'), t=self.table))

    def _query(self, template, params=()):
        with self.conn.transaction():
            return self.conn.execute(self.sql.SQL(template).format(t=self.table), params).fetchall()

    def add(self, chunk_id, text, vector, metadata):
        from psycopg.types.json import Jsonb
        with self.conn.transaction():
            self.conn.execute(self.sql.SQL('insert into {t} (chunk_id, doc_id, text, metadata, embedding) values (%s, %s, %s, %s, %s) '
                                           'on conflict (chunk_id) do update set doc_id = excluded.doc_id, text = excluded.text, '
                                           'metadata = excluded.metadata, embedding = excluded.embedding').format(t=self.table),
                              (chunk_id, metadata['doc_id'], text, Jsonb(metadata), np.asarray(vector, dtype=np.float32)))

    def search(self, vector, k):
        v = np.asarray(vector, dtype=np.float32)
        with self.conn.transaction():
            self.conn.execute("select set_config('hnsw.ef_search', %s, true)", (str(min(max(k, 40), self.MAX_EF_SEARCH)),))
            rows = self.conn.execute(self.sql.SQL('select chunk_id, doc_id, text, metadata, 1 - (embedding <=> %s) from {t} '
                                                  'order by embedding <=> %s limit %s').format(t=self.table), (v, v, k)).fetchall()
        return [{'chunk_id': c, 'doc_id': d, 'text': t, 'metadata': m, 'score': float(s)} for c, d, t, m, s in rows]

    def remove_doc(self, doc_id):
        with self.conn.transaction():
            self.conn.execute(self.sql.SQL('delete from {t} where doc_id = %s').format(t=self.table), (doc_id,))

    @property
    def ids(self):
        return [r[0] for r in self._query('select chunk_id from {t} order by chunk_id')]

    @property
    def texts(self):
        return [r[0] for r in self._query('select text from {t} order by chunk_id')]

    @property
    def matrix(self):
        rows = self._query('select embedding from {t} order by chunk_id')
        return np.vstack([r[0].to_numpy() for r in rows], dtype=np.float32) if rows else np.zeros((0, 1), np.float32)
