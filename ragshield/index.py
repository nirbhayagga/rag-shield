"""A minimal embedding index (cosine over normalized vectors) and the embedder used for retrieval and outlier scoring.

The adapter boundary is deliberately small (add / search / get / set_text) so ChromaDB and pgvector adapters can follow."""
import numpy as np

DEFAULT_MODEL = 'BAAI/bge-base-en-v1.5'  # the model the class harness serves for embeddings
QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '


class Embedder:
    def __init__(self, model=DEFAULT_MODEL, device=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, device=device)
        self.name = model

    def passages(self, texts, batch=32):
        return self.model.encode(list(texts), batch_size=batch, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    def query(self, text):
        return self.passages([QUERY_PREFIX + text])[0]


class VectorIndex:
    """Stores chunk text and metadata next to vectors, like a vector database does."""

    def __init__(self):
        self.ids, self.texts, self.meta, self._vectors = [], [], [], []
        self._matrix = None

    def add(self, chunk_id, text, vector, metadata):
        self.ids.append(chunk_id)
        self.texts.append(text)
        self.meta.append(dict(metadata))
        self._vectors.append(np.asarray(vector, dtype=np.float32))
        self._matrix = None

    @property
    def matrix(self):
        if self._matrix is None:
            self._matrix = np.vstack(self._vectors) if self._vectors else np.zeros((0, 1), np.float32)
        return self._matrix

    def search(self, vector, k):
        if not self.ids:
            return []
        scores = self.matrix @ np.asarray(vector, dtype=np.float32)
        top = np.argsort(-scores)[:k]
        return [{'chunk_id': self.ids[i], 'doc_id': self.meta[i]['doc_id'], 'text': self.texts[i],
                 'metadata': self.meta[i], 'score': float(scores[i])} for i in top]

    def remove_doc(self, doc_id):
        keep = [i for i, m in enumerate(self.meta) if m['doc_id'] != doc_id]
        self.ids = [self.ids[i] for i in keep]
        self.texts = [self.texts[i] for i in keep]
        self.meta = [self.meta[i] for i in keep]
        self._vectors = [self._vectors[i] for i in keep]
        self._matrix = None

    def set_text(self, chunk_id, text):
        """Simulates a direct write to the store that bypasses ingestion (for tamper tests)."""
        self.texts[self.ids.index(chunk_id)] = text
