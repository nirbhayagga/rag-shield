"""Content detectors run at ingestion. They score; the policy in shield.py decides what happens."""
import numpy as np

INJECTION_MODEL = 'protectai/deberta-v3-base-prompt-injection-v2'  # Apache-2.0; Llama Prompt Guard 2 is an optional gated alternative


class InjectionDetector:
    """Probability that a chunk contains instructions aimed at the model. Long chunks are scored in windows; the max wins."""

    def __init__(self, model=INJECTION_MODEL, device=None, window_chars=1500):
        from transformers import pipeline
        self.pipe = pipeline('text-classification', model=model, device=device, truncation=True, max_length=512, top_k=None)
        self.name = model
        self.window = window_chars

    def _windows(self, text):
        if len(text) <= self.window:
            return [text]
        step = self.window // 2
        return [text[i:i + self.window] for i in range(0, len(text) - step, step)]

    def score(self, texts, batch=16):
        flat, owner = [], []
        for i, text in enumerate(texts):
            for w in self._windows(text):
                flat.append(w)
                owner.append(i)
        out = np.zeros(len(texts), dtype=np.float32)
        for labels, i in zip(self.pipe(flat, batch_size=batch), owner):
            p = next((d['score'] for d in labels if d['label'].upper() in ('INJECTION', 'LABEL_1', 'MALICIOUS')), 0.0)
            out[i] = max(out[i], p)
        return out


class OutlierDetector:
    """Local Outlier Factor over embeddings of trusted content. The threshold is set from held-out trusted chunks
    (k-fold), so the false-positive rate on clean data is estimated honestly rather than on the fitting data."""

    def __init__(self, n_neighbors=10, target_fpr=0.02, folds=5, seed=0):
        self.n_neighbors, self.target_fpr, self.folds, self.seed = n_neighbors, target_fpr, folds, seed
        self.model, self.threshold, self.heldout_scores = None, None, None

    def _lof(self, X):
        from sklearn.neighbors import LocalOutlierFactor
        return LocalOutlierFactor(n_neighbors=min(self.n_neighbors, len(X) - 1), novelty=True, metric='cosine').fit(X)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(X))
        held = np.zeros(len(X), dtype=np.float32)
        for fold in np.array_split(order, self.folds):
            rest = np.setdiff1d(order, fold)
            held[fold] = -self._lof(X[rest]).score_samples(X[fold])
        self.heldout_scores = held
        self.threshold = float(np.quantile(held, 1 - self.target_fpr))
        self.model = self._lof(X)
        return self

    def score(self, X):
        return -self.model.score_samples(np.asarray(X, dtype=np.float32))

    def flag(self, X):
        return self.score(X) > self.threshold
