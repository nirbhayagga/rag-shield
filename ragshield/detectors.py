"""Content detectors run at ingestion. They score; the policy in shield.py decides what happens."""
import numpy as np

INJECTION_MODEL = 'nirbhayagga/ragshield-injection-v5'  # Apache-2.0; fine-tuned on public data only (see its model card)
OFF_THE_SHELF_MODEL = 'protectai/deberta-v3-base-prompt-injection-v2'  # Apache-2.0; v5's base, and the experiments' baseline
CONTRADICTION_MODEL = 'cross-encoder/nli-deberta-v3-base'  # Apache-2.0


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


class ContradictionDetector:
    """Probability that one passage contradicts another, from a natural-language-inference (NLI) model.

    The shield uses it on edits. A planted false statement ("gateway GW-07 listens on port 8443" on a page that says 443)
    carries no instruction, so the injection classifier passes it; this is the check that can hold it. See
    Shield._contradictions for which sentences are compared."""

    def __init__(self, model=CONTRADICTION_MODEL, device=None):
        from transformers import pipeline
        self.pipe = pipeline('text-classification', model=model, device=device, truncation=True, max_length=512, top_k=None)
        self.name = model

    def score(self, pairs, batch=16):
        """pairs: (premise, hypothesis). Returns an array with one row per pair: the probability that the hypothesis
        contradicts the premise, and the probability that the premise supports (entails) it."""
        if not pairs:
            return np.zeros((0, 2), dtype=np.float32)
        out = self.pipe([{'text': premise, 'text_pair': hypothesis} for premise, hypothesis in pairs], batch_size=batch)
        prob = lambda labels, name: next((d['score'] for d in labels if d['label'].lower().startswith(name)), 0.0)
        return np.array([[prob(labels, 'contradict'), prob(labels, 'entail')] for labels in out], dtype=np.float32)


class CombinedInjection:
    """Several classifiers as one: the highest score wins, so a chunk is flagged if any of them fires."""

    def __init__(self, detectors):
        self.detectors = list(detectors)
        self.name = '+'.join(d.name for d in self.detectors)

    def score(self, texts):
        return np.max([d.score(texts) for d in self.detectors], axis=0)


class OutlierDetector:
    """Local Outlier Factor over embeddings of trusted content. The threshold is set from held-out trusted chunks
    (k-fold), so the false-positive rate on clean data is estimated honestly rather than on the fitting data."""

    def __init__(self, n_neighbors=10, target_fpr=0.02, folds=5, seed=0):
        self.n_neighbors, self.target_fpr, self.folds, self.seed = n_neighbors, target_fpr, folds, seed
        self.model, self.threshold, self.heldout_scores = None, None, None

    def _lof(self, X):
        from sklearn.neighbors import LocalOutlierFactor
        return LocalOutlierFactor(n_neighbors=max(1, min(self.n_neighbors, len(X) - 1)), novelty=True, metric='cosine').fit(X)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        if len(X) < 2:  # LOF needs at least two points; with less, nothing is an outlier
            self.model, self.threshold, self.heldout_scores = None, float('inf'), np.zeros(len(X), dtype=np.float32)
            return self
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(X))
        held = np.zeros(len(X), dtype=np.float32)
        folds = self.folds if len(X) >= 2 * self.folds else 1  # too few trusted chunks for held-out folds: fit-set scores
        for fold in np.array_split(order, folds):
            rest = np.setdiff1d(order, fold) if folds > 1 else order
            held[fold] = -self._lof(X[rest]).score_samples(X[fold])
        self.heldout_scores = held
        self.threshold = float(np.quantile(held, 1 - self.target_fpr))
        self.model = self._lof(X)
        return self

    def score(self, X):
        X = np.asarray(X, dtype=np.float32)
        return -self.model.score_samples(X) if self.model is not None else np.zeros(len(X), dtype=np.float32)

    def flag(self, X):
        return self.score(X) > self.threshold
