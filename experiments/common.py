"""Shared setup for experiments: the class corpus and harness, and disk-cached model calls (keyed by model + text hash)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'vendor')]

import eval_harness as H  # noqa: E402  (private, not distributed)
from ragshield.detectors import INJECTION_MODEL, OutlierDetector  # noqa: E402
from ragshield.index import DEFAULT_MODEL, VectorIndex  # noqa: E402
from ragshield.provenance import Ledger, Signer, sha256  # noqa: E402
from ragshield.shield import Policy, Shield  # noqa: E402

H.RESULTS_DIR = HERE / 'results'
H.RESULTS_DIR.mkdir(exist_ok=True)  # gitignored, so missing in a fresh clone
CACHE = HERE / 'results/cache'


def vandelay():
    return H.load_corpus(HERE / 'data/vandelay/corpus'), H.load_questions(HERE / 'data/vandelay/questions.yaml')


class CachedEmbedder:
    def __init__(self, model=DEFAULT_MODEL, device=None):
        self.name, self.device, self._model = model, device, None
        self.path = CACHE / f"emb-{model.replace('/', '_')}.npz"
        self.cache = dict(np.load(self.path)) if self.path.exists() else {}

    def _real(self):
        if self._model is None:
            from ragshield.index import Embedder
            self._model = Embedder(self.name, self.device)
        return self._model

    def passages(self, texts):
        keys = [sha256(t) for t in texts]
        missing = [t for t, k in zip(texts, keys) if k not in self.cache]
        if missing:
            for t, v in zip(missing, self._real().passages(missing)):
                self.cache[sha256(t)] = v
            self.save()
        return np.vstack([self.cache[k] for k in keys])

    def query(self, text):
        return self.passages(['Represent this sentence for searching relevant passages: ' + text])[0]

    def save(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, **self.cache)


class CachedInjection:
    def __init__(self, model=INJECTION_MODEL, device=None):
        self.name, self.device, self._model = model, device, None
        self.path = CACHE / f"inj-{model.replace('/', '_')}.json"
        self.cache = json.loads(self.path.read_text()) if self.path.exists() else {}

    def score(self, texts):
        keys = [sha256(t) for t in texts]
        missing = [t for t, k in zip(texts, keys) if k not in self.cache]
        if missing:
            if self._model is None:
                from ragshield.detectors import InjectionDetector
                self._model = InjectionDetector(self.name, self.device)
            for t, s in zip(missing, self._model.score(missing)):
                self.cache[sha256(t)] = float(s)
            CACHE.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.cache))
        return np.array([self.cache[k] for k in keys], dtype=np.float32)


def make_shield(embedder, injection, policy=None, target_fpr=0.02):
    return Shield(Ledger(), Signer.generate(), embedder, VectorIndex(), injection=injection,
                  outlier=OutlierDetector(target_fpr=target_fpr), policy=policy or Policy())
