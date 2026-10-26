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


def add_eval_args(parser):
    parser.add_argument('--injection-model', default=INJECTION_MODEL, help='Hugging Face id or local path (e.g. models/ragshield-injection-v1)')
    parser.add_argument('--split', choices=['all', 'test'], default='all',
                        help='test: attack hosts, edited pages and clean-rate chunks come only from the held-out documents in data/splits.json')


def eval_doc_ids(docs, split):
    """Documents whose chunks may host attacks or count as clean negatives. A fine-tuned classifier saw train_docs, so it
    must be scored on test_docs only."""
    if split == 'all':
        return {d.doc_id for d in docs}
    return set(json.loads((HERE / 'data/splits.json').read_text())['test_docs'])


def model_tag(model):
    return Path(model).name if Path(model).is_dir() else 'base'


def out_path(stem, args):
    """results/<stem>.json for the original setup, results/<stem>-<model>-<split>.json otherwise."""
    if args.injection_model == INJECTION_MODEL and args.split == 'all':
        return HERE / f'results/{stem}.json'
    return HERE / f'results/{stem}-{model_tag(args.injection_model)}-{args.split}.json'


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
        key = model.replace('/', '_')
        if Path(model).is_dir():  # a local fine-tune: key the cache by its weights, so retraining invalidates it
            weights = Path(model) / 'model.safetensors'
            key = f'{Path(model).name}-{sha256(str(weights.stat().st_size) + str(weights.stat().st_mtime_ns))[:12]}'
        self.path = CACHE / f'inj-{key}.json'
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


OUTCOMES = ('served_silently', 'served_with_warning', 'dropped', 'outranked')


def outcome(shield, cid, hits):
    """What happened to chunk cid under the shield: served (silently or with a warning), dropped by the filter, or
    outranked (kept, but pushed below the top k by the trust weight). 'suppressed' in older results = dropped + outranked."""
    if cid in hits:
        return 'served_silently' if hits[cid]['trust']['action'] == 'pass' else 'served_with_warning'
    text = shield.index.texts[shield.index.ids.index(cid)]
    return 'dropped' if shield.decide({'chunk_id': cid, 'text': text})['action'] == 'drop' else 'outranked'


def outcome_counts(outcomes, n):
    counts = {o: sum(x == o for x in outcomes) for o in OUTCOMES}
    counts['suppressed'] = counts['dropped'] + counts['outranked']
    counts['silent_attack_success_rate'] = round(counts['served_silently'] / max(1, n), 4)
    return counts


def make_shield(embedder, injection, policy=None, target_fpr=0.02):
    return Shield(Ledger(), Signer.generate(), embedder, VectorIndex(), injection=injection,
                  outlier=OutlierDetector(target_fpr=target_fpr), policy=policy or Policy())
