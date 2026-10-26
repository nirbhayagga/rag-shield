"""Experiment 1: what does RAG Shield cost on clean data?

On the class Vandelay corpus (no attack content), with the class harness's own retrieval metric:
  - recall@5 and MRR, unshielded vs shielded (the filter must not hurt honest retrieval)
  - false-positive rate of each detector on clean chunks (outlier rate from held-out folds)
  - added latency per query, and ingestion cost per chunk
Writes results/clean_baseline.json (or the model/split-specific name, see common.out_path).
Usage: .venv/bin/python experiments/clean_baseline.py [--device cuda] [--injection-model ...] [--split test]"""
import argparse
import json
import statistics
import time

import numpy as np

from common import H, CachedEmbedder, CachedInjection, add_eval_args, eval_doc_ids, make_shield, out_path, vandelay

parser = argparse.ArgumentParser()
parser.add_argument('--device', default=None)
parser.add_argument('--k', type=int, default=5)
add_eval_args(parser)
args = parser.parse_args()

docs, questions = vandelay()
eligible = eval_doc_ids(docs, args.split)
embedder, injection = CachedEmbedder(device=args.device), CachedInjection(args.injection_model, device=args.device)

# Same build as the attack experiments: ingest everything as trusted, then fit the outlier model on that clean snapshot,
# so the three result files are comparable.
shield = make_shield(embedder, injection)
t0 = time.time()
for d in docs:
    shield.ingest(d.doc_id, d.text, source_id='vandelay-internal', trust_tier='high', metadata=d.metadata())
ingest_sec = time.time() - t0
n_chunks = len(shield.index.ids)
flagged = shield.fit_outlier(fit_ids=list(shield.index.ids))

mask = np.array([cid.split('::')[0] in eligible for cid in shield.index.ids])
records = [shield.ledger.get(cid)[0] for cid in shield.index.ids]
inj = np.array([r.scores['injection'] for r in records])[mask]
heldout = (shield.outlier.heldout_scores > shield.outlier.threshold)[mask]

raw = H.eval_retrieval('clean-baseline', {'pipeline': 'dense', 'shield': 'off'},
                       lambda q, k: shield.retrieve(q, k, shielded=False), k=args.k, questions=questions, verbose=False)
shielded = H.eval_retrieval('clean-baseline', {'pipeline': 'dense', 'shield': 'on'},
                            lambda q, k: shield.retrieve(q, k), k=args.k, questions=questions, verbose=False)


def latency(shielded_flag, reps=3):
    times = []
    for _ in range(reps):
        for q in questions:
            t = time.perf_counter()
            shield.retrieve(q['question'], args.k, shielded=shielded_flag)
            times.append(time.perf_counter() - t)
    return statistics.median(times) * 1000, float(np.percentile(times, 95)) * 1000


raw_p50, raw_p95 = latency(False)
sh_p50, sh_p95 = latency(True)
changed = [q for q, a, b in zip(raw['id'], raw['retrieved'], shielded['retrieved']) if a != b]

result = {
    'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'device': args.device or 'cpu',
    'injection_model': args.injection_model, 'split': args.split, 'eval_docs': len(eligible),
    'corpus': {'documents': len(docs), 'chunks': n_chunks, 'chunk_size': 800, 'overlap': 100},
    'models': {'embedder': embedder.name, 'injection': injection.name},
    'retrieval': {'k': args.k, 'questions': int(len(raw)),
                  'recall_off': round(float(raw.recall.mean()), 4), 'recall_on': round(float(shielded.recall.mean()), 4),
                  'mrr_off': round(float(raw.mrr.mean()), 4), 'mrr_on': round(float(shielded.mrr.mean()), 4),
                  'questions_with_different_docs': changed},
    'false_positives_on_clean': {
        'chunks_scored': int(mask.sum()),
        'injection_block_rate': round(float((inj >= shield.policy.injection_block).mean()), 4),
        'injection_warn_rate': round(float((inj >= shield.policy.injection_warn).mean()), 4),
        'injection_max': round(float(inj.max()), 4),
        'outlier_flag_rate_fit': round(flagged / n_chunks, 4), 'outlier_flag_rate_heldout': round(float(heldout.mean()), 4),
        'chunks_dropped_or_warned': sorted({e[1] for e in shield.events}),
    },
    'latency_ms': {'unshielded_p50': round(raw_p50, 2), 'unshielded_p95': round(raw_p95, 2),
                   'shielded_p50': round(sh_p50, 2), 'shielded_p95': round(sh_p95, 2),
                   'added_p50': round(sh_p50 - raw_p50, 2)},
    'ingestion': {'total_sec': round(ingest_sec, 1), 'ms_per_chunk': round(ingest_sec / n_chunks * 1000, 1),
                  'note': 'model calls are disk-cached (results/cache), so this is not a cold-start cost'},
    'warned_chunks': [{'chunk_id': r.chunk_id, **r.scores} for r in records
                      if r.scores.get('outlier_flag') or r.scores['injection'] >= shield.policy.injection_warn],
}
out = out_path('clean_baseline', args)
out.write_text(json.dumps(result, indent=2))
print(json.dumps({k: result[k] for k in ('retrieval', 'false_positives_on_clean', 'latency_ms', 'ingestion')}, indent=2))
