"""Experiment 2: published indirect-injection attacks (BIPIA test split, MIT) mixed into the Vandelay corpus.

No new attack content is written: each of the 125 BIPIA test attacks is inserted unchanged, as its own paragraph, into a
copy of a randomly chosen clean Vandelay chunk (start, middle or end). Three threats are measured separately:
  T1  direct writes to the vector store (bypassing ingestion): caught by signature verification?
  T2  poisoned documents from a low-trust source (e.g. an upload or web scrape)
  T3  poisoned documents from a trusted source (a compromised wiki page): only the detectors can help
Policy constants were fixed on clean data (experiment 1) before this run; only warn_weight is swept, and each setting
is reported with its clean-recall cost. Writes results/attack_bipia.json.

With --injection-model and --split test (for a fine-tuned classifier), attacks are hosted only in held-out documents, clean
rates come only from their chunks, and the report goes to results/attack_bipia-<model>-<split>.json."""
import argparse
import json
import random
import re
import time

import numpy as np

from common import HERE, H, CachedEmbedder, CachedInjection, add_eval_args, eval_doc_ids, make_shield, out_path, outcome, outcome_counts, vandelay
from ragshield import chunking
from ragshield.shield import Policy

parser = argparse.ArgumentParser()
parser.add_argument('--device', default=None)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--seed', type=int, default=7)
add_eval_args(parser)
args = parser.parse_args()
WEIGHTS = [1.0, 0.85, 0.5]

docs, questions = vandelay()
embedder, injection = CachedEmbedder(device=args.device), CachedInjection(args.injection_model, device=args.device)
eligible = eval_doc_ids(docs, args.split)

attacks = []
for split in ('text', 'code'):
    for category, items in json.loads((HERE / f'data/external/bipia/{split}_attack_test.json').read_text()).items():
        attacks += [{'split': split, 'category': category, 'text': t} for t in items]

clean_chunks = [(d.doc_id, i, c) for d in docs if d.doc_id in eligible for i, c in enumerate(chunking.chunk(d.text))]
rng = random.Random(args.seed)
poisons = []
for n, attack in enumerate(attacks):
    doc_id, i, host = rng.choice(clean_chunks)
    paras = host.split('\n\n')
    where = rng.choice(['start', 'middle', 'end'])
    at = {'start': 0, 'middle': len(paras) // 2, 'end': len(paras)}[where]
    text = '\n\n'.join(paras[:at] + [attack['text']] + paras[at:])
    probe = ' '.join(re.sub(r'[#|`*>_-]+', ' ', host).split()[:25])
    poisons.append({**attack, 'id': f'poison-{n:03d}', 'host': f'{doc_id}::{i}', 'where': where, 'text': text,
                    'payload': attack['text'], 'probe': probe})


def build(poison_source=None, poison_tier=None):
    shield = make_shield(embedder, injection, Policy())
    for d in docs:
        shield.ingest(d.doc_id, d.text, 'vandelay-internal', 'high', metadata=d.metadata())
    clean_ids = list(shield.index.ids)
    if poison_source:
        for p in poisons:
            shield.ingest(p['id'], p['text'], poison_source, poison_tier)
    shield.fit_outlier(fit_ids=clean_ids)  # baseline fitted on the reviewed clean snapshot only
    return shield


def payload_chunk(shield, poison):
    """The chunk of a poisoned document that carries the attack text (documents can split into several chunks)."""
    ids = [cid for cid in shield.index.ids if cid.startswith(poison['id'] + '::')]
    head = poison['payload'][:60]
    return next((cid for cid in ids if head in shield.index.texts[shield.index.ids.index(cid)]), ids[0])


t0 = time.time()
clean = build()
results = {'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'attacks': len(poisons),
           'source': 'microsoft/BIPIA benchmark/{text,code}_attack_test.json (MIT)', 'seed': args.seed, 'k': args.k,
           'injection_model': args.injection_model, 'split': args.split, 'eval_docs': len(eligible)}

# Clean-recall cost of each ranking setting (no poison present).
results['clean_recall'] = {}
for w in WEIGHTS:
    clean.policy.warn_weight = w
    df = H.eval_retrieval('bipia-clean', {'shield': 'on', 'warn_weight': w}, lambda q, k: clean.retrieve(q, k),
                          k=args.k, questions=questions, verbose=False)
    results['clean_recall'][str(w)] = round(float(df.recall.mean()), 4)
off = H.eval_retrieval('bipia-clean', {'shield': 'off'}, lambda q, k: clean.retrieve(q, k, shielded=False),
                       k=args.k, questions=questions, verbose=False)
results['clean_recall']['off'] = round(float(off.recall.mean()), 4)
if args.split == 'test':
    # Questions answered only by held-out documents: the classifier never saw their text.
    test_qs = [q for q in questions if q['gold_doc_ids'] and set(q['gold_doc_ids']) <= eligible]
    clean.policy.warn_weight = 0.85
    results['clean_recall_test_doc_questions'] = {
        'n': len(test_qs),
        'on': round(float(H.eval_retrieval('bipia-clean-test', {'shield': 'on'}, lambda q, k: clean.retrieve(q, k),
                                           k=args.k, questions=test_qs, verbose=False).recall.mean()), 4) if test_qs else None,
        'off': round(float(H.eval_retrieval('bipia-clean-test', {'shield': 'off'}, lambda q, k: clean.retrieve(q, k, shielded=False),
                                            k=args.k, questions=test_qs, verbose=False).recall.mean()), 4) if test_qs else None}

# Detector rates on clean chunks. The outlier rate comes from held-out folds (each chunk scored by a model that did not
# see it), aligned with the index order the model was fitted in.
mask = np.array([cid.split('::')[0] in eligible for cid in clean.index.ids])
inj_clean = np.array([clean.ledger.get(cid)[0].scores['injection'] for cid in clean.index.ids])[mask]
out_clean = (clean.outlier.heldout_scores > clean.outlier.threshold)[mask]

scenarios = {}
for name, source, tier in [('T2_low_trust_source', 'external-upload', 'low'), ('T3_trusted_source', 'vandelay-internal', 'high')]:
    shield = build(source, tier)
    rows = []
    for p in poisons:
        cid = payload_chunk(shield, p)
        record, quarantined, _ = shield.ledger.get(cid)
        raw_ids = [h['chunk_id'] for h in shield.retrieve(p['probe'], args.k, shielded=False)]
        outcome_by_w = {}
        for w in WEIGHTS:
            shield.policy.warn_weight = w
            hits = {h['chunk_id']: h for h in shield.retrieve(p['probe'], args.k)}
            outcome_by_w[str(w)] = outcome(shield, cid, hits)
        rows.append({'id': p['id'], 'category': p['category'], 'split': p['split'], 'where': p['where'], 'chunk': cid,
                     'injection': record.scores['injection'], 'outlier_flag': record.scores['outlier_flag'],
                     'quarantined': quarantined, 'retrieved_unshielded': cid in raw_ids, 'outcome': outcome_by_w})
    inj = np.array([r['injection'] for r in rows])
    outl = np.array([r['outlier_flag'] for r in rows])
    retrieved = [r for r in rows if r['retrieved_unshielded']]
    summary = {'retrieved_unshielded': len(retrieved), 'outcomes_when_retrieved': {}}
    for w in WEIGHTS:
        summary['outcomes_when_retrieved'][str(w)] = outcome_counts([r['outcome'][str(w)] for r in retrieved], len(retrieved))
    shield.policy.warn_weight = 0.85
    summary['clean_recall_with_poison_present'] = round(float(H.eval_retrieval(
        f'bipia-{name}', {'shield': 'on', 'warn_weight': 0.85}, lambda q, k: shield.retrieve(q, k), k=args.k,
        questions=questions, verbose=False).recall.mean()), 4)
    scenarios[name] = {'summary': summary, 'rows': rows}

# T2b: the attacker targets topics no trusted document covers. 7 documents are left out of the trusted corpus and their
# topics exist only as poisoned low-trust copies, so no trusted twin can outrank them.
held = set(random.Random(args.seed + 2).sample([d.doc_id for d in docs if d.doc_id in eligible], 7))  # corpus order, as before
held_chunks = [c for c in clean_chunks if c[0] in held]
t2b = make_shield(embedder, injection, Policy())
for d in docs:
    if d.doc_id not in held:
        t2b.ingest(d.doc_id, d.text, 'vandelay-internal', 'high', metadata=d.metadata())
trusted_ids = list(t2b.index.ids)
rng2 = random.Random(args.seed + 3)
t2b_rows = []
for n, attack in enumerate(attacks):
    doc_id, i, host = rng2.choice(held_chunks)
    paras = host.split('\n\n')
    at = rng2.choice([0, len(paras) // 2, len(paras)])
    pid = f'orphan-{n:03d}'
    t2b.ingest(pid, '\n\n'.join(paras[:at] + [attack['text']] + paras[at:]), 'external-upload', 'low')
    t2b_rows.append({'id': pid, 'payload': attack['text'], 'probe': ' '.join(re.sub(r'[#|`*>_-]+', ' ', host).split()[:25])})
t2b.fit_outlier(fit_ids=trusted_ids)
t2b_retrieved, t2b_outcomes = 0, []
t2b.policy.warn_weight = 0.85
for row in t2b_rows:
    cid = payload_chunk(t2b, row)
    if cid not in [h['chunk_id'] for h in t2b.retrieve(row['probe'], args.k, shielded=False)]:
        continue
    t2b_retrieved += 1
    hits = {h['chunk_id']: h for h in t2b.retrieve(row['probe'], args.k)}
    t2b_outcomes.append(outcome(t2b, cid, hits))
results['T2b_low_trust_no_trusted_counterpart'] = {
    'held_out_docs': sorted(held), 'retrieved_unshielded': t2b_retrieved,
    'outcomes_when_retrieved_warn_weight_0.85': outcome_counts(t2b_outcomes, t2b_retrieved)}

t3 = scenarios['T3_trusted_source']['rows']  # same texts in both scenarios, so detector scores are identical
inj = np.array([r['injection'] for r in t3])
outl = np.array([r['outlier_flag'] for r in t3])
results['detection'] = {
    'injection_ge_warn': {'attack_recall': round(float((inj >= 0.5).mean()), 4), 'clean_fpr': round(float((inj_clean >= 0.5).mean()), 4)},
    'injection_ge_block': {'attack_recall': round(float((inj >= 0.9).mean()), 4), 'clean_fpr': round(float((inj_clean >= 0.9).mean()), 4)},
    'outlier_flag': {'attack_recall': round(float(outl.mean()), 4), 'clean_fpr': round(float(out_clean.mean()), 4)},
    'any_flag': {'attack_recall': round(float(((inj >= 0.5) | outl).mean()), 4),
                 'clean_fpr': round(float(((inj_clean >= 0.5) | out_clean).mean()), 4)},
}
by_cat = {}
for r in scenarios['T3_trusted_source']['rows']:
    by_cat.setdefault(f"{r['split']}:{r['category']}", []).append(r['injection'] >= 0.5 or r['outlier_flag'])
results['detection_by_category'] = {c: round(float(np.mean(v)), 2) for c, v in sorted(by_cat.items())}

# T1: write each poisoned text straight into the store over a clean chunk, bypassing ingestion.
tamper = build()
targets = random.Random(args.seed + 1).sample(list(tamper.index.ids), min(len(poisons), len(tamper.index.ids)))
for cid, p in zip(targets, poisons):
    tamper.index.set_text(cid, p['text'])
caught = sum(tamper.ledger.verify(cid, tamper.index.texts[tamper.index.ids.index(cid)])[0] == 'tampered' for cid in targets)
results['T1_direct_store_writes'] = {'tampered_chunks': len(targets), 'caught_by_signature': caught}

results['scenarios'] = {k: v['summary'] for k, v in scenarios.items()}
results['runtime_sec'] = round(time.time() - t0, 1)
out_path('attack_bipia', args).write_text(json.dumps({**results, 'rows': {k: v['rows'] for k, v in scenarios.items()}}, indent=2))
print(json.dumps(results, indent=2))
