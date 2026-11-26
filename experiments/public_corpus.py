"""Experiment 6: does RAG Shield hold up on public text unlike the Vandelay corpus?

Corpora (fetch first with fetch_corpus.py; neither was used to build, train or tune anything):
  k8s     Kubernetes task docs (CC BY 4.0): a runbook-style wiki full of shell commands and imperative steps
  attack  MITRE ATT&CK Enterprise technique pages: security text that describes attacker behavior
Same detectors and shipped policy as on the Vandelay corpus:
  1. False alarms on clean chunks: injection score >= 0.5 (served with a warning) and >= 0.9, outlier on held-out folds,
     hidden characters, and the share of chunks that would carry any warning (the review workload).
  2. BIPIA test attacks (MIT, unchanged) placed in clean chunks, scored as whole chunks (as in experiment 2).
  3. Delta scoring on page edits: BIPIA attacks vs benign edits that add text from another page of the same section
     (as in experiment 3), at the shipped thresholds (does the Vandelay calibration transfer?) and recalibrated on
     half of this corpus's benign edits, reported on the other half.
On a CPU, --max-chunks keeps a seeded sample of whole pages (default 400 chunks); --max-chunks 0 uses everything.
Writes results/public_corpus-<corpus>-<model>.json."""
import argparse
import json
import random
import time

import numpy as np

from common import HERE, PUBLIC_CORPORA, CachedEmbedder, CachedInjection, load_public, make_shield, model_tag
from ragshield import chunking, delta, hidden
from ragshield.detectors import OFF_THE_SHELF_MODEL
from ragshield.shield import Policy

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--corpus', choices=PUBLIC_CORPORA, required=True)
parser.add_argument('--max-chunks', type=int, default=400)
parser.add_argument('--seed', type=int, default=29)
parser.add_argument('--device', default=None)
parser.add_argument('--target-fpr', type=float, default=0.05)
parser.add_argument('--injection-model', default=OFF_THE_SHELF_MODEL)
parser.add_argument('--limit', type=int, default=None, help='use only the first N attacks (smoke test)')
args = parser.parse_args()


rng = random.Random(args.seed)
docs, available = load_public(args.corpus, args.max_chunks, args.seed)
rng.shuffle([None] * available)  # same random stream as when this script shuffled the pages itself, so results reproduce
by_id = {d.doc_id: d for d in docs}
embedder, injection = CachedEmbedder(device=args.device), CachedInjection(args.injection_model, device=args.device)
attacks = []
for split in ('text', 'code'):
    for category, items in json.loads((HERE / f'data/external/bipia/{split}_attack_test.json').read_text()).items():
        attacks += [{'category': f'{split}:{category}', 'text': t} for t in items]
attacks = attacks[:args.limit] if args.limit else attacks

t0 = time.time()
# 1. False alarms on clean chunks. Flags are off in this policy so scores are collected without side effects.
shield = make_shield(embedder, injection, Policy(delta_injection=2.0, delta_offtopic=2.0, hold_edits=False))
for d in docs:
    shield.ingest(d.doc_id, d.text, args.corpus, 'high', metadata=d.metadata())
shield.fit_outlier(fit_ids=list(shield.index.ids))
records = [shield.ledger.get(cid)[0] for cid in shield.index.ids]
inj = np.array([r.scores['injection'] for r in records])
held = shield.outlier.heldout_scores > shield.outlier.threshold
hid = np.array(['hidden' in r.scores for r in records])
warned = (inj >= 0.5) | held | hid
worst = np.argsort(-inj)[:5]
results = {
    'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'corpus': args.corpus, 'seed': args.seed,
    'injection_model': args.injection_model, 'documents': len(docs), 'documents_available': available,
    'chunks': len(records),
    'clean_false_alarms': {'injection_ge_0.5': round(float((inj >= 0.5).mean()), 4), 'injection_ge_0.9': round(float((inj >= 0.9).mean()), 4),
                           'outlier_heldout': round(float(held.mean()), 4), 'hidden_chars': round(float(hid.mean()), 4),
                           'any_warning': round(float(warned.mean()), 4),
                           'highest_injection_scores': [{'chunk_id': shield.index.ids[i], 'injection': round(float(inj[i]), 4),
                                                         'starts_with': shield.index.texts[i][:150]} for i in worst]},
}

# 2. Whole-chunk detection of BIPIA attacks placed in clean chunks.
texts = list(shield.index.texts)
poisoned = []
for a in attacks:
    paras = rng.choice(texts).split('\n\n')
    at = rng.choice([0, len(paras) // 2, len(paras)])
    poisoned.append('\n\n'.join(paras[:at] + [a['text']] + paras[at:]))
p_inj = injection.score([hidden.clean(t) for t in poisoned])
p_out = shield.outlier.score(embedder.passages(poisoned)) > shield.outlier.threshold
caught = (p_inj >= 0.5) | p_out
missed = {}
for a, c in zip(attacks, caught):
    if not c:
        missed[a['category']] = missed.get(a['category'], 0) + 1
results['whole_chunk_detection'] = {'attacks': len(attacks), 'injection_ge_0.5': round(float((p_inj >= 0.5).mean()), 4),
                                    'injection_ge_0.9': round(float((p_inj >= 0.9).mean()), 4),
                                    'outlier': round(float(p_out.mean()), 4), 'any': round(float(caught.mean()), 4),
                                    'missed_by_category': dict(sorted(missed.items()))}

# 3. Delta scoring on edits.
by_type = {}
for d in docs:
    by_type.setdefault(d.doc_type, []).append(d)


def edit(doc, added, style):
    paras = delta.paragraphs(doc.text)
    at = rng.randrange(len(paras))
    new = list(paras)
    if style == 'paragraph':
        new.insert(at + 1, added)
    else:
        new[at] = new[at] + ' ' + added
    return paras, new


def benign_text(doc, style):
    others = [d for d in by_type[doc.doc_type] if d.doc_id != doc.doc_id] or [d for d in docs if d.doc_id != doc.doc_id]
    paras = [p for p in delta.paragraphs(rng.choice(others).text) if 80 <= len(p) <= 700 and not p.startswith('#')]
    para = rng.choice(paras) if paras else rng.choice(delta.paragraphs(rng.choice(docs).text))
    if style == 'paragraph':
        return para
    return rng.choice([s for s in delta.sentences(para) if len(s) > 30] or [para])


def score_edit(doc, added, style):
    before, after = edit(doc, added, style)
    scored = shield._score_additions(delta.additions(before, after), before)
    return {'offtopic': max((s['offtopic'] for s in scored), default=0.0),
            'injection': max((s['injection'] for s in scored), default=0.0),
            'hidden': max((s['hidden'] for s in scored), default=0)}


cases = []
for n, a in enumerate(attacks):
    doc, style = rng.choice(docs), ('paragraph' if n % 2 == 0 else 'inline')
    cases.append({'kind': 'attack', 'style': style, 'category': a['category'], **score_edit(doc, a['text'], style)})
for n in range(len(attacks)):
    doc, style = rng.choice(docs), ('paragraph' if n % 2 == 0 else 'inline')
    cases.append({'kind': 'benign', 'style': style, 'half': 'calibration' if (n // 2) % 2 == 0 else 'test',
                  **score_edit(doc, benign_text(doc, style), style)})
att = [c for c in cases if c['kind'] == 'attack']
ben = [c for c in cases if c['kind'] == 'benign']
cal = [c for c in ben if c['half'] == 'calibration']
test = [c for c in ben if c['half'] == 'test']


def flag_rate(rows, t_inj, t_off):
    return round(float(np.mean([c['injection'] >= t_inj or c['offtopic'] >= t_off or c['hidden'] > 0 for c in rows])), 4)


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return round(float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()), 4)


shipped = Policy()
t_off = float(np.quantile([c['offtopic'] for c in cal], 1 - args.target_fpr / 2))
t_inj = float(np.quantile([c['injection'] for c in cal], 1 - args.target_fpr / 2))
results['delta_edits'] = {
    'attacks': len(att), 'benign': len(ben),
    'shipped_thresholds': {'delta_injection': shipped.delta_injection, 'delta_offtopic': shipped.delta_offtopic,
                           'attacks_flagged': flag_rate(att, shipped.delta_injection, shipped.delta_offtopic),
                           'benign_flagged': flag_rate(ben, shipped.delta_injection, shipped.delta_offtopic),
                           'by_style': {s: {'attacks_flagged': flag_rate([c for c in att if c['style'] == s], shipped.delta_injection, shipped.delta_offtopic),
                                            'benign_flagged': flag_rate([c for c in ben if c['style'] == s], shipped.delta_injection, shipped.delta_offtopic)}
                                        for s in ('paragraph', 'inline')}},
    'recalibrated': {'target_fpr': args.target_fpr, 'delta_injection': round(t_inj, 4), 'delta_offtopic': round(t_off, 4),
                     'attacks_flagged': flag_rate(att, t_inj, t_off), 'benign_test_flagged': flag_rate(test, t_inj, t_off)},
    'injection_alone_ge_0.5': {'attacks': round(float(np.mean([c['injection'] >= 0.5 for c in att])), 4),
                               'benign': round(float(np.mean([c['injection'] >= 0.5 for c in ben])), 4)},
    'auroc_delta_injection': auroc([c['injection'] for c in att], [c['injection'] for c in ben]),
    'auroc_offtopic': auroc([c['offtopic'] for c in att], [c['offtopic'] for c in ben]),
}
results['runtime_sec'] = round(time.time() - t0, 1)
out = HERE / f'results/public_corpus-{args.corpus}-{model_tag(args.injection_model)}.json'
out.write_text(json.dumps({**results, 'edit_cases': cases}, indent=2))
print(json.dumps(results, indent=2))
print('wrote', out)
