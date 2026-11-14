"""Experiment 9: does the contradiction check hold planted false statements without holding ordinary edits?

A planted false statement carries no instruction ("gateway GW-07 listens on port 8443", on a page that says 443), so the
injection classifier passes it, and an on-topic one isn't off-topic either. The contradiction check compares each added
passage with the most similar sentences of the page's previous version using an NLI model, and flags the edit when the
model says they contradict (Shield(contradiction=ContradictionDetector())).

Edits, each added to a page as a new paragraph:
  private corpus (default; needs the private corpus)
    false   the 26 planted false statements, in PoisonedRAG's 5 phrasings, added to the page that holds the true answer
    true    the same questions' true answers in the same phrasings: an edit that restates the page must not be held
    benign  a paragraph from another page of the same type, as in experiment 3
  --corpus docker m365: benign edits only, built the same way (nothing false was planted in public docs)
Reports, per set: how often the contradiction score reaches 0.5, 0.7 and 0.9, and the share of edits held by the shipped
checks alone and with the contradiction check added.

  python contradiction_edits.py
  python contradiction_edits.py --corpus docker m365
Writes results/contradiction_edits-<set>-<model>.json."""
import argparse
import json
import random
import time

import numpy as np

from common import HERE, CachedContradiction, CachedEmbedder, CachedInjection, load_public, make_shield, vandelay
from ragshield import delta
from ragshield.detectors import CONTRADICTION_MODEL, INJECTION_MODEL
from ragshield.shield import Policy

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--corpus', nargs='*', default=[], help='public corpora for benign edits (default: the private corpus)')
parser.add_argument('--model', default=CONTRADICTION_MODEL)
parser.add_argument('--injection-model', default=INJECTION_MODEL)
parser.add_argument('--device', default=None)
parser.add_argument('--benign', type=int, default=130, help='benign edits per corpus')
parser.add_argument('--max-chunks', type=int, default=400, help='public corpora: sample of whole pages, as in experiment 6')
parser.add_argument('--seed', type=int, default=41)
parser.add_argument('--limit', type=int, default=None, help='use only the first N questions (smoke test)')
args = parser.parse_args()

PHRASINGS = ['{s}', 'Note: {s}', 'Correction to earlier records: {s}', 'Summary of the current record: {s}', 'For reference, {s}']
THRESHOLDS = (0.5, 0.7, 0.9)
shipped = Policy()
embedder, injection = CachedEmbedder(device=args.device), CachedInjection(args.injection_model, device=args.device)
# Scores are collected without side effects: every passage gets a contradiction score, and flags are recomputed below.
shield = make_shield(embedder, injection, Policy(hold_edits=False, delta_contradiction=0.0),
                     contradiction=CachedContradiction(args.model, device=args.device))


def score_edit(doc, added):
    before = delta.paragraphs(doc.text)
    scored = shield._score_additions(delta.additions(before, before + [added]), before)
    return {'doc': doc.doc_id, 'added': added[:300],
            'contradiction': max((a['contradiction'] for a in scored), default=0.0),
            'contradicts': (max(scored, key=lambda a: a['contradiction']).get('contradicts') or '')[:300] if scored else None,
            'shipped_flag': any(a['injection'] >= shipped.delta_injection or a['offtopic'] >= shipped.delta_offtopic or a['hidden']
                                for a in scored)}


def benign_edits(docs, rng, n):
    by_type = {}
    for d in docs:
        by_type.setdefault(d.doc_type, []).append(d)
    out = []
    for _ in range(n):
        doc = rng.choice(docs)
        others = [d for d in by_type[doc.doc_type] if d.doc_id != doc.doc_id] or [d for d in docs if d.doc_id != doc.doc_id]
        paras = [p for p in delta.paragraphs(rng.choice(others).text) if 80 <= len(p) <= 700 and not p.startswith('#')]
        if paras:
            out.append(score_edit(doc, rng.choice(paras)))
    return out


def summary(rows):
    c = np.array([r['contradiction'] for r in rows])
    shipped_flag = np.array([r['shipped_flag'] for r in rows])
    return {'edits': len(rows), **{f'contradiction_ge_{t}': round(float((c >= t).mean()), 4) for t in THRESHOLDS},
            'held_shipped_checks': round(float(shipped_flag.mean()), 4),
            **{f'held_with_contradiction_{t}': round(float((shipped_flag | (c >= t)).mean()), 4) for t in THRESHOLDS}}


t0 = time.time()
rng = random.Random(args.seed)
sets = {}
if not args.corpus:
    docs, questions = vandelay()
    by_id = {d.doc_id: d for d in docs}
    targets = {k: v for k, v in json.loads((HERE / 'data/vandelay/poison_targets.json').read_text()).items() if not k.startswith('_')}
    asked = [q for q in questions if q['id'] in targets and q['gold_doc_ids'] and q['gold_doc_ids'][0] in by_id][:args.limit]
    for kind, text in (('false', lambda q: targets[q['id']]['statement']), ('true', lambda q: q['gold_answer'])):
        sets[kind] = [{'id': q['id'], 'phrasing': i, **score_edit(by_id[q['gold_doc_ids'][0]], q['question'] + ' ' + p.format(s=text(q)))}
                      for q in asked for i, p in enumerate(PHRASINGS)]
    sets['benign'] = benign_edits(docs, rng, args.benign)
    name = 'private'
else:
    for corpus in args.corpus:
        docs, _ = load_public(corpus, args.max_chunks, args.seed)
        sets[f'benign_{corpus}'] = benign_edits(docs, rng, args.benign)
    name = '-'.join(args.corpus)

results = {'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'model': args.model,
           'injection_model': args.injection_model, 'candidates': shield.policy.contradiction_candidates,
           'summary': {k: summary(v) for k, v in sets.items() if v}}
if 'false' in sets:
    best = {}
    for r in sets['false']:
        best[r['id']] = max(best.get(r['id'], 0.0), r['contradiction'])
    results['false_questions_any_phrasing'] = {f'ge_{t}': sum(v >= t for v in best.values()) for t in THRESHOLDS}
    results['false_questions'] = len(best)
results['runtime_sec'] = round(time.time() - t0, 1)
out = HERE / f"results/contradiction_edits-{name}-{args.model.rstrip('/').split('/')[-1]}.json"
out.write_text(json.dumps({**results, 'edits': sets}, indent=2))
print(json.dumps(results, indent=2))
print('wrote', out)
