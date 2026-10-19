"""Experiment 3: delta scoring against compromised trusted pages (threat T3 as an edit).

Every trusted Vandelay page is ingested as version 1. Then, one at a time and restored afterwards:
  attack edits  each of the 125 BIPIA test attacks (MIT, unchanged) is added to a random page, half as a new paragraph and
                half appended to an existing paragraph ("inline")
  benign edits  125 edits of the same two styles whose added text is a real paragraph or sentence from another Vandelay
                document of the same type, plus every real reissue pair in the corpus (id and id-reissue)
The off-topic threshold is calibrated on the first half of the benign edits (target 5% false alarms) and reported on the
second half, so no reported number is tuned on the data it is reported for. Writes results/delta_edits.json."""
import argparse
import json
import random
import re
import time

import numpy as np

from common import HERE, CachedEmbedder, CachedInjection, make_shield, vandelay
from ragshield import delta
from ragshield.shield import Policy

parser = argparse.ArgumentParser()
parser.add_argument('--device', default=None)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--seed', type=int, default=11)
parser.add_argument('--target-fpr', type=float, default=0.05)
args = parser.parse_args()

docs, _ = vandelay()
by_id = {d.doc_id: d for d in docs}
embedder, injection = CachedEmbedder(device=args.device), CachedInjection(device=args.device)
rng = random.Random(args.seed)

attacks = []
for split in ('text', 'code'):
    for category, items in json.loads((HERE / f'data/external/bipia/{split}_attack_test.json').read_text()).items():
        attacks += [{'split': split, 'category': category, 'text': t} for t in items]


def edit(doc, added, style):
    """Return (edited text, probe query near the edit)."""
    paras = delta.paragraphs(doc.text)
    at = rng.randrange(len(paras))
    if style == 'paragraph':
        paras.insert(at + 1, added)
    else:
        paras[at] = paras[at] + ' ' + added
    probe = ' '.join(re.sub(r'[#|`*>_-]+', ' ', paras[at]).split()[:25])
    return '\n\n'.join(paras), probe


def benign_text(doc, style):
    others = [d for d in docs if d.doc_type == doc.doc_type and d.doc_id != doc.doc_id] or [d for d in docs if d.doc_id != doc.doc_id]
    source = rng.choice(others)
    paras = [p for p in delta.paragraphs(source.text) if 80 <= len(p) <= 700] or delta.paragraphs(source.text)
    para = rng.choice(paras)
    if style == 'paragraph':
        return para
    sents = [s for s in delta.sentences(para) if len(s) > 30] or [para]
    return rng.choice(sents)


cases = []
for n, attack in enumerate(attacks):
    doc = rng.choice(docs)
    style = 'paragraph' if n % 2 == 0 else 'inline'
    text, probe = edit(doc, attack['text'], style)
    cases.append({'kind': 'attack', 'id': f'a{n:03d}', 'doc': doc.doc_id, 'style': style, 'category': f"{attack['split']}:{attack['category']}",
                  'added': attack['text'], 'text': text, 'probe': probe})
for n in range(len(attacks)):
    doc = rng.choice(docs)
    style = 'paragraph' if n % 2 == 0 else 'inline'
    added = benign_text(doc, style)
    text, probe = edit(doc, added, style)
    # Halves are assigned in pairs so both edit styles appear in both the calibration and the test half.
    cases.append({'kind': 'benign', 'id': f'b{n:03d}', 'doc': doc.doc_id, 'style': style, 'added': added, 'text': text, 'probe': probe,
                  'half': 'calibration' if (n // 2) % 2 == 0 else 'test'})
for d in docs:
    if d.doc_id + '-reissue' in by_id:
        cases.append({'kind': 'benign_real', 'id': f'r-{d.doc_id}', 'doc': d.doc_id, 'style': 'reissue',
                      'added': '', 'text': by_id[d.doc_id + '-reissue'].text, 'probe': d.title, 'half': 'test'})


def build(policy):
    shield = make_shield(embedder, injection, policy)
    for d in docs:
        shield.ingest(d.doc_id, d.text, 'vandelay-internal', 'high', metadata=d.metadata())
    shield.fit_outlier()
    return shield


def run(shield, case, retrieve=False):
    shield.ingest(case['doc'], case['text'], 'vandelay-internal', 'high')
    rows = [(cid, shield.ledger.get(cid)[0].scores) for cid in shield.index.ids if cid.startswith(case['doc'] + '::')]
    edited = [(cid, s) for cid, s in rows if 'delta' in s]
    out = {'offtopic': max((s['delta']['offtopic'] for _, s in edited), default=0.0),
           'delta_injection': max((s['delta']['injection'] for _, s in edited), default=0.0),
           'chunk_injection': max((s['injection'] for _, s in edited), default=0.0),
           'flagged': any(s.get('delta_flag') for _, s in edited), 'edited_chunks': [cid for cid, _ in edited]}
    if retrieve and edited:
        raw = {h['chunk_id'] for h in shield.retrieve(case['probe'], args.k, shielded=False)}
        hits = {h['chunk_id']: h for h in shield.retrieve(case['probe'], args.k)}
        target = [cid for cid, _ in edited if cid in raw]
        out['retrieved_unshielded'] = bool(target)
        if target:
            cid = target[0]
            out['outcome'] = 'suppressed' if cid not in hits else ('served_silently' if hits[cid]['trust']['action'] == 'pass' else 'served_with_warning')
    shield.ingest(case['doc'], by_id[case['doc']].text, 'vandelay-internal', 'high')  # restore version 1 content
    return out


t0 = time.time()
# Pass 1: threshold-free scores (flags disabled), used for calibration.
probe_policy = Policy(delta_injection=2.0, delta_offtopic=2.0)
shield = build(probe_policy)
for case in cases:
    case['scores'] = run(shield, case)

# Both thresholds come from benign calibration edits only, each at half the target rate, so their union stays near it.
cal = [c['scores'] for c in cases if c['kind'] == 'benign' and c['half'] == 'calibration']
calibration = np.array([s['offtopic'] for s in cal])
threshold = float(np.quantile(calibration, 1 - args.target_fpr / 2))
inj_threshold = float(np.quantile([s['delta_injection'] for s in cal], 1 - args.target_fpr / 2))

# Pass 2: calibrated policy, with retrieval outcomes.
policy = Policy(delta_offtopic=threshold, delta_injection=inj_threshold)
shield = build(policy)
for case in cases:
    case['final'] = run(shield, case, retrieve=True)


def rate(rows, key):
    return round(float(np.mean([r[key] for r in rows])), 4) if rows else None


attack_rows = [c for c in cases if c['kind'] == 'attack']
benign_test = [c for c in cases if c['kind'] == 'benign' and c['half'] == 'test']
real = [c for c in cases if c['kind'] == 'benign_real']


def detection(rows):
    if not rows:
        return None
    return {'flagged': rate([c['final'] for c in rows], 'flagged'),
            'delta_injection_ge_threshold': round(float(np.mean([c['scores']['delta_injection'] >= inj_threshold for c in rows])), 4),
            'whole_chunk_injection_ge_0.5': round(float(np.mean([c['scores']['chunk_injection'] >= 0.5 for c in rows])), 4),
            'offtopic_ge_threshold': round(float(np.mean([c['scores']['offtopic'] >= threshold for c in rows])), 4)}


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return round(float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())), 4)


retrieved = [c for c in attack_rows if c['final'].get('retrieved_unshielded')]
outcomes = {o: sum(c['final']['outcome'] == o for c in retrieved) for o in ('served_silently', 'served_with_warning', 'suppressed')}
result = {
    'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'seed': args.seed,
    'attack_source': 'microsoft/BIPIA benchmark/{text,code}_attack_test.json (MIT)',
    'calibration': {'benign_calibration_edits': int(len(calibration)), 'target_fpr': args.target_fpr,
                    'offtopic_threshold': round(threshold, 4), 'delta_injection_threshold': round(inj_threshold, 4)},
    'attacks': {'n': len(attack_rows), **detection(attack_rows),
                'by_style': {s: detection([c for c in attack_rows if c['style'] == s]) for s in ('paragraph', 'inline')}},
    'benign_test_edits': {'n': len(benign_test), **detection(benign_test),
                          'by_style': {s: detection([c for c in benign_test if c['style'] == s]) for s in ('paragraph', 'inline')}},
    'benign_real_reissues': {'n': len(real), 'flagged': [c['id'] for c in real if c['final']['flagged']]},
    'auroc_offtopic_attack_vs_benign_test': auroc([c['scores']['offtopic'] for c in attack_rows], [c['scores']['offtopic'] for c in benign_test]),
    'auroc_delta_injection_attack_vs_benign_test': auroc([c['scores']['delta_injection'] for c in attack_rows], [c['scores']['delta_injection'] for c in benign_test]),
    'auroc_whole_chunk_injection_attack_vs_benign_test': auroc([c['scores']['chunk_injection'] for c in attack_rows], [c['scores']['chunk_injection'] for c in benign_test]),
    'retrieval_when_attack_reaches_top_k': {'n': len(retrieved), **outcomes,
                                            'silent_attack_success_rate': round(outcomes['served_silently'] / max(1, len(retrieved)), 4)},
    'missed_attacks_by_category': {},
    'runtime_sec': round(time.time() - t0, 1),
}
for c in attack_rows:
    if not c['final']['flagged']:
        result['missed_attacks_by_category'][c['category']] = result['missed_attacks_by_category'].get(c['category'], 0) + 1
(HERE / 'results/delta_edits.json').write_text(json.dumps({**result, 'cases': [{k: v for k, v in c.items() if k != 'text'} for c in cases]}, indent=2))
print(json.dumps(result, indent=2))
