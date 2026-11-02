"""Index exposure report: scan a folder of documents the way RAG Shield would ingest them, and say what it would flag.

This is the free, local, read-only check behind the design-partner program: nothing leaves the machine. It ingests every
text-like file as a trusted source, fits the outlier model on the result, and reports which chunks would be served with a
warning, sent to review, or (for low-trust sources) dropped. Scores are the same ones the SDK uses at ingestion.

Usage: python -m ragshield.report <folder or files> [--out report.md] [--json report.json] [--device cuda]
       [--injection-model protectai/deberta-v3-base-prompt-injection-v2] [--top 10]"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import hidden

TEXT_SUFFIXES = {'.md', '.txt', '.rst', '.markdown', '.log', '.csv', '.json', '.yaml', '.yml', '.html', '.htm'}


def find_files(paths, suffixes=TEXT_SUFFIXES):
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files += sorted(f for f in p.rglob('*') if f.is_file() and f.suffix.lower() in suffixes)
        elif p.is_file():
            files.append(p)
    return files


def scan(shield, files, root=None, source_id='local-files', trust_tier='high'):
    """Ingest files into a fresh shield and return the per-chunk findings as plain dicts (no text leaves this dict)."""
    t0 = time.time()
    docs = []
    for f in files:
        try:
            text = f.read_text(encoding='utf-8', errors='replace')
        except OSError as e:
            print(f'skip {f}: {e}', file=sys.stderr)
            continue
        if not text.strip():
            continue
        doc_id = str(f.relative_to(root)) if root and root in f.parents else f.name
        shield.ingest(doc_id, text, source_id=source_id, trust_tier=trust_tier)
        docs.append(doc_id)
    n_chunks = len(shield.index.ids)
    outlier_flags = shield.fit_outlier(fit_ids=list(shield.index.ids)) if shield.outlier and n_chunks else 0
    p = shield.policy
    chunks = []
    for cid, text in zip(shield.index.ids, shield.index.texts):
        record = shield.ledger.get(cid)[0]
        inj = record.scores.get('injection', 0.0)
        reasons = []
        if inj >= p.injection_block:
            reasons.append(f'likely instructions to the model ({inj:.2f})')
        elif inj >= p.injection_warn:
            reasons.append(f'possible instructions to the model ({inj:.2f})')
        if record.scores.get('outlier_flag'):
            reasons.append(f"unlike the rest of the corpus (outlier {record.scores['outlier']:.2f})")
        if record.scores.get('hidden'):
            reasons.append(hidden.describe(record.scores['hidden']))
        chunks.append({'chunk_id': cid, 'doc_id': record.doc_id, 'injection': round(float(inj), 4),
                       'outlier': record.scores.get('outlier'), 'outlier_flag': bool(record.scores.get('outlier_flag')),
                       'hidden': record.scores.get('hidden', {}), 'reasons': reasons, 'snippet': text[:160].replace('\n', ' ')})
    inj_scores = np.array([c['injection'] for c in chunks]) if chunks else np.zeros(0)
    summary = {
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'seconds': round(time.time() - t0, 1),
        'documents': len(docs), 'chunks': n_chunks, 'signed': n_chunks,
        'injection_block': int((inj_scores >= p.injection_block).sum()), 'injection_warn': int((inj_scores >= p.injection_warn).sum()),
        'outlier_flags': int(outlier_flags), 'hidden': sum(1 for c in chunks if c['hidden']), 'flagged': sum(1 for c in chunks if c['reasons']),
        'policy': {'injection_warn': p.injection_warn, 'injection_block': p.injection_block,
                   'outlier_target_fpr': getattr(shield.outlier, 'target_fpr', None)},
    }
    summary['clean_pct'] = round(100 * (1 - summary['flagged'] / n_chunks), 1) if n_chunks else 100.0
    return summary, chunks


def render(summary, chunks, top=10):
    s = summary
    lines = [f"# RAG Shield index exposure report", '',
             f"Scanned {s['documents']} documents → {s['chunks']} chunks in {s['seconds']} s. Nothing left this machine.", '',
             '| Check | Result | What RAG Shield would do |', '|---|---|---|',
             f"| Signed provenance | {s['signed']} of {s['chunks']} chunks signed | Any later change in the vector store is detected and dropped |",
             f"| Instructions to the model (score ≥ {s['policy']['injection_block']}) | {s['injection_block']} chunks | Trusted source: served with a warning and sent to review. Low-trust source: quarantined |",
             f"| Possible instructions (score ≥ {s['policy']['injection_warn']}) | {s['injection_warn']} chunks | Served with a warning label and reduced rank |",
             f"| Unlike the rest of the corpus (outlier) | {s['outlier_flags']} chunks (target false-positive rate {s['policy']['outlier_target_fpr']}) | Served with a warning; shown first in the review queue |",
             f"| Hidden or look-alike characters | {s['hidden']} chunks | Trusted source: served with a warning and sent to review. Low-trust source: dropped |",
             f"| **Clean** | **{s['clean_pct']}%** of chunks raise no flag | |", '']
    flagged = sorted((c for c in chunks if c['reasons']), key=lambda c: (-c['injection'], -(c['outlier'] or 0)))
    if flagged:
        lines += [f'## Top {min(top, len(flagged))} chunks to review', '', '| Document | Chunk | Why | Starts with |', '|---|---|---|---|']
        for c in flagged[:top]:
            lines.append(f"| {c['doc_id']} | {c['chunk_id'].rsplit('::', 1)[-1]} | {'; '.join(c['reasons'])} | {c['snippet'][:100]}… |")
        lines.append('')
    lines += ['## How to read this', '',
              'Flags are not verdicts. Security text (logs, incident reports, EDR exports) scores high on generic injection classifiers;',
              'that is why a trusted chunk is never deleted on a score alone. The numbers show how much of your index a reviewer would',
              'need to look at once, and how much would be auto-checked after that. Edits made after this scan are scored on what',
              'they add, which is much more sensitive than scoring whole pages (23% → 98% of poisoned edits on held-out documents).', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+')
    parser.add_argument('--out', help='write the Markdown report here (default: stdout)')
    parser.add_argument('--json', help='also write the summary and per-chunk findings as JSON')
    parser.add_argument('--device', default=None)
    parser.add_argument('--injection-model', default=None, help='HF model id (default: the SDK default classifier)')
    parser.add_argument('--embed-model', default=None)
    parser.add_argument('--top', type=int, default=10)
    args = parser.parse_args(argv)

    from .detectors import InjectionDetector, OutlierDetector
    from .index import Embedder, VectorIndex
    from .provenance import Ledger, Signer
    from .shield import Shield
    files = find_files(args.paths)
    if not files:
        parser.error('no text-like files found (looked for ' + ', '.join(sorted(TEXT_SUFFIXES)) + ')')
    print(f'scanning {len(files)} files…', file=sys.stderr)
    embedder = Embedder(**({'model': args.embed_model} if args.embed_model else {}), device=args.device)
    injection = InjectionDetector(**({'model': args.injection_model} if args.injection_model else {}), device=args.device)
    shield = Shield(Ledger(), Signer.generate(), embedder, VectorIndex(), injection=injection, outlier=OutlierDetector())
    root = Path(args.paths[0]).resolve() if len(args.paths) == 1 and Path(args.paths[0]).is_dir() else None
    summary, chunks = scan(shield, [f.resolve() for f in files], root=root)
    text = render(summary, chunks, top=args.top)
    if args.out:
        Path(args.out).write_text(text)
        print(f'wrote {args.out}', file=sys.stderr)
    else:
        print(text)
    if args.json:
        Path(args.json).write_text(json.dumps({'summary': summary, 'chunks': chunks}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
