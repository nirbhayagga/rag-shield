"""Experiment 10: how often would real page edits be held?

The harmless edits in experiments 3, 6 and 9 are synthetic: a paragraph from another page pasted in. This uses real ones:
consecutive versions of pages from the public documentation repositories' git history, cleaned as in fetch_corpus.py,
one changed page per commit. For each edit it reports which checks would hold it:
  shipped   instructions, off-topic text or hidden characters in what the edit added (the delta checks)
  facts     a replaced link, email address, phone number, IP address or amount (Policy.fact_changes)
Link targets are kept here (the m365 corpus drops them), because the fact check needs them.

It also plants one replaced fact (a link to another site, another address, number or amount) into the edited pages that
have one, to check that the rule catches it. Catching it is mostly by construction; the number that matters is the share
of real edits held.

  python real_edits.py --repo docker k8s m365 --edits 150
  python real_edits.py --facts-only      # the fact check alone, on the same edits: no models, a few minutes
Needs git and GitHub access on the first run; the partial clones are kept in results/cache/history (blobs on demand).
Writes results/real_edits-<repos>-<model>.json."""
import argparse
import json
import random
import re
import subprocess
import time

import numpy as np

from common import CACHE, HERE, CachedEmbedder, CachedInjection, make_shield
from fetch_corpus import strip_hugo, strip_learn
from ragshield import delta, facts
from ragshield.detectors import INJECTION_MODEL
from ragshield.shield import Policy

# The history is read back from these commits (each repository's default branch on the day of the first run), so the
# same edits are sampled however far the branch has moved since.
HISTORY_REF = {'docker': '553482ce632f00c6e7faabab863422b4916e7c18', 'k8s': 'a96b3ebd15296afc96819e6e5f8a90bcff00b828',
               'm365': 'ee0f8f75d0045cc6397fe0a9c2c84d2aba1e01d6'}
REPOS = {  # name: (repository, path, cleaner, skip)
    'docker': ('https://github.com/docker/docs.git', 'content/manuals', strip_hugo,
               lambda path: 'release-notes' in path or path.rsplit('/', 1)[-1].startswith('release')),
    'k8s': ('https://github.com/kubernetes/website.git', 'content/en/docs/tasks', strip_hugo, lambda path: False),
    'm365': ('https://github.com/MicrosoftDocs/microsoft-365-docs.git', 'microsoft-365/admin',
             lambda body: strip_learn(re.sub(r'(?<!!)\[([^\]]+)\]\((https?://[^)\s]+)[^)]*\)', r'\1 (\2)', body)), lambda path: False),
}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--repo', nargs='+', default=list(REPOS), choices=list(REPOS))
parser.add_argument('--edits', type=int, default=150, help='edits per repository (one changed page per commit)')
parser.add_argument('--since', default='2025-11-01', help='commits after this date')
parser.add_argument('--injection-model', default=INJECTION_MODEL)
parser.add_argument('--device', default=None)
parser.add_argument('--seed', type=int, default=43)
parser.add_argument('--facts-only', action='store_true', help='run only the fact check (no models); same edits and plants')
args = parser.parse_args()


def git(repo_dir, *a, input=None):
    return subprocess.run(['git', '-C', str(repo_dir), *a], check=True, capture_output=True, text=True, input=input).stdout


def history(name):
    url, path, _, skip = REPOS[name]
    repo_dir = CACHE / 'history' / f'{name}.git'
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '-q', '--bare', '--filter=blob:none', '--single-branch', f'--shallow-since={args.since}',
                        url, str(repo_dir)], check=True)
    log = git(repo_dir, 'log', '--no-merges', '--diff-filter=M', '--name-only', '--format=@@%H', HISTORY_REF[name], '--', path)
    commits = []
    for block in log.split('@@')[1:]:
        sha, *files = [line for line in block.split('\n') if line.strip()]
        files = [f for f in files if f.endswith('.md') and not skip(f[len(path) + 1:-3])]
        if files:
            commits.append((sha, files))
    return repo_dir, commits


def page(raw, clean):
    title, body = None, raw
    m = re.match(r'^---\n(.*?)\n---\n', raw, flags=re.S)
    if m:
        t = re.search(r'^title:\s*["\']?(.*?)["\']?\s*$', m.group(1), flags=re.M)
        title = t.group(1) if t else None
        body = raw[m.end():]
    body = clean(strip_hugo(body)) if clean is not strip_hugo else strip_hugo(body)
    return f'# {title}\n\n{body}' if title else body


def blobs(repo_dir, specs):
    """Contents for '<commit>:<path>' specs; missing blobs are fetched from the promisor remote in one batch."""
    oids = {}
    for spec in specs:
        try:
            oids[spec] = git(repo_dir, 'rev-parse', spec).strip()
        except subprocess.CalledProcessError:  # no parent in the shallow clone
            pass
    missing = [o for o in set(oids.values()) if subprocess.run(['git', '-C', str(repo_dir), 'cat-file', '-e', o],
                                                              capture_output=True).returncode]
    for i in range(0, len(missing), 200):
        subprocess.run(['git', '-C', str(repo_dir), '-c', 'fetch.negotiationAlgorithm=noop', 'fetch', '-q', '--no-tags',
                        '--no-write-fetch-head', '--filter=blob:none', 'origin', *missing[i:i + 200]], check=False, capture_output=True)
    out = {}
    for spec, oid in oids.items():
        try:
            out[spec] = git(repo_dir, 'cat-file', 'blob', oid)
        except subprocess.CalledProcessError:
            pass
    return out


def plant(text, rng):
    """Replace one occurrence of a fact on the page, as an attacker editing the page would. None if the page has none."""
    found = [(kind, v) for kind, values in facts.extract(text).items() for v in values.values()]
    if not found:
        return None
    kind, old = rng.choice(found)
    if kind == 'link':
        host = facts._url(old)[1]
        new = old.replace(host, host.split('.')[0] + '-help.net', 1)
    elif kind == 'email address':
        new = 'support@' + old.split('@')[1].split('.')[0] + '-help.net'
    elif kind == 'amount':
        new = re.sub(r'\d', lambda d: str((int(d.group(0)) + 1) % 10), old, count=1)
    else:
        new = old[:-1] + str((int(old[-1]) + 1) % 10)
    return kind, old, new, text.replace(old, new, 1)


t0 = time.time()
rng = random.Random(args.seed)
policy = Policy(hold_edits=False)
if not args.facts_only:
    embedder, injection = CachedEmbedder(device=args.device), CachedInjection(args.injection_model, device=args.device)
    embedder.save = lambda: None  # the embedding cache is large: write it once at the end, not after every edit
    shield = make_shield(embedder, injection, policy)
SHIPPED = {'instructions', 'off-topic', 'hidden'}
summary, rows = {}, {}
for name in args.repo:
    repo_dir, commits = history(name)
    rng.shuffle(commits)
    picked = [(sha, rng.choice(files)) for sha, files in commits[:args.edits * 2]]  # spare commits for pages that drop out
    texts = blobs(repo_dir, [f'{sha}^:{f}' for sha, f in picked] + [f'{sha}:{f}' for sha, f in picked])
    clean = REPOS[name][2]
    out, no_visible_change, deletions_only, planted = [], 0, 0, []
    for sha, f in picked:
        if len(out) >= args.edits:
            break
        if f'{sha}^:{f}' not in texts or f'{sha}:{f}' not in texts:
            continue
        before_text, after_text = page(texts[f'{sha}^:{f}'], clean), page(texts[f'{sha}:{f}'], clean)
        if len(before_text) < 300:
            continue
        before, after = delta.paragraphs(before_text), delta.paragraphs(after_text)
        if before == after:
            no_visible_change += 1  # front matter, comments or shortcodes only
            continue
        added = delta.additions(before, after)
        if args.facts_only:
            scored = [{'reasons': [('fact', facts.describe(*c)) for c in found], 'facts': [list(c) for c in found]}
                      for found in (facts.changes('\n\n'.join(before), '\n\n'.join(after), a) for a in added)]
        else:
            scored = shield._score_additions(added, before, after) if added else []
        if not added:
            deletions_only += 1
        kinds = {kind for a in scored for kind, _ in a['reasons']}
        out.append({'commit': sha, 'file': f, 'added': len(added), 'added_chars': sum(len(a) for a in added),
                    'shipped': bool(kinds & SHIPPED), 'facts': [c for a in scored for c in a.get('facts', [])],
                    'reasons': [r for a in scored for _, r in a['reasons']][:6]})
        trial = plant(before_text, rng)
        if trial:
            kind, old, new, text = trial
            paras = delta.paragraphs(text)
            changed = [c for p in delta.additions(before, paras) for c in facts.changes(before_text, text, p)]
            planted.append({'kind': kind, 'old': old, 'new': new, 'caught': bool(changed),
                            'old_still_on_page': old in text})
    n = len(out)
    shipped = np.array([r['shipped'] for r in out])
    fact = np.array([bool(r['facts']) for r in out])
    by_kind = {}
    for r in out:
        for kind in {c[0] for c in r['facts']}:
            by_kind[kind] = by_kind.get(kind, 0) + 1
    summary[name] = {
        'edits': n, 'skipped_no_visible_change': no_visible_change, 'deletions_only': deletions_only,
        'held_shipped_checks': round(float(shipped.mean()), 4) if n and not args.facts_only else None,
        'held_fact_check': round(float(fact.mean()), 4) if n else None,
        'held_fact_check_only': round(float((fact & ~shipped).mean()), 4) if n and not args.facts_only else None,
        'held_either': round(float((fact | shipped).mean()), 4) if n and not args.facts_only else None,
        'fact_holds_by_kind': by_kind,
        'planted': {'pages': len(planted), 'caught': sum(p['caught'] for p in planted),
                    'by_kind': {k: [sum(p['caught'] for p in planted if p['kind'] == k), sum(p['kind'] == k for p in planted)]
                                for k in sorted({p['kind'] for p in planted})}},
    }
    rows[name] = {'edits': out, 'planted': planted}
    print(name, json.dumps(summary[name]))

if not args.facts_only:
    CachedEmbedder.save(embedder)
results = {'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'injection_model': args.injection_model,
           'since': args.since, 'history_ref': {name: HISTORY_REF[name] for name in args.repo}, 'seed': args.seed, 'summary': summary, 'runtime_sec': round(time.time() - t0, 1)}
model = 'facts-only' if args.facts_only else args.injection_model.rstrip('/').split('/')[-1]
out_path = HERE / f"results/real_edits-{'-'.join(args.repo)}-{model}.json"
out_path.write_text(json.dumps({**results, 'rows': rows}, indent=2))
print(json.dumps(results, indent=2))
print('wrote', out_path)
