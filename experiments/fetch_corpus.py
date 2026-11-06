"""Fetch the public corpora for experiments 5 and 6 at pinned versions, as data/external/<name>/docs.jsonl (gitignored).

  k8s     Kubernetes documentation, content/en/docs/tasks (CC BY 4.0, (c) The Kubernetes Authors): a public runbook-style
          wiki, full of shell commands and imperative steps
  attack  MITRE ATT&CK Enterprise v20.0 technique pages (MITRE ATT&CK license: research, development and commercial use
          with MITRE's copyright notice): security text that describes attacker behavior
  docker  Docker documentation, content/manuals without release notes (Apache-2.0, Docker Inc.): how-to pages like k8s
          but written by other people, so a classifier trained on k8s text can be tested on how-to text it never saw
  m365    Microsoft 365 admin documentation, microsoft-365/admin (CC BY 4.0, (c) Microsoft Corporation): IT how-to pages
          with nothing to do with containers, the closest public text to a company's internal help pages

Each line: {"doc_id", "title", "doc_type", "text"}. SOURCE.md next to it records the version, license and attribution.
Usage: python fetch_corpus.py [k8s|attack|docker|m365|all]"""
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
K8S_REF = '177aebb48df75c35d179ae1fb0338afa15e4abd4'  # kubernetes/website main, 2026-11-02
ATTACK_REF = 'c2494b29dd268c46a5d0ea0990553e4327cd2253'  # mitre-attack/attack-stix-data master, 2026-10-13
ATTACK_VERSION = '20.0'
DOCKER_REF = '2eaa01ac4fbe81c8cb8705724a05e19474dd681a'  # docker/docs main, 2026-11-02
M365_REF = 'ee0f8f75d0045cc6397fe0a9c2c84d2aba1e01d6'  # MicrosoftDocs/microsoft-365-docs default branch, 2026-11-04


def write(name, docs, source_md):
    out = HERE / 'data/external' / name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'docs.jsonl', 'w') as f:
        for d in docs:
            f.write(json.dumps(d) + '\n')
    (out / 'SOURCE.md').write_text(source_md)
    print(f'{name}: {len(docs)} documents, {sum(len(d["text"]) for d in docs) // 1000}K characters -> {out}')


def strip_hugo(body):
    body = re.sub(r'<!--.*?-->', '', body, flags=re.S)
    body = re.sub(r'\{\{<\s*glossary_tooltip[^>]*?text="([^"]*)"[^>]*>\}\}', r'\1', body)
    headings = {'prerequisites': 'Before you begin', 'whatsnext': "What's next", 'objectives': 'Objectives',
                'cleanup': 'Clean up', 'synopsis': 'Synopsis', 'options': 'Options', 'seealso': 'See also'}
    body = re.sub(r'\{\{%\s*heading\s+"([^"]+)"\s*%\}\}', lambda m: headings.get(m.group(1), m.group(1).capitalize()), body)
    body = re.sub(r'\{\{[<%].*?[%>]\}\}', '', body, flags=re.S)
    body = re.sub(r'</?(?:br|p|div|span|table|tr|td|th|thead|tbody)\b[^>]*>', '', body)
    body = '\n'.join('' if not line.strip() or re.fullmatch(r'#+\s*', line.strip()) else line.rstrip() for line in body.split('\n'))
    return re.sub(r'\n{3,}', '\n\n', body).strip()


def strip_learn(body):
    """Microsoft Learn Markdown: drop the page's own title, images, includes, videos, monikers and callout markers, and keep
    link text without its target."""
    body = re.sub(r'\A# .*\n+', '', body)
    body = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', body)
    body = re.sub(r':::image\b.*?:::', '', body)
    body = re.sub(r'</?(?:a|b|i|em|strong|sup|sub)\b[^>]*>', '', body)
    body = re.sub(r'</?(?:ul|ol|li)\b[^>]*>', ' ', body)
    body = re.sub(r'^[ \t>]*(\[!(?:INCLUDE|VIDEO|div)\b.*|\[!(?:NOTE|IMPORTANT|TIP|WARNING|CAUTION)\]\s*|:::.*)$', '', body, flags=re.M)
    body = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', body)
    body = re.sub(r'^([ \t]*)> ?', r'\1', body, flags=re.M)
    body = '\n'.join(line.rstrip() for line in body.split('\n'))
    return re.sub(r'\n{3,}', '\n\n', body).strip()


def hugo_pages(repo, ref, path, name, default_type, skip=lambda rel: False, clean=lambda body: body):
    """Markdown pages under path in a Hugo docs repo at a pinned commit (sparse, shallow checkout)."""
    with tempfile.TemporaryDirectory() as tmp:
        git = lambda *a: subprocess.run(['git', '-C', tmp, *a], check=True, capture_output=True, text=True)
        git('init', '-q')
        git('remote', 'add', 'origin', repo)
        git('sparse-checkout', 'set', '--no-cone', f'/{path}/')
        git('fetch', '-q', '--depth', '1', '--filter=blob:none', 'origin', ref)
        git('checkout', '-q', 'FETCH_HEAD')
        root = Path(tmp) / path
        docs = []
        for page in sorted(root.rglob('*.md')):
            rel = page.relative_to(root).with_suffix('')
            if skip(rel):
                continue
            raw = page.read_text(encoding='utf-8')
            title, body = page.stem, raw
            m = re.match(r'^---\n(.*?)\n---\n', raw, flags=re.S)
            if m:
                t = re.search(r'^title:\s*["\']?(.*?)["\']?\s*$', m.group(1), flags=re.M)
                title = t.group(1) if t else title
                body = raw[m.end():]
            body = clean(strip_hugo(body))
            if len(body) < 300:
                continue
            docs.append({'doc_id': f'{name}/' + str(rel).replace('/_index', ''), 'title': title,
                         'doc_type': rel.parts[0] if len(rel.parts) > 1 else default_type, 'text': f'# {title}\n\n{body}'})
    return docs


def fetch_k8s():
    docs = hugo_pages('https://github.com/kubernetes/website.git', K8S_REF, 'content/en/docs/tasks', 'k8s', 'tasks')
    write('k8s', docs, f"""# Kubernetes documentation (tasks)

Source: https://github.com/kubernetes/website, commit {K8S_REF}, path content/en/docs/tasks.
License: Creative Commons Attribution 4.0 International (CC BY 4.0). (c) The Kubernetes Authors.
Changes: Hugo front matter, shortcodes and HTML comments removed; pages under 300 characters skipped.
""")


def fetch_attack():
    url = (f'https://raw.githubusercontent.com/mitre-attack/attack-stix-data/{ATTACK_REF}/enterprise-attack/'
           f'enterprise-attack-{ATTACK_VERSION}.json')
    with urllib.request.urlopen(url, timeout=120) as r:
        bundle = json.load(r)
    clean = lambda s: re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', re.sub(r'\s*\(Citation:[^)]*\)', '', s or '')).replace('<code>', '`').replace('</code>', '`')
    docs = []
    for o in bundle['objects']:
        if o.get('type') != 'attack-pattern' or o.get('revoked') or o.get('x_mitre_deprecated'):
            continue
        ext = next((r['external_id'] for r in o.get('external_references', []) if r.get('source_name') == 'mitre-attack'), None)
        if not ext:
            continue
        text = f"# {ext} {o['name']}\n\n{clean(o.get('description'))}"
        if o.get('x_mitre_detection'):
            text += f"\n\n## Detection\n\n{clean(o['x_mitre_detection'])}"
        phases = [p['phase_name'] for p in o.get('kill_chain_phases', [])]
        docs.append({'doc_id': f'attack/{ext}', 'title': o['name'], 'doc_type': phases[0] if phases else 'none', 'text': text.strip()})
    docs.sort(key=lambda d: d['doc_id'])
    write('attack', docs, f"""# MITRE ATT&CK Enterprise {ATTACK_VERSION} (technique pages)

Source: https://github.com/mitre-attack/attack-stix-data, commit {ATTACK_REF}, enterprise-attack/enterprise-attack-{ATTACK_VERSION}.json.
Changes: technique descriptions and detection notes only; citations and link targets removed; revoked and deprecated
techniques skipped.

"(c) 2026 The MITRE Corporation. This work is reproduced and distributed with the permission of The MITRE Corporation."

License: The MITRE Corporation (MITRE) hereby grants you a non-exclusive, royalty-free license to use ATT&CK(R) for
research, development, and commercial purposes. Any copy you make for such purposes is authorized provided that you
reproduce MITRE's copyright designation and this license in any such copy. ATT&CK(R) is a registered trademark of The
MITRE Corporation.
""")


def fetch_docker():
    docs = hugo_pages('https://github.com/docker/docs.git', DOCKER_REF, 'content/manuals', 'docker', 'manuals',
                      skip=lambda rel: 'release-notes' in str(rel) or rel.name.startswith('release'))
    write('docker', docs, f"""# Docker documentation (manuals)

Source: https://github.com/docker/docs, commit {DOCKER_REF}, path content/manuals.
License: Apache License 2.0. Copyright Docker, Inc.
Changes: Hugo front matter, shortcodes and HTML comments removed; release notes and pages under 300 characters skipped.
""")


def fetch_m365():
    docs = hugo_pages('https://github.com/MicrosoftDocs/microsoft-365-docs.git', M365_REF, 'microsoft-365/admin', 'm365', 'admin', clean=strip_learn)
    write('m365', docs, f"""# Microsoft 365 admin documentation

Source: https://github.com/MicrosoftDocs/microsoft-365-docs, commit {M365_REF}, path microsoft-365/admin.
License: Creative Commons Attribution 4.0 International (CC BY 4.0). (c) Microsoft Corporation.
Changes: front matter, images, includes, videos, callout markers, inline HTML tags and link targets removed; pages under 300 characters skipped.
""")


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('k8s', 'all'):
        fetch_k8s()
    if which in ('attack', 'all'):
        fetch_attack()
    if which in ('docker', 'all'):
        fetch_docker()
    if which in ('m365', 'all'):
        fetch_m365()
