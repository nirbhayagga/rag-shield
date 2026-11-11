"""Send flagged chunks and held edits to the review dashboard, and apply what reviewers decided there.

By default this sends chunk IDs, scores and provenance. Chunk text leaves your environment only with previews=True.
The dashboard never connects to you: pull() asks it for decisions, and each one names the exact text it covers."""
import json
import urllib.parse
import urllib.request

from .provenance import sha256
from .shield import UNSCORED

LOW_TRUST_ONLY = 'low-trust source'
ROUTINE = {LOW_TRUST_ONLY, UNSCORED}  # true of a whole source or setup, so not worth a reviewer's time chunk by chunk
PREVIEW_CHARS = 2000


def snapshot(shield, previews=False):
    """Every chunk flagged right now, every held edit, and the event log. It covers the whole index, so the dashboard
    can clear what is no longer flagged."""
    items = []
    for chunk_id, text in zip(shield.index.ids, shield.index.texts):
        decision = shield.decide({'chunk_id': chunk_id, 'text': text})
        if decision['action'] == 'pass' or set(decision['reason'].split(', ')) <= ROUTINE:
            continue
        record = shield.ledger.get(chunk_id)[0]  # None for a chunk written around the ingestion gate
        item = {'chunk_id': chunk_id, 'doc_id': record.doc_id if record else chunk_id.split('::')[0], 'text_sha256': sha256(text),
                'source_id': record.source_id if record else 'unknown', 'trust_tier': record.trust_tier if record else 'unknown',
                'ingested_at': record.ingested_at if record else None, 'key_id': record.key_id if record else None,
                'action': decision['action'], 'reason': decision['reason'], 'scores': record.scores if record else {}}
        if previews:
            item['preview'] = text[:PREVIEW_CHARS]
        items.append(item)
    for hold in shield.held():  # one item per document: the reviewer approves or rejects the whole edit
        added = hold['additions']
        item = {'chunk_id': hold['doc_id'], 'doc_id': hold['doc_id'], 'text_sha256': hold['text_sha256'],
                'source_id': hold['source_id'], 'trust_tier': hold['trust_tier'], 'ingested_at': hold['held_at'], 'key_id': None,
                'action': 'hold', 'reason': hold['reason'],
                'scores': {'added passages': len(added), 'flagged': sum(a['flag'] for a in added),
                           'instructions': max(a['injection'] for a in added), 'off-topic': max(a['offtopic'] for a in added),
                           'hidden characters': sum(a['hidden'] for a in added)}}
        if previews:  # what the edit adds, which is what the reviewer has to judge
            item['preview'] = '\n\n'.join(('[flagged] ' if a['flag'] else '') + a['text'] for a in added)[:PREVIEW_CHARS]
        items.append(item)
    return {'items': items, 'events': [{'at': at, 'chunk_id': chunk_id, 'action': action, 'reason': reason}
                                       for at, chunk_id, action, reason in shield.events]}


def _call(url, path, api_key, body=None, timeout=30):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != 'https' and parts.hostname not in ('localhost', '127.0.0.1'):
        raise ValueError('the dashboard URL must use https, or the API key would be sent in the clear')
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url.rstrip('/') + path, data=data, method='POST' if data else 'GET',
                                     headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def push(shield, url, api_key, previews=False, timeout=30):
    return _call(url, '/api/v1/sync', api_key, snapshot(shield, previews), timeout)


def apply(shield, decision):
    """Apply one reviewer decision. Returns 'applied', or 'stale' with the reason when the text changed after review."""
    kind, reviewer, digest = decision['kind'], decision.get('reviewer'), decision['text_sha256']
    try:
        if kind == 'approve':
            shield.approve(decision['doc_id'], digest, reviewer)
        elif kind == 'reject':
            shield.reject(decision['doc_id'], digest, reviewer)
        elif kind == 'quarantine':
            shield.quarantine(decision['chunk_id'], digest, reviewer)
        else:
            return {'id': decision.get('id'), 'status': 'stale', 'detail': f'unknown decision {kind!r}'}
    except (LookupError, ValueError) as error:
        return {'id': decision.get('id'), 'status': 'stale', 'detail': str(error)}
    return {'id': decision.get('id'), 'status': 'applied'}


def pull(shield, url, api_key, timeout=30):
    """Fetch reviewers' decisions (approve or reject a held edit, quarantine a chunk), apply them, and report back.

    Run it before push(), so the next snapshot shows the result. A decision about text that changed after it was
    reviewed is refused and reported as stale, never applied to the new text."""
    results = [apply(shield, d) for d in _call(url, '/api/v1/decisions', api_key, timeout=timeout)['decisions']]
    if results:
        _call(url, '/api/v1/decisions/ack', api_key, {'results': results}, timeout)
    return results
