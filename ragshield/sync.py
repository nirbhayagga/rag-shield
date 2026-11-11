"""Send flagged chunks to the review dashboard.

By default this sends chunk IDs, scores and provenance. Chunk text leaves your environment only with previews=True."""
import json
import urllib.parse
import urllib.request

from .provenance import sha256
from .shield import UNSCORED

LOW_TRUST_ONLY = 'low-trust source'
ROUTINE = {LOW_TRUST_ONLY, UNSCORED}  # true of a whole source or setup, so not worth a reviewer's time chunk by chunk
PREVIEW_CHARS = 2000


def snapshot(shield, previews=False):
    """Every chunk flagged right now, plus the event log. It covers the whole index, so the dashboard can clear what is no longer flagged."""
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
    return {'items': items, 'events': [{'at': at, 'chunk_id': chunk_id, 'action': action, 'reason': reason}
                                       for at, chunk_id, action, reason in shield.events]}


def push(shield, url, api_key, previews=False, timeout=30):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != 'https' and parts.hostname not in ('localhost', '127.0.0.1'):
        raise ValueError('the dashboard URL must use https, or the API key would be sent in the clear')
    request = urllib.request.Request(url.rstrip('/') + '/api/v1/sync', data=json.dumps(snapshot(shield, previews)).encode(), method='POST',
                                     headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)
