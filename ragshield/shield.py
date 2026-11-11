"""Ingestion gate + retrieval-time trust filter.

ingest(): chunk -> hash -> detector scores -> sign -> ledger -> index (quarantined chunks are indexed but never served).
retrieve(): over-fetch from the index, verify each chunk against the ledger, apply the policy, re-rank by similarity x trust."""
import time
from dataclasses import dataclass

import numpy as np

from . import chunking, delta, hidden
from .provenance import TIERS, new_record

UNSCORED = 'unscored: no injection score'


@dataclass
class Policy:
    injection_block: float = 0.90   # quarantine at ingestion and drop at retrieval (tiers in block_tiers)
    injection_warn: float = 0.50    # serve with a warning label and reduced trust
    # Generic injection classifiers misfire on security text (logs, EDR exports, incident reports): on the clean
    # Vandelay corpus they blocked 3.7% of chunks. A signed chunk from a trusted source is therefore never deleted on a
    # classifier score alone; it is served with a warning and sent to the review queue.
    block_tiers: tuple = ('low', 'medium')
    warn_weight: float = 0.85       # ranking multiplier for any chunk served with a warning (swept in the experiments)
    min_rank: float = 0.25          # chunks from block_tiers whose ranking weight falls below this are dropped
    delta_injection: float = 0.50   # an added passage scored on its own (no dilution by the surrounding page)
    delta_offtopic: float = 0.45    # 1 - max cosine to the previous version; calibrated on benign edits in experiment 3
    overfetch: int = 3              # fetch k * overfetch candidates before filtering, and more if too few survive
    max_candidates: int = 500       # stop there and log a 'starved' event: each candidate costs a signature check
    lockdown: bool = False          # incident kill switch: serve only verified chunks with no warning at all
    hidden_check: bool = True       # flag invisible, tag and look-alike characters (the classifier always scores cleaned text)
    # Low-trust content that looks unlike anything in the trusted corpus is dropped, not just flagged. In the PoisonedRAG
    # experiment, off-topic poison that was flagged but still reached the model worked 84% of the time, and 80% when the model
    # was shown the flag (it hedged, but repeated the false answer). Dropped, it worked 0% of the time.
    outlier_block_tiers: tuple = ('low',)
    # Fail-open vs fail-closed: a chunk with no injection score (no classifier configured, or it failed) is never treated as
    # safe. 'warn' serves it with an "unscored" label (fail-open); 'drop' refuses to serve it (fail-closed).
    unscored: str = 'warn'
    # Retrieval-frequency probe (probe_frequency): a chunk in the top k for an unusual share of typical queries is flagged.
    # Both limits apply, so a small index, where every chunk is retrieved often, doesn't flag everything.
    frequency_min_share: float = 0.10   # retrieved for at least this share of the probe queries
    frequency_ratio: float = 10.0       # and at least this many times its expected share (k / number of chunks)
    frequency_min_queries: int = 20     # fewer probe queries than this flag nothing
    frequency_block_tiers: tuple = ('low',)


class Shield:
    def __init__(self, ledger, signer, embedder, index, injection=None, outlier=None, policy=None):
        self.ledger, self.signer, self.embedder, self.index = ledger, signer, embedder, index
        self.injection, self.outlier, self.policy = injection, outlier, policy or Policy()
        ledger.trust_key(signer)
        self.events = []  # (time, chunk_id, action, reason): the audit trail the dashboard shows

    # Ingestion ---------------------------------------------------------------------------------

    def ingest(self, doc_id, text, source_id, trust_tier, chunk_size=800, overlap=100, metadata=None):
        """Ingest a document, or a new version of one: the previous version's chunks are replaced, and what was added
        since the last signed version is scored on its own (delta scoring)."""
        paragraphs = delta.paragraphs(text)
        version, previous = self.ledger.latest_version(doc_id)
        additions = self._score_additions(delta.additions(previous, paragraphs), previous) if version else []
        if version:
            self.index.remove_doc(doc_id)
            self.ledger.delete_doc_chunks(doc_id)
        self.ledger.add_version(doc_id, paragraphs)
        pieces = chunking.chunk(text, chunk_size, overlap)
        inspected = [hidden.inspect(piece) for piece in pieces]  # (what a reader sees, what was hidden)
        vectors = self.embedder.passages(pieces)
        scored = self._injection([seen for seen, _ in inspected], doc_id)
        injection = scored if scored is not None else np.zeros(len(pieces))
        records = []
        for i, (piece, vector, inj, (_, hid)) in enumerate(zip(pieces, vectors, injection, inspected)):
            chunk_id = f'{doc_id}::{i}'
            record = self.signer.sign(new_record(chunk_id, doc_id, source_id, trust_tier, i, piece))
            record.scores = {'injection': round(float(inj), 4)} if scored is not None else {'unscored': True}
            if self.policy.hidden_check and hidden.total(hid):
                record.scores['hidden'] = {k: n for k, n in hid.items() if n}
                if trust_tier not in self.policy.block_tiers:
                    record.scores['review'] = True
                self._event(chunk_id, 'review', hidden.describe(hid) + ' at ingestion')
            mine = [a for a in additions if delta.contains(piece, a['text'])]
            if mine:
                record.scores['delta'] = {'injection': max(a['injection'] for a in mine),
                                          'offtopic': max(a['offtopic'] for a in mine), 'added': len(mine)}
                if any(a['flag'] for a in mine):
                    record.scores['delta_flag'] = True
                    self._event(chunk_id, 'review', f'unreviewed edit in version {version + 1}')
            if inj >= self.policy.injection_block:
                if trust_tier in self.policy.block_tiers:
                    self.ledger.put(record)
                    self.ledger.quarantine(chunk_id, f'injection score {inj:.2f}')
                    self._event(chunk_id, 'quarantine', f'injection score {inj:.2f} at ingestion')
                else:
                    record.scores['review'] = True
                    self.ledger.put(record)
                    self._event(chunk_id, 'review', f'injection score {inj:.2f} on a trusted source')
            else:
                self.ledger.put(record)
            self.index.add(chunk_id, piece, vector, {'doc_id': doc_id, 'source_id': source_id, **(metadata or {})})
            records.append(record)
        return records

    def _score_additions(self, added, previous):
        """Score each added passage on its own: injection language, and how unrelated it is to the earlier version."""
        if not added:
            return []
        p = self.policy
        new_vecs = self.embedder.passages(added)
        old_vecs = self.embedder.passages(previous)
        offtopic = 1 - (new_vecs @ old_vecs.T).max(axis=1)
        inspected = [hidden.inspect(t) for t in added]
        scored = self._injection([seen for seen, _ in inspected], 'an edit')
        inj = scored if scored is not None else np.zeros(len(added))
        hid = [hidden.total(h) if p.hidden_check else 0 for _, h in inspected]
        return [{'text': t, 'offtopic': round(float(o), 4), 'injection': round(float(s), 4), 'hidden': n,
                 'flag': bool(s >= p.delta_injection or o >= p.delta_offtopic or n > 0)}
                for t, o, s, n in zip(added, offtopic, inj, hid)]

    def _injection(self, texts, what):
        """Injection scores, or None when there's no classifier or it fails. Those chunks are labeled unscored, never passed as safe."""
        if self.injection is None:
            return None
        try:
            return self.injection.score(texts)
        except Exception as error:  # a crashed classifier must not stop ingestion or pass content as safe
            self._event('-', 'unscored', f'injection classifier failed on {what}: {type(error).__name__}')
            return None

    def fit_outlier(self, tiers=('high',), fit_ids=None):
        """Fit the outlier model on trusted chunks (or on a reviewed snapshot, fit_ids), then score every chunk."""
        if fit_ids is not None:
            wanted = set(fit_ids)
            trusted = [i for i, cid in enumerate(self.index.ids) if cid in wanted]
        else:
            trusted = [i for i, cid in enumerate(self.index.ids) if self.ledger.get(cid)[0].trust_tier in tiers]
        self.outlier.fit(self.index.matrix[trusted])
        flags, scores = self.outlier.flag(self.index.matrix), self.outlier.score(self.index.matrix)
        for cid, flagged, s in zip(self.index.ids, flags, scores):
            record, _, _ = self.ledger.get(cid)
            record.scores.update({'outlier': round(float(s), 4), 'outlier_flag': bool(flagged)})
            self.ledger.put(record)  # scores are outside the signed fields, so the signature stays valid
        return int(flags.sum())

    def probe_frequency(self, queries, k=5):
        """Retrieval-frequency probe: run typical queries and flag chunks that land in the top k for an unusual share of them.

        A chunk written to match many unrelated questions shows up here even when its text and vector look normal. Run it on a
        schedule, as you would fit_outlier; the scores sit outside the signed fields. Returns the number of chunks flagged."""
        p, ids = self.policy, list(self.index.ids)
        counts = dict.fromkeys(ids, 0)
        for question in queries:
            for hit in self.index.search(self.embedder.query(question), k):
                counts[hit['chunk_id']] = counts.get(hit['chunk_id'], 0) + 1
        n = len(queries)
        limit = max(p.frequency_min_share, p.frequency_ratio * k / max(len(ids), 1))
        flagged = 0
        for chunk_id in ids:
            record = self.ledger.get(chunk_id)[0]
            if record is None:  # written around the ingestion gate; retrieval drops it anyway
                continue
            share = counts[chunk_id] / n if n else 0.0
            flag = n >= p.frequency_min_queries and share >= limit
            record.scores.update({'frequency': {'share': round(share, 4), 'queries': n, 'k': k}, 'frequency_flag': flag})
            self.ledger.put(record)
            if flag:
                flagged += 1
                self._event(chunk_id, 'review', f'in the top {k} for {share:.0%} of {n} probe queries')
        return flagged

    # Retrieval ---------------------------------------------------------------------------------

    def decide(self, hit):
        status, record = self.ledger.verify(hit['chunk_id'], hit['text'])
        if status != 'verified':
            return {'action': 'drop', 'reason': status, 'trust': 0.0}
        p, tier = self.policy, record.trust_tier
        inj = record.scores.get('injection', 0.0)
        blockable = tier in p.block_tiers
        if inj >= p.injection_block and blockable:
            return {'action': 'drop', 'reason': f'injection {inj:.2f}', 'trust': 0.0}
        reasons = []
        if record.scores.get('unscored'):
            if p.unscored == 'drop':
                return {'action': 'drop', 'reason': UNSCORED, 'trust': 0.0}
            reasons.append(UNSCORED)
        if inj >= p.injection_block:
            reasons.append(f'awaiting review: possible instructions ({inj:.2f})')
        elif inj >= p.injection_warn:
            reasons.append(f'possible instructions ({inj:.2f})')
        if record.scores.get('outlier_flag'):
            if tier in p.outlier_block_tiers:
                return {'action': 'drop', 'reason': 'embedding outlier from a low-trust source', 'trust': 0.0}
            reasons.append('embedding outlier')
        if record.scores.get('frequency_flag'):
            f = record.scores['frequency']
            if tier in p.frequency_block_tiers:
                return {'action': 'drop', 'reason': 'retrieved for too many unrelated queries', 'trust': 0.0}
            reasons.append(f"in the top {f['k']} for {f['share']:.0%} of probe queries")
        if record.scores.get('hidden'):
            if blockable:
                return {'action': 'drop', 'reason': hidden.describe(record.scores['hidden']), 'trust': 0.0}
            reasons.append(hidden.describe(record.scores['hidden']))
        if record.scores.get('delta_flag'):
            d = record.scores['delta']
            if blockable:
                return {'action': 'drop', 'reason': 'unreviewed suspicious edit', 'trust': 0.0}
            reasons.append(f"unreviewed edit (off-topic {d['offtopic']:.2f}, instructions {d['injection']:.2f})")
        if tier == 'low':
            reasons.append('low-trust source')
        if p.lockdown and reasons:
            return {'action': 'drop', 'reason': 'lockdown: ' + ', '.join(reasons), 'trust': 0.0}
        weight = TIERS[tier] * (p.warn_weight if reasons else 1.0)  # used for ranking and shown to the application
        if blockable and weight < p.min_rank:
            return {'action': 'drop', 'reason': ', '.join(reasons), 'trust': round(weight, 3)}
        return {'action': 'warn' if reasons else 'pass', 'reason': ', '.join(reasons), 'trust': round(weight, 3),
                'source_id': record.source_id, 'trust_tier': tier}

    def _keep(self, hits):
        kept = []
        for hit in hits:
            decision = self.decide(hit)
            if decision['action'] == 'drop':
                self._event(hit['chunk_id'], 'drop', decision['reason'])
                continue
            kept.append({**hit, 'trust': decision, 'final_score': hit['score'] * decision['trust']})
        return kept

    def filter(self, hits, k):
        kept = self._keep(hits)
        kept.sort(key=lambda h: -h['final_score'])
        return kept[:k]

    def retrieve(self, question, k=5, shielded=True):
        query = self.embedder.query(question)
        if not shielded:
            return self.index.search(query, k)
        # Dropped chunks must not use up the candidates, or a flood of bad chunks on a topic hides the good ones below them.
        cap = max(self.policy.max_candidates, k)
        fetch, seen, kept = min(k * self.policy.overfetch, cap), 0, []
        while True:
            hits = self.index.search(query, fetch)
            kept += self._keep(hits[seen:])
            seen = len(hits)
            if len(kept) >= k or seen < fetch or fetch >= cap:
                break
            fetch = min(fetch * 2, cap)
        if len(kept) < k and seen >= cap:
            self._event('-', 'starved', f'{len(kept)} of {k} passages left after {seen} candidates')
        kept.sort(key=lambda h: -h['final_score'])
        return kept[:k]

    def _event(self, chunk_id, action, reason):
        self.events.append((time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), chunk_id, action, reason))
