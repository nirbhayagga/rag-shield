"""Signed provenance for every chunk: Ed25519 over a canonical record, kept in a SQLite ledger.

Threat this closes: content that reaches the vector store without passing the ingestion gate, or is edited after it.
Such a chunk either has no ledger record or its text no longer matches the signed hash, so it is rejected at retrieval.
It does not, by itself, catch poisoned content that arrives through an approved source; the detectors do that."""
import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

TIERS = {'high': 1.0, 'medium': 0.7, 'low': 0.4}
SIGNED_FIELDS = ('chunk_id', 'doc_id', 'source_id', 'trust_tier', 'position', 'text_sha256', 'ingested_at', 'key_id')
# Signed whenever they are set. A record from before they existed still verifies, and setting or clearing one on a stored
# record changes the signed bytes, so the signature no longer matches.
OPTIONAL_SIGNED = ('version', 'editor')


def sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source_id: str
    trust_tier: str
    position: int
    text_sha256: str
    ingested_at: str
    key_id: str
    version: int = 0   # the document version this chunk belongs to (Ledger.versions)
    editor: str = ''   # who made that version, as the source system names them (a user ID or email); '' if not given
    signature: str = ''
    scores: dict = field(default_factory=dict)  # detector scores at ingestion; not signed, may be re-computed

    def canonical(self):
        fields = {k: getattr(self, k) for k in SIGNED_FIELDS}
        fields.update({k: getattr(self, k) for k in OPTIONAL_SIGNED if getattr(self, k)})
        return json.dumps(fields, sort_keys=True, separators=(',', ':')).encode()


class Signer:
    """An Ed25519 key. In production the private key lives in a KMS or HSM; here it is a 0600 PEM file."""

    def __init__(self, private_key):
        self.private_key = private_key
        raw = private_key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.public_raw = raw
        self.key_id = hashlib.sha256(raw).hexdigest()[:16]

    @classmethod
    def generate(cls):
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def load_or_create(cls, path):
        path = Path(path)
        if path.exists():
            return cls(serialization.load_pem_private_key(path.read_bytes(), password=None))
        signer = cls.generate()
        path.parent.mkdir(parents=True, exist_ok=True)
        pem = signer.private_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                              serialization.NoEncryption())
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as f:  # private from the first byte
            f.write(pem)
        return signer

    def sign(self, record):
        record.key_id = self.key_id
        record.signature = self.private_key.sign(record.canonical()).hex()
        return record


class Ledger:
    """Provenance records keyed by chunk id, plus the public keys allowed to sign them."""

    def __init__(self, path=':memory:'):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        self.db.executescript('''
            create table if not exists keys(key_id text primary key, public_raw blob not null, revoked integer default 0);
            create table if not exists chunks(chunk_id text primary key, record text not null, quarantined integer default 0, reason text);
            create table if not exists versions(doc_id text not null, version integer not null, paragraphs text not null,
              ingested_at text not null, primary key(doc_id, version));
            create table if not exists held(id integer primary key, doc_id text not null, text text not null, text_sha256 text not null,
              source_id text not null, trust_tier text not null, chunk_size integer not null, overlap integer not null,
              metadata text not null, additions text not null, reason text not null, held_at text not null,
              status text not null default 'held', reviewer text, decided_at text);
            create unique index if not exists one_hold_per_doc on held(doc_id) where status = 'held';
            create table if not exists blocked(text_sha256 text primary key, chunk_id text not null, reason text not null,
              reviewer text, blocked_at text not null);
        ''')
        for table in ('versions', 'held'):  # added after the first release candidate: upgrade older ledger files in place
            if 'editor' not in {row[1] for row in self.db.execute(f'pragma table_info({table})')}:
                with self.db:
                    self.db.execute(f'alter table {table} add column editor text')
        self._keys = {}

    def trust_key(self, signer_or_public_raw):
        raw = getattr(signer_or_public_raw, 'public_raw', signer_or_public_raw)
        key_id = hashlib.sha256(raw).hexdigest()[:16]
        with self.lock, self.db:
            self.db.execute('insert or ignore into keys(key_id, public_raw) values(?,?)', (key_id, raw))
        return key_id

    def revoke_key(self, key_id):
        with self.lock, self.db:
            self.db.execute('update keys set revoked=1 where key_id=?', (key_id,))
        self._keys.pop(key_id, None)

    def public_key(self, key_id):
        if key_id not in self._keys:
            row = self.db.execute('select public_raw, revoked from keys where key_id=?', (key_id,)).fetchone()
            if not row or row[1]:
                return None
            self._keys[key_id] = Ed25519PublicKey.from_public_bytes(row[0])
        return self._keys[key_id]

    def put(self, record):
        with self.lock, self.db:
            # upsert on the record only: 'insert or replace' would re-insert the row and reset quarantined/reason
            self.db.execute('insert into chunks(chunk_id, record) values(?,?) on conflict(chunk_id) do update set record=excluded.record',
                            (record.chunk_id, json.dumps(asdict(record))))

    def get(self, chunk_id):
        row = self.db.execute('select record, quarantined, reason from chunks where chunk_id=?', (chunk_id,)).fetchone()
        if not row:
            return None, False, None
        return ChunkRecord(**json.loads(row[0])), bool(row[1]), row[2]

    def quarantine(self, chunk_id, reason):
        with self.lock, self.db:
            self.db.execute('update chunks set quarantined=1, reason=? where chunk_id=?', (reason, chunk_id))

    def latest_version(self, doc_id):
        """(version number, paragraph texts) of the last ingested version, or (0, [])."""
        row = self.db.execute('select version, paragraphs from versions where doc_id=? order by version desc limit 1',
                              (doc_id,)).fetchone()
        return (row[0], json.loads(row[1])) if row else (0, [])

    def add_version(self, doc_id, paragraphs, editor=None):
        version = self.latest_version(doc_id)[0] + 1
        with self.lock, self.db:
            self.db.execute('insert into versions(doc_id, version, paragraphs, ingested_at, editor) values(?,?,?,?,?)',
                            (doc_id, version, json.dumps(paragraphs), now(), editor))
        return version

    def versions(self, doc_id):
        """Every version of a document that was served, oldest first: when it was ingested and who made it."""
        return [{'version': v, 'ingested_at': at, 'editor': editor} for v, at, editor in self.db.execute(
            'select version, ingested_at, editor from versions where doc_id=? order by version', (doc_id,))]

    def delete_doc_chunks(self, doc_id):
        with self.lock, self.db:
            self.db.execute("delete from chunks where chunk_id like ? escape '\\'", (doc_id.replace('_', '\\_').replace('%', '\\%') + '::%',))

    def records(self):
        return [ChunkRecord(**json.loads(r[0])) for r in self.db.execute('select record from chunks order by chunk_id')]

    # Review: versions held back until someone approves them, and text a reviewer quarantined -------------------------

    def hold(self, doc_id, text, source_id, trust_tier, chunk_size, overlap, metadata, additions, reason, editor=None):
        """Store a version for review in place of serving it. A newer held version of the same document replaces it."""
        with self.lock, self.db:
            self.db.execute("update held set status='superseded', decided_at=? where doc_id=? and status='held'", (now(), doc_id))
            self.db.execute('''insert into held(doc_id, text, text_sha256, source_id, trust_tier, chunk_size, overlap, metadata,
                                 additions, reason, held_at, editor) values(?,?,?,?,?,?,?,?,?,?,?,?)''',
                            (doc_id, text, sha256(text), source_id, trust_tier, chunk_size, overlap, json.dumps(metadata, default=str),
                             json.dumps(additions), reason, now(), editor))

    def held(self, doc_id=None):
        """Versions waiting for review, at most one per document, oldest first."""
        query = "select * from held where status='held'" + (' and doc_id=?' if doc_id else '') + ' order by id'
        cursor = self.db.execute(query, (doc_id,) if doc_id else ())
        names = [c[0] for c in cursor.description]
        rows = [dict(zip(names, row)) for row in cursor.fetchall()]
        for row in rows:
            row['metadata'], row['additions'] = json.loads(row['metadata']), json.loads(row['additions'])
        return rows

    def hold_status(self, doc_id, text_sha256):
        """The last review outcome for this exact version of the document, or None."""
        row = self.db.execute('select status, reviewer from held where doc_id=? and text_sha256=? order by id desc limit 1',
                              (doc_id, text_sha256)).fetchone()
        return row if row else None

    def close_hold(self, doc_id, status, reviewer=None):
        with self.lock, self.db:
            return self.db.execute("update held set status=?, reviewer=?, decided_at=? where doc_id=? and status='held'",
                                   (status, reviewer, now(), doc_id)).rowcount

    def block_text(self, text_sha256, chunk_id, reason, reviewer=None):
        with self.lock, self.db:
            self.db.execute('insert or replace into blocked(text_sha256, chunk_id, reason, reviewer, blocked_at) values(?,?,?,?,?)',
                            (text_sha256, chunk_id, reason, reviewer, now()))

    def blocked(self, text_sha256):
        """Why a reviewer quarantined this exact text, or None."""
        row = self.db.execute('select reason from blocked where text_sha256=?', (text_sha256,)).fetchone()
        return row[0] if row else None

    def verify(self, chunk_id, text):
        """Return (status, record). Status is one of: verified, unknown, tampered, bad_signature, untrusted_key, quarantined."""
        record, quarantined, _ = self.get(chunk_id)
        if record is None:
            return 'unknown', None
        if sha256(text) != record.text_sha256:
            return 'tampered', record
        key = self.public_key(record.key_id)
        if key is None:
            return 'untrusted_key', record
        try:
            key.verify(bytes.fromhex(record.signature), record.canonical())
        except (InvalidSignature, ValueError):
            return 'bad_signature', record
        return ('quarantined' if quarantined else 'verified'), record


def new_record(chunk_id, doc_id, source_id, trust_tier, position, text, version=0, editor=None):
    if trust_tier not in TIERS:
        raise ValueError(f'trust_tier must be one of {list(TIERS)}')
    if editor is not None and not isinstance(editor, str):
        raise TypeError('editor must be a string, such as a user ID or email from the source system')
    return ChunkRecord(chunk_id=chunk_id, doc_id=doc_id, source_id=source_id, trust_tier=trust_tier, position=position,
                       text_sha256=sha256(text), ingested_at=now(), key_id='', version=version, editor=editor or '')
