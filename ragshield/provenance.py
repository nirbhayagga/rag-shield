"""Signed provenance for every chunk: Ed25519 over a canonical record, kept in a SQLite ledger.

Threat this closes: content that reaches the vector store without passing the ingestion gate, or is edited after it.
Such a chunk either has no ledger record or its text no longer matches the signed hash, so it is rejected at retrieval.
It does not, by itself, catch poisoned content that arrives through an approved source; the detectors do that."""
import hashlib
import json
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


def sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


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
    signature: str = ''
    scores: dict = field(default_factory=dict)  # detector scores at ingestion; not signed, may be re-computed

    def canonical(self):
        return json.dumps({k: getattr(self, k) for k in SIGNED_FIELDS}, sort_keys=True, separators=(',', ':')).encode()


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
        path.write_bytes(signer.private_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                          serialization.NoEncryption()))
        path.chmod(0o600)
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
        ''')
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
            self.db.execute('insert or replace into chunks(chunk_id, record) values(?,?)', (record.chunk_id, json.dumps(asdict(record))))

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

    def add_version(self, doc_id, paragraphs):
        version = self.latest_version(doc_id)[0] + 1
        with self.lock, self.db:
            self.db.execute('insert into versions(doc_id, version, paragraphs, ingested_at) values(?,?,?,?)',
                            (doc_id, version, json.dumps(paragraphs), time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        return version

    def delete_doc_chunks(self, doc_id):
        with self.lock, self.db:
            self.db.execute("delete from chunks where chunk_id like ? escape '\\'", (doc_id.replace('_', '\\_').replace('%', '\\%') + '::%',))

    def records(self):
        return [ChunkRecord(**json.loads(r[0])) for r in self.db.execute('select record from chunks order by chunk_id')]

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


def new_record(chunk_id, doc_id, source_id, trust_tier, position, text):
    if trust_tier not in TIERS:
        raise ValueError(f'trust_tier must be one of {list(TIERS)}')
    return ChunkRecord(chunk_id=chunk_id, doc_id=doc_id, source_id=source_id, trust_tier=trust_tier, position=position,
                       text_sha256=sha256(text), ingested_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), key_id='')
