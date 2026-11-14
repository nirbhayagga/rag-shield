# Changelog

## 0.1.0 (unreleased)

First public release.

- **Signed provenance.** Every chunk's hash, source, trust tier and position are signed with Ed25519 at ingestion and
  kept in a SQLite ledger. At retrieval, a chunk that is unknown, changed, badly signed or signed by a revoked key is
  dropped.
- **Delta scoring.** A re-ingested page is compared with its last signed version, and only the added passages are
  scored for instructions, off-topic content and hidden characters.
- **Held edits.** A flagged edit is held for review while the previous version keeps serving (`Policy.hold_edits`, on by
  default). `Shield.approve`, `reject` and `quarantine` take the SHA-256 of the text the reviewer read, so a decision never
  covers text nobody saw.
- **Editor and version.** `ingest(..., editor=...)` signs who made a version, and the version number, into every chunk
  record. Retrieved hits carry `version`, `editor` and `key_id`, and `Ledger.versions(doc_id)` lists each version's
  editor. Records without them still verify.
- **Fact changes.** An edit that swaps a link for one to another site, or replaces an email address, phone number, IP
  address or amount the page already had, is held (`Policy.fact_changes`, on by default).
- **Plain-language reasons.** Each held edit says what it does ("adds instructions aimed at the AI assistant", "points a
  link to a site the page didn't link to before", ...) and who made it, so a reviewer doesn't need to read scores.
- **Trust filter.** Three trust tiers re-rank results by similarity × trust. Low and medium-trust chunks with a high
  injection score or hidden characters, and low-trust outliers, are dropped. Trusted chunks are never dropped on a score
  alone. Chunks with no injection score carry an `unscored` label (`Policy.unscored = 'drop'` fails closed instead). A
  `lockdown` switch serves only verified chunks with no warnings.
- **Detectors.** An injection classifier, an embedding outlier model and a hidden-character check (zero-width, look-alike,
  fullwidth and tag characters). The classifier scores the cleaned text.
- **Contradiction check (optional).** `Shield(contradiction=ContradictionDetector())` holds an edit when an added
  sentence naming a host, ticket, CVE or version contradicts what the page said about it, and nothing on the page
  supports it.
- **Flood resistance.** Retrieval keeps fetching candidates until k survive, up to `Policy.max_candidates`, and logs a
  `starved` event when it runs out.
- **Integrations.** LangChain and LlamaIndex retrievers, ChromaDB and pgvector adapters, and a sync client that sends
  flagged chunks and held edits to a review dashboard and applies the reviewers' decisions.
- **Exposure report.** `ragshield-report <folder>` scans a folder of documents locally and writes a Markdown report.
- **Experiment.** `Shield.probe_frequency` flags chunks that show up in the results for an unusual share of probe
  queries. So far it has only been tested on synthetic data.
- **Benchmarks.** The scripts in `experiments/` reproduce every number in the README.
