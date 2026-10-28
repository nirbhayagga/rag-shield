# RAG Shield

Signed provenance, poisoning detection and a retrieval-time trust filter for RAG pipelines.

## How it works

| Stage | Module | What it does |
|---|---|---|
| Ingestion signer | `ragshield/provenance.py` | SHA-256 + Ed25519 signature per chunk, SQLite ledger, key revocation, document versions |
| Delta scoring | `ragshield/delta.py`, `shield.py` | On re-ingestion, isolates new paragraphs (or new sentences in modified paragraphs) and scores only those |
| Detectors | `ragshield/detectors.py` | Injection classifier (ProtectAI DeBERTa-v3, or the fine-tuned one) and an embedding-outlier model (LOF over bge-base) |
| Trust filter | `ragshield/shield.py` | Verifies each retrieved chunk, drops tampered/unknown/quarantined chunks, warns and queues suspicious trusted content, re-ranks by similarity × trust; `lockdown` kill switch |
| Index | `ragshield/index.py` | Minimal cosine index with the adapter surface for ChromaDB/pgvector |

The policy is deliberately asymmetric. Trusted, signed sources are never deleted on a classifier score alone (warn + review).
Low and medium-trust sources are dropped on a high injection score or a flagged edit (off-topic or injection-like additions),
and low-trust chunks that also carry a warning are dropped when their ranking weight falls below `Policy.min_rank`.
Unverifiable chunks are always dropped.

## Data

| Path | Source | License |
|---|---|---|
| `data/vandelay/` | Private corpus and questions | private |
| `vendor/eval_harness.py` | Evaluation harness | private |
| `data/external/bipia/` | Microsoft BIPIA attack files, train and test splits | MIT (LICENSE included) |
| `data/splits.json` | Document-level train/test split of Vandelay, created by the fine-tune script and reused by every experiment that uses a tuned model | — |

No new attack content is written anywhere. Published attacks are inserted unchanged.

### Private corpus (not in git)

The Vandelay corpus, `questions.yaml` and the evaluation harness are private and not committed. To set them up:

```bash
unzip private-corpus.zip 'reproducibility_sources/*' -d /tmp/a4
mkdir -p data/vandelay vendor
cp -r /tmp/a4/reproducibility_sources/corpus /tmp/a4/reproducibility_sources/questions.yaml data/vandelay/
cp /path/to/eval_harness.py vendor/              # unmodified
```

Without them, the unit tests still run (one class-contract test is skipped), but the experiments need the corpus.
`results/` is also kept local because experiment outputs quote corpus text.

## Run (CPU)

```bash
cd rag-shield
python3.12 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests        # 20 tests, no model downloads
cd experiments
../.venv/bin/python clean_baseline.py                # exp 1: cost on clean data
../.venv/bin/python attack_bipia.py                  # exp 2: T1/T2/T2b/T3 with BIPIA test attacks
../.venv/bin/python delta_edits.py                   # exp 3: delta scoring vs benign edits
```

Model calls are cached in `results/cache/` by model and text hash, so reruns and policy sweeps are fast.

### Scan your own corpus (index exposure report)

```bash
.venv/bin/python -m ragshield.report /path/to/docs --out report.md        # add --device cuda on a GPU
```

Ingests every text-like file as a trusted source, signs the chunks, scores them with the injection classifier and the
outlier model, and writes a Markdown report: how many chunks would be served with a warning or sent to review, the top
chunks to look at, and what share of the index raises no flag. Read-only and local; nothing leaves the machine. On the
class corpus (36 files, 261 chunks) it takes about 6 minutes on CPU and flags 7% of chunks, most of them security logs
and incident reports that generic classifiers misread, which is exactly what the review queue is for.

## Run on the RTX 4080 (or 2080)

```bash
git clone https://github.com/nirbhayagga/rag-shield.git && cd rag-shield
# copy the private files (gitignored) from the machine that has them: data/vandelay/ and vendor/eval_harness.py
python3.12 -m venv .venv          # or: mamba create -n ragshield python=3.12 && mamba activate ragshield
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
cd experiments
../.venv/bin/python finetune_injection.py --device cuda --dry-run                     # smoke test
../.venv/bin/python finetune_injection.py --device cuda --name ragshield-injection-v1-nodeepset   # about a minute on a 4080
../.venv/bin/python finetune_injection.py --device cuda --deepset                     # also trains on deepset
# RTX 2080 (Turing has no fast bf16): add --precision fp16 --batch 8 --grad-accum 2
```

Each run writes `models/<name>/` and `results/finetune_<name>.json`. The report compares base and fine-tuned models on
**held-out documents only**, with BIPIA **test** attacks, in three styles (whole chunk, delta, inline): AUROC, TPR at 1% and
5% FPR, and FPR at 0.5. deepset's test split is always reported, as `deepset_out_of_domain` when deepset was not used for
training and as `deepset_in_domain` when it was.

### End-to-end with a fine-tuned classifier

The fine-tune saw the 25 training documents, so the pipeline experiments must use only the 10 held-out ones:

```bash
../.venv/bin/python attack_bipia.py --injection-model ../models/ragshield-injection-v1 --split test
../.venv/bin/python delta_edits.py  --injection-model ../models/ragshield-injection-v1 --split test
../.venv/bin/python attack_bipia.py --split test     # base classifier on the same documents, for comparison
../.venv/bin/python delta_edits.py  --split test
```

These write `results/<experiment>-<model>-<split>.json` (`base` for the off-the-shelf classifier). The shipped model is
`ragshield-injection-v1` (trained with deepset): the `-nodeepset` variant scores slightly better on the Vandelay held-out
documents but flags 62.5% of deepset's benign prompts, so it has learned the corpus rather than injection. Every number in
the proposal and on the site comes from `-v1`.

### Reading the numbers

- `suppressed` = `dropped` (the filter removed the chunk) + `outranked` (kept, but the trust weight pushed it below the
  top k). Most low-trust suppressions are outranking, not detection; the T3 and delta-edit tables are the detection claims.
- `silent_attack_success_rate` is 0 for low-trust sources *by policy*: a low-tier chunk is always served with a label.
- The calibrated `delta_injection` threshold is the 1 − target FPR quantile of benign edits. With 63 calibration edits and
  the off-the-shelf classifier that quantile is ≈1.0, so for the base model the injection channel is effectively off and
  its delta-scoring gain comes from the off-topic score alone. The fine-tuned model separates the two, so its threshold is
  informative (0.96 and 0.008 in the two runs).
- `tpr_at_1pct_fpr` is only reported when the negative set has at least 100 items.

## Results

See `results/*.json`. The summary is kept in the proposal's §4 table.

## Limits

- The ledger is SQLite with a local PEM key. Production needs Postgres and a KMS or HSM.
- Document version history and detector scores are stored unsigned: the signature covers the chunk's identity, tier, position
  and hash, so anyone with ledger write access can lower a score. Signing verdicts (or keying quarantine to a reviewer) is
  November work.
- Revoking a key drops every chunk it ever signed; re-signing on rotation is not implemented.
- The ChromaDB and pgvector adapters, the review dashboard and Haiku reviewer summaries are November work.
- Benign edits in experiment 3 are built from other Vandelay documents of the same type. That is likely *harder* (more
  off-topic) than real edits, so its false-alarm rate is a conservative estimate.
