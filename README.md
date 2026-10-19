# RAG Shield

Signed provenance, poisoning detection and a retrieval-time trust filter for RAG pipelines.
The proposal is in [docs/proposals/rag-shield.md](../../docs/proposals/rag-shield.md) and the interview kit in
[docs/rag-shield-interview-kit.md](../../docs/rag-shield-interview-kit.md).

## How it works

| Stage | Module | What it does |
|---|---|---|
| Ingestion signer | `ragshield/provenance.py` | SHA-256 + Ed25519 signature per chunk, SQLite ledger, key revocation, document versions |
| Delta scoring | `ragshield/delta.py`, `shield.py` | On re-ingestion, isolates new paragraphs (or new sentences in modified paragraphs) and scores only those |
| Detectors | `ragshield/detectors.py` | Injection classifier (ProtectAI DeBERTa-v3, or the fine-tuned one) and an embedding-outlier model (LOF over bge-base) |
| Trust filter | `ragshield/shield.py` | Verifies each retrieved chunk, drops tampered/unknown/quarantined chunks, warns and queues suspicious trusted content, re-ranks by similarity × trust; `lockdown` kill switch |
| Index | `ragshield/index.py` | Minimal cosine index with the adapter surface for ChromaDB/pgvector |

The policy is deliberately asymmetric. Trusted, signed sources are never deleted on a classifier score alone (warn + review).
Low and medium-trust sources are dropped on high scores. Unverifiable chunks are always dropped.

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
cd projects/rag-shield
python3.12 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests        # 15 tests, no model downloads
cd experiments
../.venv/bin/python clean_baseline.py                # exp 1: cost on clean data
../.venv/bin/python attack_bipia.py                  # exp 2: T1/T2/T2b/T3 with BIPIA test attacks
../.venv/bin/python delta_edits.py                   # exp 3: delta scoring vs benign edits
```

Model calls are cached in `results/cache/` by model and text hash, so reruns and policy sweeps are fast.

## Run on the RTX 4080 (or 2080)

```bash
# from this laptop:
rsync -a --exclude .venv --exclude results/cache projects/rag-shield/ <gpu-host>:rag-shield/
# then on the GPU machine:
cd ~/rag-shield
python3.12 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt pyarrow
.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
cd experiments
../.venv/bin/python finetune_injection.py --device cuda --deepset          # 4080: bf16, about 5-10 minutes
# RTX 2080 (Turing has no fast bf16):
../.venv/bin/python finetune_injection.py --device cuda --precision fp16 --batch 8 --grad-accum 2 --deepset
```

The fine-tune output goes to `models/ragshield-injection-v1/` and the report to `results/finetune_ragshield-injection-v1.json`.
The report compares base and fine-tuned models on **held-out documents only**, with BIPIA **test** attacks, in three styles
(whole chunk, delta, inline). It reports AUROC, TPR at 1% and 5% FPR, FPR at 0.5, and deepset's test split as an
out-of-domain check. Copy `results/` and `data/splits.json` back to this machine afterwards.

Smoke-test the whole pipeline on CPU first with `--dry-run`.

## Results

See `results/*.json`. The summary is kept in the proposal's §4 table.

## Limits

- The ledger is SQLite with a local PEM key. Production needs Postgres and a KMS or HSM.
- Document version history is stored unsigned.
- The ChromaDB and pgvector adapters, the review dashboard and Haiku reviewer summaries are November work.
- Benign edits in experiment 3 are built from other Vandelay documents of the same type. That is likely *harder* (more
  off-topic) than real edits, so its false-alarm rate is a conservative estimate.
