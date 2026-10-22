"""Fine-tune the injection classifier for security-domain RAG content (run on a CUDA GPU, e.g. an RTX 4080).

Why: the off-the-shelf classifier caught 7% of published attacks inside trusted pages while flagging 3.7% of clean
security logs and incident reports. This trains it on security text so it learns that `lsass.exe`, `rule=` and
incident timelines are normal, while instructions aimed at the model are not.

Data (no new attack content is written; attacks are inserted unchanged):
  positives  BIPIA *train* attacks (MIT) inserted into Vandelay chunks (whole-chunk style), on their own (delta style),
             and appended to a sentence (inline style)
  negatives  Vandelay chunks, paragraphs and sentences
  optional   deepset/prompt-injections train split (Apache-2.0) with --deepset, to keep general coverage
Leakage controls:
  - Vandelay is split BY DOCUMENT (data/splits.json, reused if present): training never sees a test document
  - BIPIA train items train; BIPIA test items are only used for the final report
  - 15% of BIPIA train items and 15% of training negatives are held out for checkpoint selection
Report: base vs fine-tuned on test documents with BIPIA test attacks: AUROC, TPR at 1% and 5% FPR, FPR at 0.5,
plus deepset's test split (out-of-domain when trained without --deepset, in-domain with it). Writes models/<name>/ and
results/finetune_<name>.json.

  python finetune_injection.py --device cuda                 # RTX 4080: bf16, a few minutes
  python finetune_injection.py --device cuda --precision fp16 --batch 8 --grad-accum 2   # RTX 2080 (no fast bf16)
  python finetune_injection.py --dry-run                     # CPU smoke test of the whole pipeline"""
import argparse
import json
import math
import random
import time

import numpy as np

from common import HERE, vandelay
from ragshield import chunking, delta
from ragshield.detectors import INJECTION_MODEL

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--device', default=None, help='cuda, cuda:0, cpu (default: cuda if available)')
parser.add_argument('--base', default=INJECTION_MODEL)
parser.add_argument('--name', default='ragshield-injection-v1')
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--grad-accum', type=int, default=1)
parser.add_argument('--max-len', type=int, default=512)
parser.add_argument('--precision', choices=['auto', 'bf16', 'fp16', 'fp32'], default='auto')
parser.add_argument('--test-frac', type=float, default=0.3)
parser.add_argument('--hosts-per-attack', type=int, default=3)
parser.add_argument('--deepset', action='store_true', help='also train on deepset/prompt-injections (its test split is then in-domain)')
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()

import torch  # noqa: E402  (after argparse so --help is fast)
from sklearn.metrics import roc_auc_score  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
precision = args.precision
if precision == 'auto':
    precision = 'bf16' if device.type == 'cuda' and torch.cuda.is_bf16_supported() else ('fp16' if device.type == 'cuda' else 'fp32')
if args.dry_run:
    args.epochs, args.max_len, args.batch = 1, 128, 8

# ----------------------------------------------------------------------------------------------- data

docs, _ = vandelay()
split_path = HERE / 'data/splits.json'
if split_path.exists():
    split = json.loads(split_path.read_text())
else:
    ids = sorted(d.doc_id for d in docs)
    random.Random(args.seed).shuffle(ids)
    n_test = round(len(ids) * args.test_frac)
    split = {'seed': args.seed, 'test_docs': sorted(ids[:n_test]), 'train_docs': sorted(ids[n_test:]),
             'note': 'Document-level split. Anything trained on train_docs must be evaluated on test_docs only.'}
    split_path.write_text(json.dumps(split, indent=2))
test_docs = set(split['test_docs'])


def load_attacks(name):
    out = []
    for kind in ('text', 'code'):
        for category, items in json.loads((HERE / f'data/external/bipia/{kind}_attack_{name}.json').read_text()).items():
            out += [{'category': f'{kind}:{category}', 'text': t} for t in items]
    return out


def units(doc_ids):
    """Clean text at the three granularities the classifier sees in production: chunks, paragraphs (deltas), sentences."""
    chunks, paras, sents = [], [], []
    for d in docs:
        if d.doc_id not in doc_ids:
            continue
        chunks += chunking.chunk(d.text)
        for p in delta.paragraphs(d.text):
            if len(p) >= 40:
                paras.append(p)
            sents += [s for s in delta.sentences(p) if len(s) >= 30]
    return chunks, paras, sents


def insert(host, payload, rng):
    paras = host.split('\n\n')
    at = rng.choice([0, len(paras) // 2, len(paras)])
    return '\n\n'.join(paras[:at] + [payload] + paras[at:])


def positives(attacks, chunks, sents, rng, hosts_per_attack):
    out = []
    for a in attacks:
        for host in rng.sample(chunks, min(hosts_per_attack, len(chunks))):
            out.append((insert(host, a['text'], rng), 1, 'whole_chunk'))
        out.append((a['text'], 1, 'delta'))
        out.append((rng.choice(sents) + ' ' + a['text'], 1, 'inline'))
    return out


rng = random.Random(args.seed)
train_ids = {d.doc_id for d in docs} - test_docs
tr_chunks, tr_paras, tr_sents = units(train_ids)
te_chunks, te_paras, te_sents = units(test_docs)

train_attacks = load_attacks('train')
rng.shuffle(train_attacks)
n_val = max(1, round(len(train_attacks) * 0.15))
val_attacks, fit_attacks = train_attacks[:n_val], train_attacks[n_val:]
negatives = [(t, 0, 'chunk') for t in tr_chunks] + [(t, 0, 'paragraph') for t in tr_paras] + \
            [(t, 0, 'sentence') for t in rng.sample(tr_sents, min(len(tr_sents), 2 * len(tr_paras)))]
rng.shuffle(negatives)
n_val_neg = max(1, round(len(negatives) * 0.15))
val_neg, fit_neg = negatives[:n_val_neg], negatives[n_val_neg:]
train_set = positives(fit_attacks, tr_chunks, tr_sents, rng, args.hosts_per_attack) + fit_neg
val_set = positives(val_attacks, tr_chunks, tr_sents, rng, 1) + val_neg

test_attacks = load_attacks('test')
test_sets = {
    'whole_chunk': [(insert(rng.choice(te_chunks), a['text'], rng), 1, 'whole_chunk') for a in test_attacks] + [(t, 0, 'chunk') for t in te_chunks],
    'delta': [(a['text'], 1, 'delta') for a in test_attacks] + [(t, 0, 'paragraph') for t in te_paras],
    'inline': [(rng.choice(te_sents) + ' ' + a['text'], 1, 'inline') for a in test_attacks] +
              [(t, 0, 'sentence') for t in rng.sample(te_sents, min(len(te_sents), 300))],
}

# deepset's test split is always reported; it only counts as out-of-domain when its train split was not used.
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
files = list_repo_files('deepset/prompt-injections', repo_type='dataset')
for part in ('train', 'test') if args.deepset else ('test',):
    path = hf_hub_download('deepset/prompt-injections', next(f for f in files if f.endswith('.parquet') and part in f), repo_type='dataset')
    frame = pd.read_parquet(path)
    rows = [(str(t), int(label), 'deepset') for t, label in zip(frame['text'], frame['label'])]
    if part == 'train':
        train_set += rows
    else:
        test_sets['deepset_in_domain' if args.deepset else 'deepset_out_of_domain'] = rows

if args.dry_run:
    train_set = rng.sample(train_set, 32)
    val_set = rng.sample(val_set, 16)
    test_sets = {k: rng.sample(v, min(16, len(v))) for k, v in test_sets.items()}
rng.shuffle(train_set)

# --------------------------------------------------------------------------------------------- model

tokenizer = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForSequenceClassification.from_pretrained(args.base).to(device)
assert model.config.id2label[1].upper() == 'INJECTION', model.config.id2label
autocast = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(precision)


def batches(rows, size, shuffle=False):
    order = list(range(len(rows)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order), size):
        part = [rows[j] for j in order[i:i + size]]
        enc = tokenizer([r[0] for r in part], truncation=True, max_length=args.max_len, padding=True, return_tensors='pt')
        yield {k: v.to(device) for k, v in enc.items()}, torch.tensor([r[1] for r in part], device=device)


@torch.no_grad()
def scores(m, rows):
    m.eval()
    out = []
    for enc, _ in batches(rows, args.batch * 2):
        with torch.autocast(device.type, dtype=autocast, enabled=autocast is not None):
            logits = m(**enc).logits.float()
        out += torch.softmax(logits, -1)[:, 1].tolist()
    return np.array(out)


def metrics(rows, s):
    y = np.array([r[1] for r in rows])
    neg = np.sort(s[y == 0])
    res = {'n_pos': int(y.sum()), 'n_neg': int((y == 0).sum())}
    if y.min() == y.max():
        return res
    res['auroc'] = round(float(roc_auc_score(y, s)), 4)
    for fpr in (0.01, 0.05):
        thr = neg[min(len(neg) - 1, math.ceil((1 - fpr) * len(neg)) - 1)] if len(neg) else 1.0
        res[f'tpr_at_{int(fpr * 100)}pct_fpr'] = round(float((s[y == 1] > thr).mean()), 4)
    res['tpr_at_0.5'] = round(float((s[y == 1] >= 0.5).mean()), 4)
    res['fpr_at_0.5'] = round(float((s[y == 0] >= 0.5).mean()), 4)
    return res


base_report = {name: metrics(rows, scores(model, rows)) for name, rows in test_sets.items()}
print('base model on test documents:', json.dumps(base_report, indent=1))

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
steps = math.ceil(len(train_set) / args.batch / args.grad_accum) * args.epochs
warmup = max(1, int(0.1 * steps))
schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / warmup, max(0.0, (steps - s) / max(1, steps - warmup))))
scaler = torch.amp.GradScaler(device.type, enabled=precision == 'fp16')
out_dir = HERE / 'models' / args.name
best, history, t0 = -1.0, [], time.time()
for epoch in range(args.epochs):
    model.train()
    total, n = 0.0, 0
    for step, (enc, labels) in enumerate(batches(train_set, args.batch, shuffle=True)):
        with torch.autocast(device.type, dtype=autocast, enabled=autocast is not None):
            loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), labels) / args.grad_accum
        scaler.scale(loss).backward()
        if (step + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            schedule.step()
        total, n = total + loss.item() * args.grad_accum, n + 1
    val = metrics(val_set, scores(model, val_set))
    history.append({'epoch': epoch + 1, 'train_loss': round(total / max(1, n), 4), 'val': val})
    print(f"epoch {epoch + 1}: loss {total / max(1, n):.4f}  val {val}")
    if val.get('auroc', 0) > best:  # checkpoint selection uses validation only
        best = val.get('auroc', 0)
        if not args.dry_run:
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

if not args.dry_run:
    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
tuned_report = {name: metrics(rows, scores(model, rows)) for name, rows in test_sets.items()}
result = {
    'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'dry_run': args.dry_run,
    'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None, 'precision': precision,
    'base_model': args.base, 'output': str(out_dir) if not args.dry_run else None,
    'config': {k: v for k, v in vars(args).items() if k not in ('dry_run',)},
    'data': {'train_examples': len(train_set), 'val_examples': len(val_set), 'train_docs': len(train_ids), 'test_docs': len(test_docs),
             'bipia_train_items_fit': len(fit_attacks), 'bipia_train_items_val': len(val_attacks), 'bipia_test_items': len(test_attacks),
             'deepset': args.deepset, 'split_file': 'data/splits.json'},
    'history': history, 'train_minutes': round((time.time() - t0) / 60, 1),
    'test_base': base_report, 'test_finetuned': tuned_report,
}
name = 'finetune_dryrun.json' if args.dry_run else f'finetune_{args.name}.json'
(HERE / 'results' / name).write_text(json.dumps(result, indent=2))
if not args.dry_run:
    (out_dir / 'README.md').write_text(
        f"# {args.name}\n\nFine-tuned from `{args.base}` (Apache-2.0) for RAG Shield. Training data: BIPIA train-split attacks (MIT, "
        f"Microsoft) inserted unchanged into Vandelay class-corpus text from training documents only"
        f"{', plus deepset/prompt-injections train (Apache-2.0)' if args.deepset else ''}. Labels: 0 SAFE, 1 INJECTION.\n"
        f"Evaluated on held-out documents ({', '.join(sorted(test_docs))}) with BIPIA test-split attacks; see results/{name}.\n")
print(json.dumps({'test_base': base_report, 'test_finetuned': tuned_report}, indent=1))
