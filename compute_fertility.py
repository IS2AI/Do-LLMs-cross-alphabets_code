"""Tokenizer fertility on the benchmark text of every script condition.

fertility = tokenizer tokens / whitespace-separated words

Computed on the benchmark input text (question, plus options for the two
multiple-choice sets) without a chat template, without the English instructions
and without special tokens, separately for every tokenizer, benchmark and script
condition. Model outputs are never tokenized.

Tokenizers come from models.json. Models sharing a tokenizer are measured once
and the sharing is recorded in the `shared_by` column.

Outputs results/fertility.csv.

Run:  python compute_fertility.py [--models ...] [--datasets ...] [--scripts ...]
"""
import argparse
import csv
import json
from pathlib import Path

from transformers import AutoTokenizer

from transliterate import SCRIPT_CONDITIONS

HERE = Path(__file__).parent
DATA_ROOT = HERE / "data"
OUT_PATH = HERE / "results" / "fertility.csv"
MODELS_PATH = HERE / "models.json"

DATASETS = ["kazmmlu", "kazculture", "gsm8k"]


def load_models():
    registry = json.loads(MODELS_PATH.read_text())
    return {m["name"]: m for m in registry["models"]}


def content_of(dataset, row):
    if dataset == "kazmmlu":
        parts = [row.get("Question", "")]
        parts += [row[f"Option {l}"] for l in "ABCDE" if row.get(f"Option {l}")]
    elif dataset == "kazculture":
        parts = [row.get("question", "")]
        parts += [row[l] for l in "abcd" if row.get(l)]
    elif dataset == "gsm8k":
        parts = [row.get("question", "")]
    else:
        raise ValueError(dataset)
    return "\n".join(p for p in parts if p)


def iter_contents(dataset, script):
    root = DATA_ROOT / dataset / script
    if not root.exists():
        return
    for path in sorted(root.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield content_of(dataset, json.loads(line))


def measure(tokenizer, dataset, script):
    contents = [c for c in iter_contents(dataset, script) if c]
    if not contents:
        return None
    words = sum(len(c.split()) for c in contents)
    tokens = sum(len(tokenizer.encode(c, add_special_tokens=False)) for c in contents)
    chars = sum(len(c) for c in contents)
    return {"n_items": len(contents), "total_words": words,
            "total_tokens": tokens, "total_chars": chars}


def main():
    registry = load_models()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(registry))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--scripts", default=",".join(SCRIPT_CONDITIONS))
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in registry]
    if unknown:
        raise SystemExit(f"unknown models: {unknown}; known: {list(registry)}")
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    scripts = [s.strip() for s in args.scripts.split(",") if s.strip()]

    # group models by tokenizer so a shared tokenizer is measured once
    groups = {}
    for name in models:
        tokenizer_id = registry[name].get("tokenizer_id", "")
        if not tokenizer_id:
            print(f"[skip] {name}: no tokenizer_id in models.json")
            continue
        groups.setdefault(tokenizer_id, []).append(name)

    rows = []
    for tokenizer_id, shared_by in groups.items():
        print(f"== {tokenizer_id}  ({', '.join(shared_by)})")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        for dataset in datasets:
            for script in scripts:
                counts = measure(tokenizer, dataset, script)
                if counts is None:
                    print(f"  [skip] {dataset}/{script}: no data")
                    continue
                fertility = counts["total_tokens"] / counts["total_words"]
                rows.append({
                    "tokenizer_id": tokenizer_id,
                    "shared_by": "|".join(shared_by),
                    "model_families": "|".join(
                        sorted({registry[m].get("family", "") for m in shared_by})),
                    "dataset": dataset, "script": script, **counts,
                    "tokens_per_word": round(fertility, 4),
                    "chars_per_token": round(counts["total_chars"] / counts["total_tokens"], 4),
                })
                print(f"  {dataset}/{script}: fertility={fertility:.3f}")

    if not rows:
        print("no rows produced.")
        return
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
