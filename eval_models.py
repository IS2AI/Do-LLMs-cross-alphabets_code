"""Evaluate served open-weight models on KazMMLU, KazCulture and GSM8k-Kazakh.

One run covers a model, an inference mode (thinking / non-thinking), a script
condition (cyrillic, 2017, 2018, 2019, 2021) and a prompt condition:

    baseline  the evaluation prompt on its own
    hint      one sentence stating that the Kazakh text is in the Latin alphabet
    mapping   the complete Cyrillic-to-Latin mapping of the active Latin variant

`hint` and `mapping` apply to the Latin conditions only; the Cyrillic condition is
always evaluated with the baseline prompt. The instructions are in English in every
condition and only the question content changes script.

The model is asked for a JSON object with an `answer` key and scored by exact
match: an answer letter for KazMMLU and KazCulture, a number for GSM8k-Kazakh.

Outputs one JSONL row per item to results/raw/ next to a .meta.json recording the
full configuration, and upserts a row into results/summary.csv.

Run:
  python eval_models.py --models qwen3.5-4b --datasets kazmmlu --scripts cyrillic,2021
  python eval_models.py --prompt hint --scripts 2017,2018,2019,2021
  python eval_models.py --mode think
"""
import argparse
import csv
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from transliterate import SCRIPT_CONDITIONS, mapping_table_text

try:
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
    _LANGDETECT_OK = True
except Exception:
    _LANGDETECT_OK = False

HERE = Path(__file__).parent
DATA_ROOT = HERE / "data"
OUT_ROOT = HERE / "results"
RAW_DIR = OUT_ROOT / "raw"
MODELS_PATH = HERE / "models.json"

DATASETS = ["kazmmlu", "kazculture", "gsm8k"]
PROMPT_CONDITIONS = ["baseline", "hint", "mapping"]
MODES = ["no-think", "think"]

SEED = 42
TEMPERATURE = 0.0


# ---------- configuration ----------

def load_models():
    registry = json.loads(MODELS_PATH.read_text())
    return {m["name"]: m for m in registry["models"]}


def mode_kwargs(model_cfg, mode):
    if mode == "think":
        if not model_cfg.get("dual_mode") or model_cfg.get("think") is None:
            raise SystemExit(f"{model_cfg['name']}: no thinking mode configured in models.json")
        return model_cfg["think"]
    return model_cfg.get("no_think") or {}


# ---------- prompts ----------

HINT_SENTENCE = "The following Kazakh text uses the Latin alphabet."

MCQ_INSTRUCTIONS = (
    "The following is a multiple-choice question. Choose the correct option.\n"
    "Reply with a single JSON object and nothing else, in exactly this format:\n"
    '{"answer": "<letter>"}\n\n'
)
NUMERIC_INSTRUCTIONS = (
    "The following is a grade-school math question. Solve it.\n"
    "Reply with a single JSON object and nothing else, in exactly this format:\n"
    '{"answer": <number>}\n\n'
)


def build_preamble(script, prompt_condition):
    if script == "cyrillic" or prompt_condition == "baseline":
        return ""
    if prompt_condition == "hint":
        return HINT_SENTENCE + "\n\n"
    return ("Cyrillic-to-Latin letter mapping for the text below:\n"
            + mapping_table_text(script) + "\n\n")


def format_row(dataset, row, script, prompt_condition):
    """-> (kind, prompt, gold, n_options)"""
    preamble = build_preamble(script, prompt_condition)
    if dataset == "kazmmlu":
        options = [(letter, row[f"Option {letter}"]) for letter in "ABCDE"
                   if row.get(f"Option {letter}")]
        body = "\n".join(f"{letter}. {text}" for letter, text in options)
        prompt = (preamble + MCQ_INSTRUCTIONS + "Question:\n" + row["Question"]
                  + "\n" + body)
        return "mcq", prompt, str(row["Answer Key"]).strip().upper(), len(options)
    if dataset == "kazculture":
        if not row.get("answer_label"):
            raise KeyError("kazculture row has no answer_label; "
                           f"available fields: {sorted(row)}")
        options = [(letter, row[letter]) for letter in "abcd" if row.get(letter)]
        body = "\n".join(f"{letter.upper()}. {text}" for letter, text in options)
        prompt = (preamble + MCQ_INSTRUCTIONS + "Question:\n" + row["question"]
                  + "\n" + body)
        return "mcq", prompt, str(row["answer_label"]).strip().upper(), len(options)
    if dataset == "gsm8k":
        prompt = preamble + NUMERIC_INSTRUCTIONS + "Question:\n" + row["question"]
        return "numeric", prompt, gsm8k_gold(row["answer_text"]), 0
    raise ValueError(dataset)


# ---------- answer extraction and scoring ----------

_THINK_END = "</think>"
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.S)
_LETTER_RE = re.compile(r"^([A-E])(?:\s*[.):\-–—]\s*.*)?$", re.S)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def final_segment(text):
    return text.split(_THINK_END)[-1] if _THINK_END in text else text


def extract_answer(text):
    """Return (value, status). status: ok | no_json | missing_key | empty."""
    if not text or not text.strip():
        return None, "empty"
    segment = final_segment(text).replace("```json", " ").replace("```", " ")
    saw_object = False
    for candidate in reversed(_JSON_OBJECT_RE.findall(segment)):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                obj = json.loads(candidate.replace("'", '"'))
            except json.JSONDecodeError:
                continue
        if isinstance(obj, dict):
            saw_object = True
            for key in obj:
                if key.strip().lower() == "answer":
                    return obj[key], "ok"
    return None, "missing_key" if saw_object else "no_json"


def normalize_letter(value, n_options):
    allowed = "ABCDE"[:n_options] if n_options else "ABCDE"
    text = str(value).strip().strip("\"'").strip().upper()
    match = _LETTER_RE.match(text)
    if not match:
        return None
    letter = match.group(1)
    return letter if letter in allowed else None


def normalize_number(value):
    text = str(value).strip()
    for junk in (",", "$", "₸", " ", " ", "%"):
        text = text.replace(junk, "")
    text = text.replace("−", "-").rstrip(".")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return None
    return float(text)


def gsm8k_gold(answer_text):
    match = re.search(r"####\s*([^\n]+)", str(answer_text))
    if not match:
        raise ValueError(f"no #### answer in: {answer_text!r}")
    gold = normalize_number(match.group(1).strip())
    if gold is None:
        raise ValueError(f"unparseable gold answer: {match.group(1)!r}")
    return gold


def score(kind, value, gold, n_options):
    """Return (score, status). A non-`ok` status is scored 0."""
    if kind == "mcq":
        letter = normalize_letter(value, n_options)
        if letter is None:
            return 0, "invalid_answer"
        return int(letter == gold), "ok"
    number = normalize_number(value)
    if number is None:
        return 0, "invalid_answer"
    return int(abs(number - gold) < 1e-6), "ok"


def detect_lang(text):
    if not _LANGDETECT_OK or not text or len(text.strip()) < 3:
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"


# ---------- data ----------

def load_items(dataset, script, prompt_condition):
    root = DATA_ROOT / dataset / script
    if not root.exists():
        return []
    items = []
    for path in sorted(root.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for index, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                kind, prompt, gold, n_options = format_row(
                    dataset, row, script, prompt_condition)
                items.append((f"{path.stem}#{index}", kind, prompt, gold, n_options))
    return items


# ---------- inference ----------

def call_model(model_cfg, prompt, max_tokens, extra_kwargs, timeout=600):
    payload = {
        "model": model_cfg["served_name"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "seed": SEED,
    }
    payload.update(extra_kwargs)
    response = requests.post(f"{model_cfg['endpoint']}/v1/chat/completions",
                             json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    usage = body.get("usage", {})
    return (body["choices"][0]["message"]["content"],
            usage.get("prompt_tokens"), usage.get("completion_tokens"))


# ---------- runner ----------

def run_cell(model_cfg, mode, dataset, script, prompt_condition, args):
    if script == "cyrillic" and prompt_condition != "baseline":
        print(f"  [skip] {dataset}/{script}: {prompt_condition} applies to Latin only")
        return None

    items = load_items(dataset, script, prompt_condition)
    if args.limit:
        items = items[:args.limit]
    if not items:
        print(f"  [skip] {dataset}/{script}: no data (run convert_datasets.py first)")
        return None

    name = model_cfg["name"]
    stem = f"{name}__{mode}__{dataset}__{script}__{prompt_condition}"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"{stem}.jsonl"
    extra_kwargs = mode_kwargs(model_cfg, mode)

    config = {
        "model": name, "model_hf_id": model_cfg.get("hf_id", ""),
        "model_family": model_cfg.get("family", ""),
        "served_name": model_cfg["served_name"], "endpoint": model_cfg["endpoint"],
        "mode": mode, "request_extra": extra_kwargs,
        "dataset": dataset, "script": script, "prompt_condition": prompt_condition,
        "temperature": TEMPERATURE, "seed": SEED, "max_tokens": args.max_tokens,
        "n_items": len(items),
    }

    if args.overwrite and raw_path.exists():
        raw_path.unlink()
    done = set()
    records = []
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("status") == "request_error":
                    continue
                done.add(record["item_id"])
                records.append(record)
    remaining = [item for item in items if item[0] not in done]
    print(f"  -> {stem}: {len(remaining)}/{len(items)} items")

    def work(item):
        item_id, kind, prompt, gold, n_options = item
        try:
            text, in_tokens, out_tokens = call_model(
                model_cfg, prompt, args.max_tokens, extra_kwargs)
        except Exception as error:
            return {"item_id": item_id, "kind": kind, "status": "request_error",
                    "error": str(error), "score": None}
        value, status = extract_answer(text)
        if status == "ok":
            item_score, status = score(kind, value, gold, n_options)
        else:
            item_score = 0
        return {
            "item_id": item_id, "kind": kind, "gold": gold,
            "answer_value": value, "status": status, "score": item_score,
            "prediction": text, "prompt_tokens": in_tokens,
            "completion_tokens": out_tokens, "lang": detect_lang(text),
        }

    started = time.time()
    if remaining:
        with raw_path.open("a", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(work, item) for item in remaining]
                for future in as_completed(futures):
                    record = future.result()
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    records.append(record)
    elapsed = time.time() - started

    scored = [r for r in records if r["status"] in ("ok", "invalid_answer",
                                                    "no_json", "missing_key", "empty")]
    correct = sum(r["score"] for r in scored)
    invalid = sum(1 for r in scored if r["status"] != "ok")
    errors = sum(1 for r in records if r["status"] == "request_error")
    prompt_tokens = [r["prompt_tokens"] for r in scored if r.get("prompt_tokens")]
    completion_tokens = [r["completion_tokens"] for r in scored if r.get("completion_tokens")]

    summary = dict(config)
    summary.pop("request_extra")
    summary.update({
        "request_extra": json.dumps(extra_kwargs, ensure_ascii=False),
        "n_scored": len(scored), "n_correct": correct,
        "accuracy": round(correct / len(scored), 6) if scored else None,
        "n_invalid_answer": invalid, "n_request_error": errors,
        "avg_prompt_tokens": round(sum(prompt_tokens) / len(prompt_tokens), 1) if prompt_tokens else 0,
        "avg_completion_tokens": round(sum(completion_tokens) / len(completion_tokens), 1) if completion_tokens else 0,
        "elapsed_s": round(elapsed, 1),
    })
    (RAW_DIR / f"{stem}.meta.json").write_text(
        json.dumps({**config, "request_extra": extra_kwargs, "summary": summary},
                   indent=2, ensure_ascii=False))
    print(f"     accuracy={summary['accuracy']} invalid={invalid} errors={errors}")
    return summary


def upsert_summary(rows):
    if not rows:
        return
    path = OUT_ROOT / "summary.csv"
    key_fields = ("model", "mode", "dataset", "script", "prompt_condition")
    existing = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))
    keys = {tuple(row[k] for k in key_fields) for row in rows}
    merged = [row for row in existing
              if tuple(row.get(k, "") for k in key_fields) not in keys]
    merged.extend({k: ("" if v is None else v) for k, v in row.items()} for row in rows)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)
    print(f"wrote {path}")


def main():
    registry = load_models()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(registry))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--scripts", default=",".join(SCRIPT_CONDITIONS))
    ap.add_argument("--prompt", default="baseline", choices=PROMPT_CONDITIONS)
    ap.add_argument("--mode", default="no-think", choices=MODES)
    ap.add_argument("--limit", type=int, default=0, help="0 = all items")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    random.seed(SEED)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in registry]
    if unknown:
        raise SystemExit(f"unknown models: {unknown}; known: {list(registry)}")
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    scripts = [s.strip() for s in args.scripts.split(",") if s.strip()]
    unknown = [s for s in scripts if s not in SCRIPT_CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown script conditions: {unknown}; "
                         f"expected {SCRIPT_CONDITIONS}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in models:
        model_cfg = registry[name]
        try:
            requests.get(f"{model_cfg['endpoint']}/v1/models", timeout=5)
        except Exception as error:
            print(f"[warn] {name} unreachable at {model_cfg['endpoint']}: {error}",
                  file=sys.stderr)
            continue
        print(f"== {name} ({args.mode}) @ {model_cfg['endpoint']}")
        for dataset in datasets:
            for script in scripts:
                summary = run_cell(model_cfg, args.mode, dataset, script,
                                   args.prompt, args)
                if summary:
                    rows.append(summary)
    upsert_summary(rows)
    if not rows:
        print("no results produced.", file=sys.stderr)


if __name__ == "__main__":
    main()
