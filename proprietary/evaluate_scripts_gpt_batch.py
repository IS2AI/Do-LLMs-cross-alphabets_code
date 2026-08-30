
import argparse
import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LETTERS = [chr(ord("A") + i) for i in range(26)]

try:
    from datasets import load_dataset, get_dataset_config_names
except ImportError as e:
    print(f"pip install datasets\nError: {e}")
    exit(1)

try:
    from openai import OpenAI
except ImportError as e:
    print(f"pip install openai\nError: {e}")
    exit(1)


def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "evaluation.log")
    logger = logging.getLogger("script_eval_gpt")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setLevel(logging.WARNING)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger



def load_hf_dataset(
    dataset_name: str,
    split: str,
    portion: float = 1.0,
    config_name: str = None,
    public: bool = False,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    token = None if public else HF_TOKEN
    if not public and not HF_TOKEN:
        raise ValueError("Set HF_TOKEN env var for private datasets.")
    print(f"  Loading {dataset_name} (config={config_name}, split={split})...")
    ds = load_dataset(dataset_name, name=config_name, split=split, token=token)
    data = list(ds)
    if portion < 1.0:
        random.seed(seed)
        k = max(1, int(len(data) * portion))
        data = random.sample(data, k)
    print(f"  Loaded {len(data)} samples")
    return data


def load_local_jsonl(path: str, portion: float = 1.0, seed: int = 42) -> List[Dict]:
    print(f"  Loading {path}...")
    items = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if portion < 1.0:
        random.seed(seed)
        k = max(1, int(len(items) * portion))
        items = random.sample(items, k)
    print(f"  Loaded {len(items)} samples")
    return items


def load_dataset_auto(
    path: str,
    split: str,
    portion: float = 1.0,
    config_name: str = None,
    public: bool = False,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    if os.path.isfile(path):
        if path.endswith(".jsonl"):
            return load_local_jsonl(path, portion, seed)
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                if portion < 1.0:
                    random.seed(seed)
                    k = max(1, int(len(data) * portion))
                    data = random.sample(data, k)
                return data
            raise ValueError(f"JSON file must contain a list, got {type(data)}")
    if os.path.isdir(path):
        ds = load_dataset(path, split=split)
        data = list(ds)
        if portion < 1.0:
            random.seed(seed)
            k = max(1, int(len(data) * portion))
            data = random.sample(data, k)
        print(f"  Loaded {len(data)} samples from {path}")
        return data
    return load_hf_dataset(path, split, portion, config_name, public, seed)



def prompt_mcq(question: str, options: List[str], labels: List[str]) -> str:
    parts = [f"Question: {question}\n", "Options:"]
    for i, opt in enumerate(options):
        parts.append(f"{labels[i]}: {opt}")
    parts.append('\nProvide your answer as JSON: {"answer": "LETTER"}')
    return "\n".join(parts)


def prompt_math(question: str) -> str:
    return (
        f"Problem: {question}\n\n"
        'Solve step by step. Final answer as JSON: {"answer": NUMBER}'
    )



def _get_final(raw: str, think_end: str) -> str:
    parts = raw.split(think_end)
    return parts[-1].strip() if len(parts) > 1 else raw.strip()


def extract_mcq(raw: str, think_end: str, max_letter: str = "E") -> Optional[str]:
    final = _get_final(raw, think_end)
    pat = f"[A-{max_letter}]"

    for m in re.finditer(r"\{[^}]+\}", final):
        try:
            d = json.loads(m.group())
            ans = d.get("answer", "")
            found = re.search(f"({pat})", str(ans))
            if found:
                return found.group(1).upper()
        except (json.JSONDecodeError, AttributeError):
            pass

    matches = re.findall(rf"(?:answer|Answer|ANSWER)\s*[:\s]*({pat})", final)
    if matches:
        return matches[-1].upper()

    m = re.search(rf"\\boxed\{{({pat})\}}", final)
    if m:
        return m.group(1).upper()

    m = re.search(rf"\b({pat})\)?\s*$", final.strip())
    if m:
        return m.group(1).upper()

    m = re.search(
        rf"(?:therefore|thus|hence|so)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?({pat})\b",
        final, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    return None


def extract_numerical(raw: str, think_end: str) -> Optional[float]:
    final = _get_final(raw, think_end)

    for m in re.finditer(r"\{[^}]+\}", final):
        try:
            d = json.loads(m.group())
            ans = d.get("answer")
            if ans is not None:
                nums = re.findall(
                    r"(-?\d+\.?\d*)", str(ans).replace(",", "").replace("$", "")
                )
                if nums:
                    return float(nums[-1])
        except Exception:
            pass

    m = re.search(r"####\s*(-?\d+\.?\d*)", final.replace(",", ""))
    if m:
        return float(m.group(1))

    m = re.search(r"\\boxed\{(-?\d+\.?\d*)\}", final.replace(",", ""))
    if m:
        return float(m.group(1))

    nums = re.findall(r"(-?\d+\.?\d*)", final.replace(",", "").replace("$", ""))
    if nums:
        return float(nums[-1])
    return None


MODELS_NO_TEMPERATURE = {"gpt-5"}
MODELS_NO_JSON_FORMAT = {"gpt-5"}

def _build_body(
    model_name: str,
    temperature: float,
    max_tokens: int,
    messages: list,
    response_format: dict = None,
) -> dict:
    body = {
        "model": model_name,
        "max_completion_tokens": max_tokens,
        "messages": messages,
    }
    if model_name not in MODELS_NO_TEMPERATURE:
        body["temperature"] = temperature
    if response_format and model_name not in MODELS_NO_JSON_FORMAT:
        body["response_format"] = response_format
    return body


def build_kaz_culture_tasks(
    data: List[Dict],
    model_name: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> List[Dict]:
    labels = ["A", "B", "C", "D"]
    tasks = []
    for idx, item in enumerate(data):
        try:
            q    = item["question"]
            opts = [item["a"], item["b"], item["c"], item["d"]]
            gt   = str(item["answer_label"]).strip().upper()
        except KeyError:
            continue
        if gt not in labels:
            continue

        tasks.append({
            "custom_id": f"kaz_culture-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": _build_body(
                model_name, temperature, max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt_mcq(q, opts, labels)},
                ],
                response_format={"type": "json_object"},
            ),
            "_meta": {
                "question": q,
                "options": opts,
                "ground_truth": gt,
                "benchmark": "kaz_culture",
            },
        })
    return tasks


def build_kaz_mmlu_tasks(
    data: List[Dict],
    model_name: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    subset_name: str = "all",
) -> List[Dict]:
    labels = ["A", "B", "C", "D", "E"]
    tasks = []
    for idx, item in enumerate(data):
        try:
            q    = item["Question"]
            opts = [item["Option A"], item["Option B"],
                    item["Option C"], item["Option D"]]
            opt_e = item.get("Option E")
            if opt_e and str(opt_e).strip():
                opts.append(opt_e)
            gt = str(item["Answer Key"]).upper().strip()
        except KeyError:
            continue
        if gt not in labels[:len(opts)]:
            continue

        tasks.append({
            "custom_id": f"kaz_mmlu-{subset_name}-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": _build_body(
                model_name, temperature, max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt_mcq(q, opts, labels[:len(opts)])},
                ],
                response_format={"type": "json_object"},
            ),
            "_meta": {
                "question": q,
                "options": opts,
                "ground_truth": gt,
                "subset": subset_name,
                "benchmark": "kaz_mmlu",
            },
        })
    return tasks


def build_gsm8k_kaz_tasks(
    data: List[Dict],
    model_name: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> List[Dict]:
    tasks = []
    for idx, item in enumerate(data):
        try:
            q       = item["question"]
            raw_ans = str(item["answer_text"])
        except KeyError:
            continue

        m      = re.search(r"####\s*(.+)", raw_ans)
        gt_str = m.group(1).strip() if m else raw_ans.strip()
        gt_str = gt_str.replace(",", "").replace("$", "")
        try:
            gt = float(gt_str)
        except ValueError:
            continue

        tasks.append({
            "custom_id": f"gsm8k_kaz-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": _build_body(
                model_name, temperature, max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt_math(q)},
                ],
            ),
            "_meta": {
                "question": q,
                "ground_truth": gt,
                "benchmark": "gsm8k_kaz",
            },
        })
    return tasks


TASK_BUILDERS = {
    "kaz_culture": build_kaz_culture_tasks,
    "kaz_mmlu":    build_kaz_mmlu_tasks,
    "gsm8k_kaz":   build_gsm8k_kaz_tasks,
}


def load_benchmark_data(args) -> List[Dict[str, Any]]:
    if args.benchmark == "kaz_culture":
        return load_dataset_auto(
            args.dataset_path, "test", args.data_portion, public=True
        )

    if args.benchmark == "kaz_mmlu":
        try:
            subsets = get_dataset_config_names(args.dataset_path)
        except Exception:
            subsets = [None]
        all_data = []
        for sn in subsets:
            try:
                chunk = load_dataset_auto(
                    args.dataset_path, "test", args.data_portion,
                    config_name=sn, public=True,
                )
                for item in chunk:
                    item["_subset"] = str(sn) if sn else "all"
                all_data.extend(chunk)
            except Exception as e:
                print(f"  Skipping subset '{sn}': {e}")
        return all_data

    if args.benchmark == "gsm8k_kaz":
        config = "kazakh" if "issai" in args.dataset_path else None
        return load_dataset_auto(
            args.dataset_path, "test", args.data_portion, config_name=config
        )

    raise ValueError(f"Unknown benchmark: {args.benchmark}")



def build_batch_tasks(data: List[Dict], args) -> List[Dict]:
    kwargs = dict(
        model_name=args.model_name,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    if args.benchmark == "kaz_culture":
        return build_kaz_culture_tasks(data, **kwargs)

    if args.benchmark == "kaz_mmlu":
        tasks = []
        by_subset: Dict[str, List] = defaultdict(list)
        for item in data:
            by_subset[item.get("_subset", "all")].append(item)
        for sn, chunk in by_subset.items():
            tasks.extend(build_kaz_mmlu_tasks(chunk, subset_name=sn, **kwargs))
        return tasks

    if args.benchmark == "gsm8k_kaz":
        return build_gsm8k_kaz_tasks(data, **kwargs)

    raise ValueError(f"Unknown benchmark: {args.benchmark}")



def write_jsonl_batch_file(tasks: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            record = {k: v for k, v in t.items() if k != "_meta"}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(tasks)} tasks → {path}")


def write_meta_file(tasks: List[Dict], path: str):
    meta_map = {t["custom_id"]: t["_meta"] for t in tasks}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta_map, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Wrote metadata ({len(meta_map)} entries) → {path}")



def submit_batch(client: OpenAI, jsonl_path: str, logger: logging.Logger) -> str:
    print("  Uploading batch file...")
    with open(jsonl_path, "rb") as fh:
        batch_file = client.files.create(file=fh, purpose="batch")
    logger.info(f"Uploaded file id={batch_file.id}")
    print(f"  File id: {batch_file.id}")

    print("  Creating batch job...")
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    logger.info(f"Batch created id={batch_job.id} status={batch_job.status}")
    print(f"  Batch id: {batch_job.id}  (status={batch_job.status})")
    return batch_job.id


def poll_batch(
    client: OpenAI,
    batch_id: str,
    logger: logging.Logger,
    poll_interval: int = 60,
) -> Any:
    print(f"  Polling every {poll_interval}s ...")
    while True:
        batch = client.batches.retrieve(batch_id)
        counts    = batch.request_counts
        completed = counts.completed if counts else "?"
        total     = counts.total     if counts else "?"
        print(
            f"    [{time.strftime('%H:%M:%S')}] "
            f"status={batch.status}  {completed}/{total} done"
        )
        logger.info(f"Batch {batch_id}: status={batch.status} {completed}/{total}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            return batch
        time.sleep(poll_interval)


def retrieve_results(client: OpenAI, batch: Any) -> List[Dict]:
    if batch.status != "completed":
        raise RuntimeError(
            f"Batch {batch.id} ended with status={batch.status}. "
            "Check the OpenAI dashboard for error details."
        )
    print("  Downloading results...")
    raw_bytes = client.files.content(batch.output_file_id).content
    results = []
    for line in raw_bytes.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            results.append(json.loads(line))
    print(f"  Downloaded {len(results)} result rows")
    return results



def score_results(
    results: List[Dict],
    meta_map: Dict[str, Dict],
    args,
    logger: logging.Logger,
) -> List[Dict]:
    scored = []
    think_end = args.think_end_token

    for res in results:
        cid  = res.get("custom_id", "")
        meta = meta_map.get(cid)
        if not meta:
            logger.warning(f"No metadata for custom_id={cid}")
            continue

        try:
            raw = (
                res["response"]["body"]["choices"][0]["message"]["content"] or ""
            )
        except (KeyError, IndexError, TypeError):
            raw = ""
            logger.warning(f"No content for {cid}")

        benchmark = meta.get("benchmark", args.benchmark)

        if benchmark == "kaz_culture":
            ans = extract_mcq(raw, think_end, max_letter="D")
            gt  = meta["ground_truth"]
            row = {
                "custom_id":       cid,
                "question":        meta.get("question", ""),
                "options":         meta.get("options", []),
                "ground_truth":    gt,
                "model_output_raw": raw,
                "predicted_answer": ans,
                "correct":         ans is not None and ans == gt,
                "script":          args.script_label,
            }

        elif benchmark == "kaz_mmlu":
            ans = extract_mcq(raw, think_end, max_letter="E")
            gt  = meta["ground_truth"]
            row = {
                "custom_id":       cid,
                "question":        meta.get("question", ""),
                "options":         meta.get("options", []),
                "ground_truth":    gt,
                "subset":          meta.get("subset", "all"),
                "model_output_raw": raw,
                "predicted_answer": ans,
                "correct":         ans is not None and ans == gt,
                "script":          args.script_label,
            }

        elif benchmark == "gsm8k_kaz":
            ans = extract_numerical(raw, think_end)
            gt  = meta["ground_truth"]
            row = {
                "custom_id":       cid,
                "question":        meta.get("question", ""),
                "ground_truth":    gt,
                "model_output_raw": raw,
                "predicted_answer": ans,
                "correct":         ans is not None and abs(ans - gt) < 1e-4,
                "script":          args.script_label,
            }

        else:
            logger.warning(f"Unknown benchmark '{benchmark}' in meta for {cid}")
            continue

        scored.append(row)

    return scored



def save_results(
    results_dir: str,
    benchmark: str,
    script: str,
    results: List[Dict],
    metrics: Dict[str, float],
    logger: logging.Logger,
):
    os.makedirs(results_dir, exist_ok=True)
    safe_name = re.sub(r"[^\w\-_.]", "_", f"{benchmark}_{script}")

    jsonl_path = os.path.join(results_dir, f"{safe_name}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    summary_path = os.path.join(results_dir, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except json.JSONDecodeError:
            pass

    total   = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    valid   = [r for r in results if r.get("predicted_answer") is not None]

    entry = {
        "script":             script,
        "total_samples":      total,
        "correct":            correct,
        "accuracy":           round(correct / len(valid), 4) if valid else 0,
        "valid_samples":      len(valid),
        "failed_extractions": total - len(valid),
        "timestamp":          time.strftime("%Y-%m-%d %H:%M:%S"),
        **{k: round(v, 4) for k, v in metrics.items()},
    }
    summary[safe_name] = entry

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {jsonl_path} | accuracy={entry['accuracy']:.4f}")


def print_results(benchmark: str, script: str, results: List[Dict]):
    total   = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    acc     = correct / total * 100 if total else 0
    print(f"  [{script:15s}] {benchmark}: {acc:.2f}% ({correct}/{total})")


def compute_and_save(
    scored: List[Dict], args, results_dir: str, logger: logging.Logger
):
    total   = len(scored)
    correct = sum(r["correct"] for r in scored)
    acc     = correct / total if total else 0
    metrics = {"accuracy": acc}

    if args.benchmark == "kaz_mmlu":
        by_subset: Dict[str, List] = defaultdict(list)
        for r in scored:
            by_subset[r.get("subset", "all")].append(r)
        for sn, items in by_subset.items():
            safe_sn = re.sub(r"[^\w]", "_", str(sn))[:50]
            sub_acc = sum(i["correct"] for i in items) / len(items) if items else 0
            metrics[f"subset_{safe_sn}_acc"] = sub_acc

    save_results(results_dir, args.benchmark, args.script_label, scored, metrics, logger)
    print_results(args.benchmark, args.script_label, scored)



def save_batch_id(results_dir: str, benchmark: str, script: str, batch_id: str):
    path = os.path.join(results_dir, "batch_ids.json")
    ids  = {}
    if os.path.exists(path):
        with open(path) as f:
            ids = json.load(f)
    ids[f"{benchmark}_{script}"] = batch_id
    with open(path, "w") as f:
        json.dump(ids, f, indent=2)
    print(f"  batch_id persisted → {path}")


def load_batch_id(results_dir: str, benchmark: str, script: str) -> Optional[str]:
    path = os.path.join(results_dir, "batch_ids.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get(f"{benchmark}_{script}")



def main():
    parser = argparse.ArgumentParser(
        description="Kazakh benchmark evaluation via OpenAI Batch API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    
    parser.add_argument("--model_name", type=str, required=True,
                        help="OpenAI model name, e.g. gpt-4o or gpt-5")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key (falls back to OPENAI_API_KEY in .env)")

    
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=list(TASK_BUILDERS.keys()))
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="HF dataset ID, local directory, or JSONL file path")
    parser.add_argument("--script_label", type=str, required=True,
                        help="Script variant label (e.g. cyrillic, latin_2021)")

    
    parser.add_argument("--mode", type=str, default="submit_and_wait",
                        choices=["submit", "retrieve", "submit_and_wait"],
                        help=(
                            "submit          – upload batch, save ID, exit\n"
                            "retrieve        – score an already-finished batch\n"
                            "submit_and_wait – submit then poll until done (default)"
                        ))
    parser.add_argument("--batch_id", type=str, default=None,
                        help="Required for --mode retrieve unless auto-saved")
    parser.add_argument("--poll_interval", type=int, default=60,
                        help="Seconds between status-check calls (default 60)")

    
    parser.add_argument("--data_portion", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Per-request max_tokens (MCQ needs ~50; math ~512)")
    parser.add_argument("--system_prompt", type=str,
                        default="You are a helpful assistant.")
    parser.add_argument("--think_end_token", type=str, default="</think>")

    
    parser.add_argument("--output_dir", type=str, default="./results_scripts")

    args = parser.parse_args()

    
    api_key = args.api_key or OPENAI_API_KEY
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY in your .env file or pass --api_key")
        exit(1)
    client = OpenAI(api_key=api_key)

    
    model_safe  = re.sub(r"[^\w\-_.]", "_", args.model_name)
    results_dir = os.path.join(args.output_dir, model_safe)
    batch_dir   = os.path.join(results_dir, "batch_files")
    os.makedirs(batch_dir, exist_ok=True)
    logger = setup_logging(results_dir)

    safe_label = re.sub(r"[^\w\-_.]", "_", args.script_label)
    jsonl_path = os.path.join(batch_dir, f"{args.benchmark}_{safe_label}_batch.jsonl")
    meta_path  = os.path.join(batch_dir, f"{args.benchmark}_{safe_label}_meta.json")

    print(f"\n{'='*65}")
    print(f"GPT BATCH API EVALUATION")
    print(f"{'='*65}")
    print(f"Model:      {args.model_name}")
    print(f"Benchmark:  {args.benchmark}")
    print(f"Dataset:    {args.dataset_path}")
    print(f"Script:     {args.script_label}")
    print(f"Mode:       {args.mode}")
    print(f"Results:    {results_dir}")
    print(f"{'='*65}\n")

    if args.mode in ("submit", "submit_and_wait"):
        print(">> Loading dataset...")
        data = load_benchmark_data(args)
        if not data:
            print("ERROR: No data loaded.")
            exit(1)

        print(">> Building batch tasks...")
        tasks = build_batch_tasks(data, args)
        if not tasks:
            print("ERROR: No valid tasks built. Check dataset field names.")
            exit(1)
        print(f"  Built {len(tasks)} tasks")

        write_jsonl_batch_file(tasks, jsonl_path)
        write_meta_file(tasks, meta_path)

        print("\n>> Submitting to OpenAI Batch API...")
        batch_id = submit_batch(client, jsonl_path, logger)
        save_batch_id(results_dir, args.benchmark, args.script_label, batch_id)

        if args.mode == "submit":
            print(f"\n✓ Batch submitted.")
            print(f"  Batch ID : {batch_id}")
            print(
                f"  Retrieve : python {__file__} "
                f"--model_name {args.model_name} "
                f"--benchmark {args.benchmark} "
                f"--dataset_path {args.dataset_path} "
                f"--script_label {args.script_label} "
                f"--mode retrieve --batch_id {batch_id}"
            )
            return

        args.batch_id = batch_id  

    batch_id = args.batch_id or load_batch_id(
        results_dir, args.benchmark, args.script_label
    )
    if not batch_id:
        print("ERROR: --batch_id required for --mode retrieve "
              "(or run --mode submit first to auto-save it)")
        exit(1)

    if not os.path.exists(meta_path):
        print(f"ERROR: metadata file not found: {meta_path}")
        print("  Run --mode submit from the same machine and --output_dir first.")
        exit(1)

    with open(meta_path) as f:
        meta_map = json.load(f)

    print(f"\n>> Polling batch {batch_id}...")
    batch = poll_batch(client, batch_id, logger, args.poll_interval)

    print("\n>> Retrieving results...")
    raw_results = retrieve_results(client, batch)

    print(">> Scoring results...")
    scored = score_results(raw_results, meta_map, args, logger)
    print(f"  Scored {len(scored)} / {len(raw_results)} items")

    compute_and_save(scored, args, results_dir, logger)

    summary_path = os.path.join(results_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"\n{'='*65}")
        print("SUMMARY (all runs in this output dir)")
        print(f"{'='*65}")
        for key, val in sorted(summary.items()):
            print(
                f"  {key:50s}  acc={val.get('accuracy', 'N/A')}  "
                f"script={val.get('script', '?')}"
            )

    print(f"\nResults saved to: {results_dir}/")


if __name__ == "__main__":
    main()