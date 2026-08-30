import argparse
import asyncio
import base64
import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
GOOGLE_API_KEY =os.getenv("API_KEY")
LETTERS = [chr(ord("A") + i) for i in range(26)]

try:
    from datasets import load_dataset, get_dataset_config_names
    from tqdm.asyncio import tqdm as tqdm_async
except ImportError as e:
    print(
        f"Install required libraries:\n"
        f"  pip install datasets tqdm python-dotenv\n"
        f"Error: {e}"
    )
    exit(1)

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "evaluation.log")
    logger = logging.getLogger("script_eval")
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



class GeminiClient:
     
    def __init__(self, api_key: str, model_name: str):
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Install google-generativeai:\n"
                "  pip install google-generativeai"
            )
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
    
    async def call_text(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        retries: int = 5,
    ) -> str:
        delay = 1.0
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=f"{system_prompt}\n\n{user_prompt}",
                )
                return response.text.strip() if response.text else ""
            except Exception as e:
                if attempt < retries - 1:
                    logging.warning(f"Gemini API error (attempt {attempt+1}): {e}")
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    logging.error(f"Gemini API failed after {retries} attempts: {e}")
                    return ""
        return ""
    
    async def call_vision(
        self,
        user_prompt: str,
        image_data: Any,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        retries: int = 5,
    ) -> str:
        delay = 1.0
        for attempt in range(retries):
            try:
                
                from PIL import Image
                
                if isinstance(image_data, str):
                    
                    if image_data.startswith("data:"):
                        
                        b64_data = image_data.split(",")[1]
                        image_bytes = base64.b64decode(b64_data)
                        img = Image.open(BytesIO(image_bytes))
                    else:
                        
                        img = Image.open(image_data)
                elif isinstance(image_data, bytes):
                    img = Image.open(BytesIO(image_data))
                elif isinstance(image_data, dict) and "bytes" in image_data:
                    img = Image.open(BytesIO(image_data["bytes"]))
                else:
                    img = image_data  
                
                response = self.model.generate_content(
                    [
                        f"{system_prompt}\n\n{user_prompt}",
                        img,
                    ],
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        top_p=top_p,
                    ),
                )
                return response.text.strip() if response.text else ""
            except Exception as e:
                if attempt < retries - 1:
                    logging.warning(f"Gemini Vision API error (attempt {attempt+1}): {e}")
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    logging.error(f"Gemini Vision API failed after {retries}: {e}")
                    return ""
        return ""



async def call_model_text_openai(
    client: "AsyncOpenAI",
    user_prompt: str,
    system_prompt: str,
    args,
    retries: int = 5,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.strip()},
    ]

    extra_body = {}
    if args.enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": args.enable_thinking
        }

    delay = 1.0
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                extra_body=extra_body if extra_body else None,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                logging.warning(f"API error (attempt {attempt+1}): {e}")
                await asyncio.sleep(delay)
                delay *= 2
            else:
                logging.error(f"API failed after {retries} attempts: {e}")
                return ""
    return ""


async def call_model_vision_openai(
    client: "AsyncOpenAI",
    user_prompt: str,
    image_data: Any,
    system_prompt: str,
    args,
    retries: int = 5,
) -> str:
    b64_str = _image_to_base64(image_data)
    if not b64_str:
        logging.error("Failed to encode image")
        return ""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_str}"},
                },
                {"type": "text", "text": user_prompt.strip()},
            ],
        },
    ]

    extra_body = {}
    if args.enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": args.enable_thinking
        }

    delay = 1.0
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                extra_body=extra_body if extra_body else None,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                logging.warning(f"Vision API error (attempt {attempt+1}): {e}")
                await asyncio.sleep(delay)
                delay *= 2
            else:
                logging.error(f"Vision API failed after {retries}: {e}")
                return ""
    return ""


def _image_to_base64(image_data: Any) -> Optional[str]:
    try:
        from PIL import Image
    except ImportError:
        print("pip install Pillow")
        return None
    try:
        if isinstance(image_data, str):
            return image_data
        if isinstance(image_data, bytes):
            return base64.b64encode(image_data).decode("utf-8")
        if isinstance(image_data, Image.Image):
            buf = BytesIO()
            image_data.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        if isinstance(image_data, dict) and "bytes" in image_data:
            return base64.b64encode(image_data["bytes"]).decode("utf-8")
        if hasattr(image_data, "convert"):  # PIL-like
            buf = BytesIO()
            image_data.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        logging.error(f"Unknown image type: {type(image_data)}")
        return None
    except Exception as e:
        logging.error(f"Image encoding error: {e}")
        return None



async def call_model_text(
    client: Union[GeminiClient, "AsyncOpenAI"],
    user_prompt: str,
    system_prompt: str,
    args,
    provider: str,
) -> str:
    if provider == "gemini":
        return await client.call_text(
            user_prompt, system_prompt, args.temperature, args.max_tokens, args.top_p
        )
    else:  
        return await call_model_text_openai(client, user_prompt, system_prompt, args)


async def call_model_vision(
    client: Union[GeminiClient, "AsyncOpenAI"],
    user_prompt: str,
    image_data: Any,
    system_prompt: str,
    args,
    provider: str,
) -> str:
    if provider == "gemini":
        return await client.call_vision(
            user_prompt, image_data, system_prompt, args.temperature, args.max_tokens, args.top_p
        )
    else:  
        return await call_model_vision_openai(client, user_prompt, image_data, system_prompt, args)


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

    m = re.search(rf"(?:therefore|thus|hence|so)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?({pat})\b", final, re.IGNORECASE)
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


def prompt_vision_mcq(question: str) -> str:
    return (
        f"{question}\n\n"
        'Provide your answer as JSON: {"answer": "LETTER"}'
    )



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
            with open(summary_path, "r") as f:
                summary = json.load(f)
        except json.JSONDecodeError:
            pass

    total = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    valid = [r for r in results if r.get("predicted_answer") is not None]

    entry = {
        "script": script,
        "total_samples": total,
        "correct": correct,
        "accuracy": round(correct / len(valid), 4) if valid else 0,
        "valid_samples": len(valid),
        "failed_extractions": total - len(valid),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **{k: round(v, 4) for k, v in metrics.items()},
    }
    summary[safe_name] = entry

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {jsonl_path} | accuracy={entry['accuracy']:.4f}")


def print_results(benchmark: str, script: str, results: List[Dict]):
    total = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    acc = correct / total * 100 if total else 0
    print(f"  [{script:15s}] {benchmark}: {acc:.2f}% ({correct}/{total})")


async def evaluate_kaz_culture(client, args, logger, results_dir, script, provider):
    benchmark = "kaz_culture"
    print(f"\n>> {benchmark.upper()} [{script}]")

    data = load_dataset_auto(args.dataset_path, "test", args.data_portion, public=True)
    labels = ["A", "B", "C", "D"]
    sem = asyncio.Semaphore(args.batch_size)

    async def process(item):
        async with sem:
            try:
                q = item["question"]
                opts = [item["a"], item["b"], item["c"], item["d"]]
                gt = str(item["answer_label"]).strip().upper()
            except KeyError as e:
                logger.error(f"{benchmark}: missing key {e}")
                return None
            if gt not in labels:
                return None

            raw = await call_model_text(
                client, prompt_mcq(q, opts, labels), args.system_prompt, args, provider
            )
            ans = extract_mcq(raw, args.think_end_token, max_letter="D")

            return {
                "question": q,
                "options": opts,
                "ground_truth": gt,
                "model_output_raw": raw,
                "predicted_answer": ans,
                "correct": ans is not None and ans == gt,
                "script": script,
            }

    results = [
        r for r in await tqdm_async.gather(
            *[process(it) for it in data], desc=f"{benchmark}_{script}"
        ) if r
    ]

    acc = sum(r["correct"] for r in results) / len(results) if results else 0
    save_results(results_dir, benchmark, script, results, {"accuracy": acc}, logger)
    print_results(benchmark, script, results)
    return results


async def evaluate_kaz_mmlu(client, args, logger, results_dir, script, provider):
    benchmark = "kaz_mmlu"
    dataset_path = args.dataset_path
    print(f"\n>> {benchmark.upper()} [{script}] (all subsets)")

    try:
        subsets = get_dataset_config_names(dataset_path)
        print(f"  Found {len(subsets)} subsets")
    except Exception:
        print(f"  No configs found, loading as single dataset")
        subsets = [None]

    labels = ["A", "B", "C", "D", "E"]
    sem = asyncio.Semaphore(args.batch_size)
    all_results = []

    async def process(item, subset_name):
        async with sem:
            try:
                q = item["Question"]
                opts = [item["Option A"], item["Option B"],
                        item["Option C"], item["Option D"]]
                opt_e = item.get("Option E")
                if opt_e and str(opt_e).strip():
                    opts.append(opt_e)
                gt = str(item["Answer Key"]).upper().strip()
            except KeyError as e:
                logger.error(f"KazMMLU {subset_name}: {e}")
                return None
            if gt not in labels[:len(opts)]:
                return None

            raw = await call_model_text(
                client, prompt_mcq(q, opts, labels[:len(opts)]), args.system_prompt, args, provider
            )
            ans = extract_mcq(raw, args.think_end_token, max_letter="E")

            return {
                "question": q,
                "options": opts,
                "ground_truth": gt,
                "subset": subset_name or "all",
                "model_output_raw": raw,
                "predicted_answer": ans,
                "correct": ans is not None and ans == gt,
                "script": script,
            }

    for sn in subsets:
        try:
            data = load_dataset_auto(
                dataset_path, "test", args.data_portion,
                config_name=sn, public=True,
            )
        except Exception as e:
            logger.warning(f"Skipping KazMMLU subset '{sn}': {e}")
            continue

        chunk_size = 50
        sub_res = []

        tasks = [process(it, sn) for it in data]

        outer_pbar = tqdm(
            total=len(tasks),
            desc=f"TOTAL KazMMLU:{str(sn)[:40]}_{script}",
        )

        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i + chunk_size]

            results = await tqdm_async.gather(
                *chunk,
                desc=f"Chunk {i//chunk_size + 1}",
            )

            sub_res.extend(results)

            outer_pbar.update(len(chunk))

            if i + chunk_size < len(tasks):
                await asyncio.sleep(2)

        outer_pbar.close()

        all_results.extend([r for r in sub_res if r])

    if not all_results:
        return []

    acc = sum(r["correct"] for r in all_results) / len(all_results)

    subset_metrics = {}
    by_subset = defaultdict(list)
    for r in all_results:
        by_subset[r["subset"]].append(r)
    for sn, items in by_subset.items():
        safe_sn = re.sub(r"[^\w]", "_", str(sn))[:50]
        sub_acc = sum(i["correct"] for i in items) / len(items) if items else 0
        subset_metrics[f"subset_{safe_sn}_acc"] = sub_acc

    save_results(
        results_dir, benchmark, script, all_results,
        {"accuracy": acc, **subset_metrics}, logger
    )
    print_results(benchmark, script, all_results)
    return all_results


async def evaluate_gsm8k_kaz(client, args, logger, results_dir, script, provider):
    benchmark = "gsm8k_kaz"
    print(f"\n>> {benchmark.upper()} [{script}]")

    config = "kazakh" if "issai" in args.dataset_path else None
    data = load_dataset_auto(
        args.dataset_path, "test", args.data_portion, config_name=config,
    )
    sem = asyncio.Semaphore(args.batch_size)

    async def process(item):
        async with sem:
            try:
                q = item["question"]
                raw_ans = str(item["answer_text"])
            except KeyError as e:
                logger.error(f"{benchmark}: {e}")
                return None

            m = re.search(r"####\s*(.+)", raw_ans)
            gt_str = m.group(1).strip() if m else raw_ans.strip()
            gt_str = gt_str.replace(",", "").replace("$", "")
            try:
                gt = float(gt_str)
            except ValueError:
                return None

            raw = await call_model_text(
                client, prompt_math(q), args.system_prompt, args, provider
            )
            ans = extract_numerical(raw, args.think_end_token)

            return {
                "question": q,
                "ground_truth": gt,
                "model_output_raw": raw,
                "predicted_answer": ans,
                "correct": ans is not None and abs(ans - gt) < 1e-4,
                "script": script,
            }

    results = [
        r for r in await tqdm_async.gather(
            *[process(it) for it in data], desc=f"{benchmark}_{script}"
        ) if r
    ]

    acc = sum(r["correct"] for r in results) / len(results) if results else 0
    save_results(results_dir, benchmark, script, results, {"accuracy": acc}, logger)
    print_results(benchmark, script, results)
    return results



BENCHMARKS = {
    "kaz_culture":  evaluate_kaz_culture,
    "kaz_mmlu":     evaluate_kaz_mmlu,
    "gsm8k_kaz":    evaluate_gsm8k_kaz,
}

FERTILITY_TEXT_FIELDS = {
    "kaz_culture":  ["question", "a", "b", "c", "d"],
    "kaz_mmlu":     ["Question", "Option A", "Option B", "Option C", "Option D"],
    "gsm8k_kaz":    ["question"],
}



async def async_main():
    parser = argparse.ArgumentParser(
        description="Script-change evaluation for Kazakh LLMs (pre-converted datasets)\n"
                    "Supports both vLLM (OpenAI-compatible) and Google Generative AI (Gemini)"
    )
    
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "gemini"],
                        help="API provider: 'openai' for vLLM, 'gemini' for Google Generative AI")
    
    parser.add_argument("--api_base", type=str, default=None,
                        help="[OpenAI] vLLM OpenAI-compatible API base URL")
    parser.add_argument("--model_path", type=str, default=None,
                        help="[OpenAI] Local model path (for tokenizer fertility)")
    
    parser.add_argument("--api_key", type=str, default=None,
                        help="[Gemini] Google API key (or use GOOGLE_API_KEY env var)")

    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name (e.g. 'gemini-2.0-flash-lite' or 'Qwen/Qwen2.5-7B-Instruct')")

    parser.add_argument("--benchmark", type=str, required=True,
                        choices=list(BENCHMARKS.keys()),
                        help="Which benchmark to run")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the (pre-converted) dataset. "
                             "Can be HF ID, local dir, or JSONL file.")
    parser.add_argument("--script_label", type=str, required=True,
                        help="Label for this script variant (e.g. cyrillic, latin_2021)")

    parser.add_argument("--batch_size", type=int, default=8,
                        help="Concurrent requests (note: Gemini has lower rate limits)")
    parser.add_argument("--data_portion", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--system_prompt", type=str,
                        default="You are a helpful assistant.")
    parser.add_argument("--think_end_token", type=str, default="</think>")
    parser.add_argument("--enable_thinking", type=str, default=None,
                        choices=["true", "false"])

    parser.add_argument("--output_dir", type=str, default="./results_scripts")

    args = parser.parse_args()

    if args.enable_thinking == "true":
        args.enable_thinking = True
    elif args.enable_thinking == "false":
        args.enable_thinking = False
    else:
        args.enable_thinking = None

    if args.provider == "gemini":
        if not GEMINI_AVAILABLE:
            print("ERROR: google-generativeai not installed")
            print("  pip install google-generativeai")
            exit(1)
        api_key = args.api_key or GOOGLE_API_KEY
        if not api_key:
            print("ERROR: Set --api_key or GOOGLE_API_KEY environment variable")
            exit(1)
        client = GeminiClient(api_key, args.model_name)
    else:  
        if not OPENAI_AVAILABLE:
            print("ERROR: openai not installed")
            print("  pip install openai")
            exit(1)
        if not args.api_base:
            print("ERROR: --api_base required for OpenAI provider")
            exit(1)
        client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=args.api_base)
        if not args.model_path:
            args.model_path = args.model_name

    model_safe = re.sub(r"[^\w\-_.]", "_", args.model_name)
    results_dir = os.path.join(args.output_dir, model_safe)
    os.makedirs(results_dir, exist_ok=True)
    logger = setup_logging(results_dir)

    print(f"\n{'='*65}")
    print(f"SCRIPT-CHANGE EVALUATION")
    print(f"{'='*65}")
    print(f"Provider:    {args.provider}")
    print(f"Model:       {args.model_name}")
    if args.provider == "openai":
        print(f"Model path:  {args.model_path}")
        print(f"API:         {args.api_base}")
    print(f"Benchmark:   {args.benchmark}")
    print(f"Dataset:     {args.dataset_path}")
    print(f"Script:      {args.script_label}")
    print(f"Results:     {results_dir}")
    print(f"Params:      temp={args.temperature} top_p={args.top_p} "
          f"max_tokens={args.max_tokens}")
    print(f"{'='*65}\n")

    eval_fn = BENCHMARKS[args.benchmark]
    await eval_fn(client, args, logger, results_dir, args.script_label, args.provider)

    summary_path = os.path.join(results_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        print(f"\n{'='*65}")
        print(f"SUMMARY (all runs)")
        print(f"{'='*65}")
        for key, val in sorted(summary.items()):
            print(f"  {key:45s}  acc={val.get('accuracy', 'N/A')}  "
                  f"script={val.get('script', '?')}")

    print(f"\nResults saved to: {results_dir}/")


if __name__ == "__main__":
    asyncio.run(async_main())