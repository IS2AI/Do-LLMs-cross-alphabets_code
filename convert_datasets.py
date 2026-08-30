"""Build the five script conditions of the three Kazakh benchmarks.

Downloads the Kazakh test data of KazMMLU, KazCulture and GSM8k-Kazakh and writes
one copy per script condition (original Cyrillic + the four Latin proposals) to

    data/<dataset>/<condition>/<file>.jsonl

Only the question and option fields are transliterated. Answer keys, numeric
values, identifiers and every other field are copied verbatim, so the five
conditions differ in orthography alone.

The conversion is deterministic: it contains no sampling and no shuffling, so the
same input rows always produce byte-identical output files.

Run:  python convert_datasets.py [--conditions cyrillic,2017,...] [--datasets kazmmlu,kazculture,gsm8k]
"""
import argparse
import gc
import json
from pathlib import Path

from datasets import load_dataset

from transliterate import LATIN_VERSIONS, SCRIPT_CONDITIONS, transliterate

HERE = Path(__file__).parent
OUT_ROOT = HERE / "data"

# Kazakh-language sections only. The Russian sections of KazMMLU are not evaluated.
KAZMMLU_SUBSETS = [
    "Biology (High School in kaz)",
    "Chemistry (High School in kaz)",
    "Geography (High School in kaz)",
    "Informatics (High School in kaz)",
    "Kazakh History (High School in kaz)",
    "Kazakh Language (High School in kaz)",
    "Kazakh Literature (High School in kaz)",
    "Law (High School in kaz)",
    "Math (High School in kaz)",
    "Physics (High School in kaz)",
    "Reading Literacy (High School in kaz)",
    "World History (High School in kaz)",
]

DATASETS = {
    "kazmmlu": {
        "hf_id": "MBZUAI/KazMMLU",
        "revision": None,
        "split": "test",
        "expected_size": 9870,
        "content_fields": ["Question", "Option A", "Option B", "Option C",
                           "Option D", "Option E"],
        "gold_field": "Answer Key",
    },
    "kazculture": {
        "hf_id": "issai/KazCulture",
        "revision": None,
        "split": "test",
        "expected_size": 1334,
        "content_fields": ["question", "a", "b", "c", "d"],
        "gold_field": "answer_label",
    },
    "gsm8k": {
        "hf_id": "issai/GSM8k_Kazakh_Russian",
        "config": "kazakh",
        "revision": None,
        "split": "test",
        "expected_size": 1319,
        "content_fields": ["question"],
        "gold_field": "answer_text",
    },
}


def convert_row(row, condition, content_fields):
    """Transliterate the content fields only; copy everything else verbatim."""
    if condition == "cyrillic":
        return dict(row)
    out = dict(row)
    for field in content_fields:
        value = out.get(field)
        if isinstance(value, str):
            out[field] = transliterate(value, condition)
    return out


def write_split(rows, out_path, condition, content_fields):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(convert_row(row, condition, content_fields),
                               ensure_ascii=False) + "\n")


def check_gold_unchanged(rows, condition, spec):
    """The gold field must be identical in every condition."""
    field = spec["gold_field"]
    for row in rows:
        original = row.get(field)
        converted = convert_row(row, condition, spec["content_fields"]).get(field)
        if original != converted:
            raise AssertionError(
                f"gold field {field!r} changed under condition {condition}: "
                f"{original!r} -> {converted!r}")


def run_dataset(name, conditions, strict_size=True):
    spec = DATASETS[name]
    manifest = {"dataset": name, "hf_id": spec["hf_id"],
                "revision": spec["revision"], "split": spec["split"],
                "conditions": conditions, "files": {}, "n_rows": 0}

    if name == "kazmmlu":
        parts = [(subset.replace(" (High School in kaz)", "").replace(" ", "_").lower(),
                  subset) for subset in KAZMMLU_SUBSETS]
    else:
        parts = [(name, None)]

    total = 0
    for fname, subset in parts:
        config = subset if subset is not None else spec.get("config")
        print(f"[{name}] loading {config or spec['hf_id']}")
        ds = load_dataset(spec["hf_id"], config, split=spec["split"],
                          revision=spec["revision"])
        rows = list(ds)
        total += len(rows)
        for condition in conditions:
            check_gold_unchanged(rows[:50], condition, spec)
            out_path = OUT_ROOT / name / condition / f"{fname}.jsonl"
            write_split(rows, out_path, condition, spec["content_fields"])
            manifest["files"].setdefault(condition, []).append(str(out_path.relative_to(HERE)))
        del ds, rows
        gc.collect()

    manifest["n_rows"] = total
    print(f"[{name}] {total} rows (expected {spec['expected_size']})")
    if total != spec["expected_size"]:
        message = (f"{name}: got {total} rows, expected {spec['expected_size']}")
        if strict_size:
            raise AssertionError(message + " (pass --skip-size-check to override)")
        print("  WARNING: " + message)

    manifest_path = OUT_ROOT / name / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default=",".join(SCRIPT_CONDITIONS),
                    help=f"comma-separated subset of {SCRIPT_CONDITIONS}")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--skip-size-check", action="store_true")
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in SCRIPT_CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown conditions: {unknown}; expected {SCRIPT_CONDITIONS}")
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown datasets: {unknown}; expected {list(DATASETS)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in datasets:
        run_dataset(name, conditions, strict_size=not args.skip_size_check)
    print(f"done. Latin variants written: {[c for c in conditions if c in LATIN_VERSIONS]}")


if __name__ == "__main__":
    main()
