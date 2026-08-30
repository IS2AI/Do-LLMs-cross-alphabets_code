# Do LLMs Cross Alphabets?

## What each file does

| File | Purpose |
|---|---|
| `transliterate.py` | Kazakh Cyrillic to Latin transliteration for the 2017, 2018, 2019 and 2021 alphabet proposals. |
| `test_transliterate.py` | Unit tests for the transliteration. |
| `convert_datasets.py` | Downloads KazMMLU, KazCulture and GSM8k-Kazakh and writes one copy per script condition to `data/`. |
| `models.json` | Model, tokenizer, endpoint and inference-mode registry used by the evaluation scripts. |
| `eval_models.py` | Evaluates served open-weight models under the baseline, hint and mapping prompt conditions. |
| `compute_fertility.py` | Tokenizer fertility for every tokenizer, benchmark and script condition. |
| `proprietary/evaluate_scripts_gpt_batch.py` | Evaluates the OpenAI models through the Batch API. |
| `proprietary/evaluate_scripts_gemini.py` | Evaluates the Gemini models. |

## Order

```bash
python test_transliterate.py
python convert_datasets.py
python eval_models.py --prompt baseline
python eval_models.py --prompt hint
python eval_models.py --prompt mapping
python eval_models.py --mode think
python compute_fertility.py
python proprietary/evaluate_scripts_gpt_batch.py --help
python proprietary/evaluate_scripts_gemini.py --help
```

Results are written to `results/`: one JSONL and one `.meta.json` per run in
`results/raw/`, one row per run in `results/summary.csv`, and
`results/fertility.csv`.
