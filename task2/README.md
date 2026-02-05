# Task 2: Recipe Chatbot

Local recipe chatbot: you give ingredients (e.g. "egg, onion" or "I have rice and veggies, what can I make?"), it returns a recipe (name, ingredients, steps, time). Runs as an API and a small CLI. No external services.

**The trained adapter is in `artifacts/adapter/`. You don’t need to run training—just install deps and run.**

## Tech in a nutshell

- **Base model:** Qwen 1.5B Instruct (small, runs on a laptop).
- We fine-tuned it with **LoRA** on 600 recipe examples so it answers in a consistent format.
- Only the adapter weights are in the repo; the app loads the base model from Hugging Face once, then uses our adapter.
- Device: uses GPU or MPS if available, else CPU.

## What you need

Python 3.10+. The adapter is already in `artifacts/adapter/` (no training).

## Setup

**Windows:**
```cmd
cd task2
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux/macOS:**
```bash
cd task2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First run of the API will download the base model from Hugging Face once.

## How to run

**Tests:** `python run_tests.py` — runs pytest and a short smoke demo (API + 3 sample queries). Non-zero exit on failure.

**API:** `python -m src.api` — leave it running. Listens on http://127.0.0.1:8000.

**CLI:** In another terminal, `python -m src.cli_chat`. Type ingredients, press Enter. Type `quit` to exit. If the API isn’t running, the CLI can use local inference (needs the adapter).

**Inference only (JSON):** `python -m src.infer --message "egg, onion"` — output is JSON with recipe_title, ingredients, steps, time_minutes, notes.

## Sample input/output

| Input | What to expect |
|-------|----------------|
| `egg, onion` | Recipe like Masala Omelette; JSON with recipe_title, steps (list), etc. |
| `rice, vegetables` | Recipe like Vegetable Fried Rice; same JSON schema. |
| `banana, milk, oats` | Recipe like Banana Oat Smoothie; same JSON schema. |

Response JSON: `query`, `normalized_ingredients`, `recipe_title`, `ingredients`, `steps`, `time_minutes`, `notes`. Exact text may vary; structure is fixed.

## Verification checklist

1. `pip install -r requirements.txt` (after venv activate).
2. `python run_tests.py` — must pass.
3. `python -m src.api` — start server.
4. In another terminal: `python -m src.cli_chat` — type e.g. `egg, onion` and confirm you get a recipe.

If the adapter is missing you’ll get a clear error; we ship it in the repo so you shouldn’t see that.

## Training (optional)

If you need to retrain or change data: `python -m src.train_lora`. Not required for normal use. Data: `data/train.jsonl` and `data/eval.jsonl` (included). Training uses cuda > mps > cpu; config in `src/config.py` (max_seq_len 512, max_steps 600, etc.). Saves to `artifacts/adapter/` and writes `artifacts/training_log.json`.

## Project structure

```
task2/
  data/           train.jsonl, eval.jsonl
  artifacts/
    adapter/      trained LoRA weights (included)
    training_log.json
    eval_report.json
  src/            config, dataset_gen, dataset_format, train_lora, infer, api, cli_chat, eval
  tests/          test_dataset, test_infer_smoke, test_api_smoke
  requirements.txt
  README.md
  run_tests.py
```

## Eval (optional)

`python -m src.eval` — runs inference on the first 20 lines of eval.jsonl, prints a short report (JSON validity, recipe_title present, steps length) and writes `artifacts/eval_report.json`.
