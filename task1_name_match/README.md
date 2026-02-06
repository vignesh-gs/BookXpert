# Name Matching System

Finds the best-matching names from a dataset for a given query. Handles punctuation (e.g. `Vignesh.G.S` vs `Vignesh G S`), transliteration (Geetha/Gita/Geeta), initials (Vignesh K R vs Vignesh Kumar R), and typos (amal vs aman). Returns a ranked list with scores.

## How it works

- **Phonetic index:** At startup we build a key from the first name so we don’t compare against every row—we only score a shortlist. Lookup is O(1); we do O(k) scoring with k much smaller than n.
- **Scoring:** Weighted mix of first-name match, edit distance, other name parts, initials, and phonetic similarity, minus penalties (e.g. missing initials, length difference). Same-length, character-similar names (aman, amar) rank above longer substring matches (ajmal) for a query like amal.
- **Punctuation → spaces:** We replace punctuation with spaces so `Vignesh.G.S` becomes `vignesh g s` and we keep initials; removing punctuation would give `vigneshgs` and break that.

If the shortlist is too small we merge in the first-letter bucket; if still empty we fall back to a full scan. Shortlist is capped at 2000 so worst-case time is bounded.

## Setup

**Prerequisite:** Python 3.x. No databases or external services.

**Windows:**
```cmd
cd task1_name_match
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux/macOS:**
```bash
cd task1_name_match
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

Commands below use the default data file `data/Indian_Names.csv` unless you pass `--data`.

**Interactive:** `python -m src.cli` — then type a name. Use `top N` to change result count, `quit` or `exit` to exit.

**CLI:**
```bash
python -m src.cli --name "Geetha B.S" --top_k 5
python -m src.cli --name "Vignesh G.S" --json
python -m src.cli --name "amal" --data data/names.csv
```

**Using the data folder:** `data/` includes two CSVs (each with a `name` or `Name` column):
- **`data/Indian_Names.csv`** — larger set (~6.5k names); **default** for both CLI and interactive mode.
- **`data/names.csv`** — smaller set (~300 names); good for quick tests.

Use `--data` to choose which file:
```bash
python -m src.cli --name "Geetha B.S" --data data/Indian_Names.csv
python -m src.cli --name "amal" --data data/names.csv
```
For interactive mode, pass `--data` when starting: `python -m src.cli --data data/names.csv`.

**Tests:** `python run_tests.py` or `pytest tests/ -v`

## Sample inputs

| Query | What to expect |
|-------|----------------|
| `amal` | Top: aman, amar, amil, amol (same length, 1-char diff); ajmal below (length penalty). |
| `Geetha B.S` | Best: Geetha B S or equivalent; initials match. |
| `Gita` | Best: Gita; Geetha/Geeta also score well (phonetic). |
| `Vignesh Kumar R` | Best: Vignesh Kumar R; Vignesh K R next; Vignesh Kumar (no R) lower. |

**Verification (sample run):** From `task1_name_match/` with venv activated, run:
```bash
python -m src.cli --name "Geetha B.S" --top_k 3 --data data/names.csv
```
You should see a ranked list with at least one match containing "Geetha" and scores (e.g. best match score in the 80–100 range). To run the full test suite: `python run_tests.py`.

## Project structure

```
task1_name_match/
├── data/           Indian_Names.csv (default), names.csv
├── src/             config, normalize, phonetic, initials, index, scoring, matcher, cli
├── tests/           test_normalize, test_phonetic, test_initials, test_matcher_rankings
├── requirements.txt
├── run_tests.py
└── README.md
```

## Dependencies

- **rapidfuzz** (≥3.0.0) — fuzzy matching (WRatio) and Levenshtein
- **pytest** (≥7.0.0) — tests

## Limitations

- Phonetic rules are tuned for Hindi/Sanskrit-style transliteration; other scripts may need more rules.
- Nicknames (e.g. Vicky → Vignesh) are not mapped.

---

## Reference: score formula and components

Final score = weighted sum of components (each 0–100) minus penalties, clamped to [0, 100].

**Components:** first_name (0.30), edit_distance (0.15), other_core (0.20), initials (0.15), phonetic_core (0.10), full_string (0.10). All use WRatio or Levenshtein-derived values.

**Penalties:** missing_initial (12 per), extra_initial (4 per), missing_core (6 per token), overlong (3 per token), length_diff (8 per char). When the first name is a strong but not exact match (e.g. Ganu vs Ganesh), initial and length penalties are softened so close first names can outrank initial-only matches.

**Token types:** Core = full name parts (length > 2); initial = single letter; merged initials (e.g. gs, bs) expanded to two letters. Punctuation → spaces so token boundaries stay correct.

**Phonetic rules (examples):** ee→i, aa→a, oo→u, th→t, sh→s; double consonants collapsed. Used for the index key and phonetic_core_score so Geetha/Gita/Geeta match.
