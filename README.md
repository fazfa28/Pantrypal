# PantryPal

Photos of your fridge become an inventory; the app flags items near expiry,
suggests recipes from what you have, and builds a shopping list.

**Design rule:** the model only reads photos and talks. Plain Python does
everything that must be correct: recipe ranking, expiry dates, calories,
storage advice and shopping lists.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-eval.txt                    # app + tests + eval extras
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your OpenRouter key
pytest                                                  # 44 tests, no API calls
```

Download the data (see `data/README.md`), build the recipe index once, then run:

```bash
python -m core.recipes build data/raw/recipes.csv
streamlit run app.py
```

Deploy: push to GitHub, create an app on Streamlit Community Cloud pointing at
`app.py`, and paste the contents of `secrets.toml` into its Secrets box.
The hosted app installs only `requirements.txt`.

## Layout

```
app.py                  Streamlit: Scan | Inventory | Chat | Shopping
config.yaml             model ids, prices, paths, tunable weights (edit here, not in code)
core/
  config.py             loads config.yaml, resolves paths and secrets
  llm.py                one OpenAI-compatible client; optional on-disk response cache
  vlm.py                photos -> validated item JSON; flags items for review
  canon.py              any food name -> canonical id (exact alias, then strict typo-only fuzzy)
  inventory.py          SQLite: items, profile, event log; scan diff and apply
  recipes.py            recipe index build + match-score ranking + baselines
  rules.py              expiry, urgency, Mifflin-St Jeor, storage advice, shopping/restock
  assistant.py          tool-calling chat loop + [[recipe]] grounding check
data/                   small YAML files; raw downloads go in data/raw/
  aliases.yaml          hand-edited canonical food list (wins on conflict)
  aliases_extra.yaml    generated ingredient -> id table (scripts/build_ingredient_map.py)
eval/
  datasets.py           build manifests (Freiburg, Grocery Store); fridge is hand-written
  run_recognition.py    Experiment 1: accuracy, F1, cost per image, latency (--closed-set)
  report.py             Experiment 1 results table, McNemar tests, per-class F1, confusions
  compare.py            McNemar test between two runs (e.g. 1 photo vs 3 photos)
  make_cases.py         Experiment 2 test cases: pantry + natural-language request
  run_retrieval.py      Experiment 2 runner (being extended to the request-based arms)
  metrics.py            accuracy, macro-F1, set P/R/F1, bootstrap CI, McNemar
scripts/
  build_shelf_life.py   USDA FoodKeeper JSON -> data/shelf_life.yaml
  build_ingredient_map.py  one-off LLM mapping of recipe ingredient strings to ids
  sample_mappings.py    100-string before/after check of ingredient mapping
tests/                  pytest, no network
```

## Experiments

The MVP is evaluated on PantryPal's two main functions.

### Experiment 1: ingredient recognition (done)

*Which open-weight vision-language model identifies grocery items best per dollar?*

- **Data:** Freiburg Groceries, 4,947 images of single products in 25 classes (200 dev, 4,747 test).
- **Task:** closed-set classification. Every model gets the same frozen prompt listing the 25 classes and must answer with one of them.
- **Models:** GLM 5.3 Flash, Gemma 4 26B, Qwen3-VL 235B, Qwen3-VL 8B, Llama 4 Scout (via OpenRouter).
- **Metrics:** top-1 accuracy (bootstrap 95% CI), macro-F1, valid-JSON rate, US$ per image, latency, paired McNemar tests.
- **Result:** GLM 5.3 Flash is best: 96.0% accuracy, US$0.026 per 1,000 images, significantly ahead of every other model.
- **Not yet done:** the multi-photo fridge scenes (1 vs 3 photos) need the team's own photo set.

```bash
python -m eval.datasets freiburg data/raw/images
python -m eval.run_recognition eval/manifests/freiburg.csv --model glm-5.3-flash --split dev --limit 20 --closed-set
python -m eval.run_recognition eval/manifests/freiburg.csv --model glm-5.3-flash --split test --closed-set
python -m eval.report                      # table + McNemar + per-class F1 -> eval/results/report.md
```

### Experiment 2: recipe recommendation (in progress)

*Does PantryPal's design (an LLM to understand the request, code to retrieve and check recipes)
recommend better recipes than the usual alternatives?*

- **Test cases:** 60 (10 dev, 50 test). Each is a realistic pantry plus a request such as
  "Vegetarian dinner for 2 under 20 minutes, something spicy, use up the spinach",
  generated from a structured spec (diet, time limit, exclusion, use-up item, soft preference).
- **Arms** (same chat model, GLM 5.3 Flash, for all):
  - **LLM-only:** pantry and request go straight to the model, which writes 5 recipes.
  - **RAG:** embedding search (MiniLM) over the 5,000 recipes finds 20 candidates; the model picks 5.
  - **PantryPal:** the model turns the request into filters; code ranks recipes by ingredient
    coverage and returns 20 candidates; the model picks 5 with the same prompt as RAG.
- **Automatic metrics (code):** grounding (recipe exists in the database), cookable@5
  (at most 1 ingredient missing and at least 3 in stock, staples excluded), hallucinated
  ingredients (LLM-only), diet/time/exclusion violations, use-up rate, cost and latency.
- **Judged metric:** preference fit, 1–5 against a rubric, by GPT-6.1 Sol (blind to the arm),
  with Qwen3-VL 235B on a subset and teammate ratings on 40 items to check agreement.
- **Prerequisite done:** recipe ingredients are mapped to canonical ids by a reviewed lookup
  table; on a 100-string sample, wrong matches fell from 19% to 1%.

```bash
python -m eval.make_cases                  # writes eval/cases/exp2_cases.jsonl + review CSV
```

The fridge photo set and the user study are on hold.

Every model reply is cached in `eval/cache/`, keyed by the full request, and
finished rows are skipped on rerun. Recomputing metrics costs nothing.
Set a credit limit on the OpenRouter key before bulk runs.

## Team ownership

| Role | Owns |
| --- | --- |
| Vision lead | `core/vlm.py`, `core/canon.py`, `eval/run_recognition.py` |
| Recipe lead | `core/recipes.py`, `data/substitutes.yaml`, `eval/run_retrieval.py` |
| App lead | `app.py`, `core/inventory.py`, deployment |
| Assistant lead | `core/assistant.py`, scripted chat tests |
| Rules lead | `core/rules.py`, `data/storage_rules.yaml`, `scripts/build_shelf_life.py` |
| Data and study lead | `data/aliases.yaml`, label maps, fridge manifest, user study |

