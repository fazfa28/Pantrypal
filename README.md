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
pytest                                                  # 37 tests, no API calls
```

Download the data (see `data/README.md`), build the recipe index once, then run:

```bash
python -m core.recipes build data/raw/recipes.parquet
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
  canon.py              any food name -> canonical id (exact, fuzzy, optional embeddings)
  inventory.py          SQLite: items, profile, event log; scan diff and apply
  recipes.py            recipe index build + match-score ranking + baselines
  rules.py              expiry, urgency, Mifflin-St Jeor, storage advice, shopping/restock
  assistant.py          tool-calling chat loop + [[recipe]] grounding check
data/                   small hand-edited YAML files; raw downloads go in data/raw/
eval/
  datasets.py           build manifests (Freiburg, Grocery Store); fridge is hand-written
  run_recognition.py    Experiment 1: accuracy, F1, cost per image, latency
  compare.py            McNemar test between two runs (e.g. 1 photo vs 3 photos)
  run_retrieval.py      Experiment 2: PantryPal vs Jaccard vs LLM-only
  metrics.py            accuracy, macro-F1, set P/R/F1, bootstrap CI, McNemar
scripts/
  build_shelf_life.py   USDA FoodKeeper JSON -> data/shelf_life.yaml
tests/                  pytest, no network
```

## Running the experiments

```bash
# Manifests
python -m eval.datasets freiburg data/raw/freiburg/images
python -m eval.datasets grocery  data/raw/grocery_store/dataset --split test
cp eval/manifests/fridge_template.csv eval/manifests/fridge.csv    # then add your 30 scenes

# Experiment 1: tune the prompt on --split dev first, then freeze it
python -m eval.run_recognition eval/manifests/freiburg.csv --model qwen3-vl-8b --limit 20   # cost check
python -m eval.run_recognition eval/manifests/freiburg.csv --model qwen3-vl-8b
python -m eval.run_recognition eval/manifests/fridge.csv --model qwen3-vl-8b --photos first
python -m eval.run_recognition eval/manifests/fridge.csv --model qwen3-vl-8b --photos all
python -m eval.compare eval/results/fridge__qwen3-vl-8b__first__test.jsonl \
                       eval/results/fridge__qwen3-vl-8b__all__test.jsonl

# Experiment 2
python -m eval.run_retrieval --fridge eval/manifests/fridge.csv --random 20
python -m eval.run_retrieval --predicted eval/results/fridge__qwen3-vl-8b__all__test.jsonl --methods pantrypal
```

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

## Before the real runs (TODOs in the code)

- [ ] Regenerate `data/shelf_life.yaml` from FoodKeeper (current rows are placeholders).
- [ ] Verify and cite each rule in `data/storage_rules.yaml`; grow it to about 30.
- [ ] Cite a guideline for `rules.calorie_floor` and `GOAL_ADJUSTMENT`.
- [ ] Confirm Molmo 2 8B has a live endpoint; re-check prices in `config.yaml`.
- [ ] Grow `data/aliases.yaml` until `core.recipes build` reports few unmatched ingredients.
- [ ] Check the Freiburg folder names against `data/label_maps/freiburg.yaml`.
- [ ] `log_cooked` removes one unit per ingredient; ask users for real quantities.
