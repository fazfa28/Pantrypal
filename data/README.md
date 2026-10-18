# Data

Raw downloads go in `data/raw/` (git-ignored). Nothing in this folder except
the small YAML files is committed.

| File | Source | How to get it |
| --- | --- | --- |
| `raw/recipes.parquet` | Food.com Recipes and Reviews (Kaggle, `irkaal/foodcom-recipes-and-reviews`) | Download from Kaggle, keep `recipes.parquet` |
| `raw/foodkeeper.json` | USDA FSIS FoodKeeper data, catalog.data.gov | Download the English JSON |
| `raw/freiburg/` | Freiburg Groceries Dataset (github.com/PhilJd/freiburg_groceries_dataset) | `python download_dataset.py` from that repo, then move `images/` here |
| `raw/grocery_store/` | Grocery Store Dataset (github.com/marcusklasson/GroceryStoreDataset) | `git clone`, keep the `dataset/` folder |
| `raw/fridge/` | Your own 30 fridge and pantry scenes, 3 photos each | Name photos `<scene>_<1-3>.jpg`; label in `eval/manifests/fridge.csv` |

## Build steps (run once)

```bash
# 1. Recipe index: filter to ~5,000 recipes and canonicalise ingredients
python -m core.recipes build data/raw/recipes.parquet

# 2. Shelf-life table from FoodKeeper (replaces the placeholder rows)
python -m scripts.build_shelf_life data/raw/foodkeeper.json

# 3. Evaluation manifests
python -m eval.datasets freiburg data/raw/freiburg/images
python -m eval.datasets grocery data/raw/grocery_store/dataset --split test
```

## Small files you edit by hand

- `aliases.yaml`: the canonical food list. Every other file uses these ids.
- `staples.yaml`: ignored in recipe scoring and shopping lists.
- `substitutes.yaml`: half-credit substitutes for recipe matching.
- `shelf_life.yaml`: days per location; regenerated from FoodKeeper.
- `storage_rules.yaml`: curated storage advice with sources.
- `label_maps/<dataset>.yaml`: dataset label to canonical id.
