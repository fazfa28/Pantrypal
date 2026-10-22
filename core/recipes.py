"""Recipe index and match-score ranking.

Score(I, D) = coverage + alpha * urgency - beta * |missing|, over the recipe's
non-staple ingredients R (all ids canonicalised, so "scallion" == "spring onion"):
  m(r) = 1 if in stock, 0.5 if a listed substitute is in stock, else 0
  u(r) = urgency of the matched item (1 if expiring within 2 days, 0.5 within 5)
Diet, time and exclusions are hard filters applied before ranking.

Build the index once:  python -m core.recipes build data/raw/recipes.parquet
"""
from __future__ import annotations

import argparse
import math
import re
import threading
from functools import lru_cache
from pathlib import Path

import pandas as pd

from core import rules
from core.canon import get_canon, normalize
from core.config import cfg, load_yaml, path

# Columns of the Food.com "Recipes and Reviews" dataset (Kaggle: irkaal).
REQUIRED = ["RecipeId", "Name", "TotalTime", "RecipeIngredientParts", "RecipeServings", "Calories", "ReviewCount"]

_R_STRING = re.compile(r'"((?:[^"\\]|\\.)*)"')
_R_ITEM = re.compile(r'"(?:[^"\\]|\\.)*"|\bNA\b')   # quoted strings and bare NA, for counting R vectors
_ISO_DURATION = re.compile(r"^PT(?:(\d+)H)?(?:(\d+)M)?")

NON_VEGETARIAN = {"meat", "seafood"}
NON_VEGAN = NON_VEGETARIAN | {"dairy", "egg"}
MEAT_WORDS = ("chicken", "beef", "pork", "bacon", "ham", "lamb", "mutton", "sausage", "turkey", "fish",
              "shrimp", "prawn", "salmon", "tuna", "anchov", "crab", "gelatin", "lard", "veal", "steak",
              "venison", "duck", "meat", "lobster", "tilapia", "cod", "scallop", "clam", "mussel", "chorizo",
              "pepperoni", "prosciutto", "salami", "oyster", "worcestershire", "bouillon")
ANIMAL_WORDS = MEAT_WORDS + ("milk", "cheese", "butter", "cream", "yogurt", "egg", "honey", "mayonnaise", "mayo")
# Plant-based phrases removed before the word check, so "soy milk" or "peanut butter" is not flagged.
_PLANT = re.compile(
    r"\b(?:soy|soya|almond|oat|rice|coconut|cashew|hemp|vegan|plant[- ]based|non[- ]dairy|dairy[- ]free)"
    r"\s+(?:milk|butter|cream|cheese|yogurt|yoghurt)s?\b"
    r"|\b(?:peanut|almond|cashew|nut|apple|cocoa|sunflower|seed)\s+butter\b"
    r"|\bbutternut\b|\bcream of tartar\b|\bcream of coconut\b|\beggplants?\b|\begg replacer\b")


# Meal type of a recipe, from Food.com's RecipeCategory plus a few name words (categories are noisy).
DESSERT_CATS = {"Dessert", "Drop Cookies", "Bar Cookie", "Pie", "Cheesecake", "Candy", "Frozen Desserts",
                "Tarts", "Gelatin", "Ice Cream", "Brownie", "Cookie & Brownie"}
BREAKFAST_CATS = {"Breakfast", "Quick Breads", "Scones", "Smoothies", "Breakfast Eggs", "Oatmeal"}
NOT_A_MAIN = {"Sauces", "Spreads", "Salad Dressings", "Beverages", "Shakes", "Punch Beverage", "Jellies",
              "Chutneys", "Marinades", "Lemonade", "Breads", "Yeast Breads"} | DESSERT_CATS | BREAKFAST_CATS
NOT_A_MAIN_WORDS = ("sauce", "dressing", "dip", "marinade", "seasoning", "rub", "spread", "smoothie", "drink",
                    "buns", "bread", "cookie", "frosting", "glaze", "syrup", "vinaigrette", "mix", "batter",
                    "relish", "oil", "jam", "jelly", "butter", "salsa", "chutney", "pesto", "gravy", "essence",
                    "topper", "topping", "dog treat", "dog")


def suits_meal(rec: dict, meal: str | None) -> bool:
    """meal: 'dinner' | 'breakfast' | 'dessert' | None (no filter)."""
    if not meal:
        return True
    cat = rec.get("category", "") or ""
    if meal == "dessert":
        return cat in DESSERT_CATS
    if meal == "breakfast":
        return cat in BREAKFAST_CATS
    name = str(rec.get("name", "")).lower()
    return cat not in NOT_A_MAIN and not any(re.search(rf"\b{w}s?\b", name) for w in NOT_A_MAIN_WORDS)


# ---------- parsing helpers ----------


def parse_list(v) -> list[str]:
    """Parquet arrays, Python lists, or R-style 'c("a", "b")' strings -> list[str]."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return []
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if x is not None]
    s = str(v)
    if s.startswith("c(") or s.startswith('"'):
        return _R_STRING.findall(s)
    return [s] if s and s not in ("nan", "character(0)") else []


def iso_minutes(s) -> int | None:
    """'PT1H30M' -> 90"""
    m = _ISO_DURATION.match(str(s or ""))
    if not m or not any(m.groups()):
        return None
    return int(m.group(1) or 0) * 60 + int(m.group(2) or 0)


def _num(v, default=None):
    try:
        f = float(v)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


# ---------- build ----------


def build_index(src: str | Path, out: str | Path | None = None, limit: int | None = None,
                use_staples: bool = True) -> pd.DataFrame:
    src = Path(src)
    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{src} is missing columns {missing}; expected the Food.com Kaggle schema")

    max_minutes = cfg("recipes", "max_minutes", default=60)
    max_ingr = cfg("recipes", "max_ingredients", default=12)
    limit = limit or cfg("recipes", "limit", default=5000)

    # Cheap vectorised filters first, most-reviewed recipes first.
    df["minutes"] = df["TotalTime"].map(iso_minutes)
    df["n_ingr"] = df["RecipeIngredientParts"].map(lambda v: len(parse_list(v)))
    # Food.com quirk: some ingredient names are silently dropped (e.g. "Chicken Noodle Soup" without chicken).
    # Such recipes have fewer names than quantities; drop them, since their ingredient list is incomplete.
    if "RecipeIngredientQuantities" in df.columns:
        n_qty = df["RecipeIngredientQuantities"].map(lambda v: len(_R_ITEM.findall(str(v))))
        df = df[df["n_ingr"] >= n_qty]
    df["ReviewCount"] = pd.to_numeric(df["ReviewCount"], errors="coerce").fillna(0)
    df = df[
        df["minutes"].notna() & (df["minutes"] <= max_minutes)
        & (df["n_ingr"] > 0) & (df["n_ingr"] <= max_ingr)
        & (df["ReviewCount"] >= cfg("recipes", "min_reviews", default=1))
    ].sort_values("ReviewCount", ascending=False)

    canon, staples = get_canon(), rules.staples(use_staples)
    rows = []
    for r in df.itertuples(index=False):
        names = parse_list(r.RecipeIngredientParts)
        ids = [canon.match(n).id or f"raw:{normalize(n)}" for n in names]
        core_ids = [i for i in dict.fromkeys(ids) if i not in staples]
        if len(core_ids) < cfg("recipes", "min_core_ingredients", default=3):
            continue
        rows.append({
            "recipe_id": int(r.RecipeId),
            "name": str(r.Name),
            "minutes": int(r.minutes),
            "category": str(getattr(r, "RecipeCategory", "") or "").replace("nan", ""),
            "servings": _num(r.RecipeServings, 1.0) or 1.0,
            "calories_per_serving": _num(r.Calories),
            "review_count": int(r.ReviewCount),
            "rating": _num(getattr(r, "AggregatedRating", None)),
            "ingredients": names,
            "ingredient_ids": core_ids,
            "instructions": parse_list(getattr(r, "RecipeInstructions", None)),
        })
        if len(rows) >= limit:
            break

    index = pd.DataFrame(rows)
    out = Path(out or path("recipes"))
    out.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(out, index=False)
    unmatched = sum(i.startswith("raw:") for ids in index["ingredient_ids"] for i in ids)
    total = sum(len(ids) for ids in index["ingredient_ids"])
    print(f"Saved {len(index)} recipes to {out}; {unmatched}/{total} ingredients not canonicalised "
          f"(add aliases to data/aliases.yaml to reduce this).")
    return index


@lru_cache(maxsize=4)
def load_index(file: str | None = None) -> pd.DataFrame:
    p = Path(file) if file else path("recipes")
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run: python -m core.recipes build data/raw/recipes.parquet")
    df = pd.read_parquet(p)
    for col in ("ingredients", "ingredient_ids", "instructions"):
        df[col] = df[col].map(parse_list)
    return df


# ---------- scoring ----------


def score_recipe(ingredient_ids: list[str], have: dict[str, int | None], alpha: float | None = None,
                 beta: float | None = None) -> dict:
    """have: canonical_id -> days left (None if unknown). See module docstring for the formula."""
    alpha = cfg("recipes", "alpha", default=0.3) if alpha is None else alpha
    beta = cfg("recipes", "beta", default=0.05) if beta is None else beta
    subs = load_yaml("substitutes")
    n = len(ingredient_ids) or 1
    coverage = urgency = 0.0
    missing, substituted = [], {}
    for r in ingredient_ids:
        if r in have:
            coverage += 1.0
            urgency += rules.urgency(have[r])
            continue
        sub = next((s for s in subs.get(r, []) if s in have), None)
        if sub:
            coverage += 0.5
            urgency += rules.urgency(have[sub])
            substituted[r] = sub
        else:
            missing.append(r)
    return {
        "score": coverage / n + alpha * urgency / n - beta * len(missing),
        "coverage": coverage / n,
        "urgency": urgency / n,
        "missing": missing,
        "substituted": substituted,
    }


def is_cookable(ingredient_ids, have, max_missing: int | None = None, min_matched: int | None = None) -> bool:
    """One rule for every method in Experiment 2: few enough non-staple ids missing AND enough in stock,
    so a recipe that is mostly staples does not count as cookable by default. No substitute credit."""
    max_missing = cfg("recipes", "cookable_max_missing", default=1) if max_missing is None else max_missing
    min_matched = cfg("recipes", "cookable_min_matched", default=3) if min_matched is None else min_matched
    ids, have = set(ingredient_ids), set(have)
    return len(ids - have) <= max_missing and len(ids & have) >= min_matched


def jaccard_score(ingredient_ids: list[str], have: dict) -> dict:
    """Baseline for Experiment 2: plain set overlap, no substitutes, no expiry term."""
    r, i = set(ingredient_ids), set(have)
    return {"score": len(r & i) / (len(r | i) or 1), "missing": sorted(r - i)}


def violates_diet(recipe: dict, diet: str | None) -> bool:
    if not diet or diet == "none":
        return False
    banned_cats = NON_VEGAN if diet == "vegan" else NON_VEGETARIAN
    words = ANIMAL_WORDS if diet == "vegan" else MEAT_WORDS
    cats = get_canon().category
    if any(cats.get(i) in banned_cats for i in recipe["ingredient_ids"]):
        return True
    # Word-start match on the raw ingredient text AND the recipe name catches what canonicalisation missed
    # (and ingredients the dataset dropped). Plant-based phrases are removed first (_PLANT).
    # Known false positive left: "hamburger buns".
    text = _PLANT.sub(" ", (" ".join(recipe["ingredients"]) + " " + str(recipe.get("name", ""))).lower())
    return any(re.search(rf"\b{w}", text) for w in words)


def search(
    have: dict[str, int | None],
    max_minutes: int | None = None,
    diet: str | None = None,
    exclude: list[str] | None = None,
    top_k: int = 5,
    calorie_target: float | None = None,
    method: str = "pantrypal",
    df: pd.DataFrame | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    meal: str | None = None,
    query: str | None = None,
) -> list[dict]:
    """Rank recipes for an inventory. method: 'pantrypal' (match score) or 'jaccard' (baseline).

    With a `query` (the user's request in their words), PantryPal ranks in three steps:
      1. hard filters (diet, time, meal, exclusions),
      2. feasibility gate: keep recipes that are cookable from the inventory (is_cookable),
      3. order those by semantic similarity to the query (MiniLM recipe embeddings);
    if fewer than top_k pass the gate, the rest are filled by match score. Without a query, or if
    sentence-transformers is not installed, ranking is by match score only (the original behaviour).
    """
    df = load_index() if df is None else df
    exclude_ids = {get_canon().match(e).id or normalize(e) for e in (exclude or [])}
    scorer = jaccard_score if method == "jaccard" else (lambda ids, h: score_recipe(ids, h, alpha, beta))
    results = []
    for rec in df.to_dict("records"):
        if max_minutes and rec["minutes"] > max_minutes:
            continue
        if violates_diet(rec, diet):
            continue
        if meal and meal != "any" and not suits_meal(rec, meal):
            continue
        if exclude_ids & set(rec["ingredient_ids"]):
            continue
        s = scorer(rec["ingredient_ids"], have)
        results.append({**rec, **s})

    def key(r):
        # Calorie fit (per serving vs one meal = a third of the day) only breaks ties.
        cal_gap = abs((r.get("calories_per_serving") or 0) - calorie_target / 3) if calorie_target else 0
        return (-round(r["score"], 3), cal_gap, -r.get("review_count", 0))

    results.sort(key=key)
    if query and method == "pantrypal":
        emb = recipe_embeddings(df)
        if emb is not None:
            sim = emb.similarity(query)
            feasible = [r for r in results if is_cookable(r["ingredient_ids"], have)]
            feasible.sort(key=lambda r: -sim.get(int(r["recipe_id"]), 0.0))
            ids = {int(r["recipe_id"]) for r in feasible[:top_k]}
            return feasible[:top_k] + [r for r in results if int(r["recipe_id"]) not in ids][: max(0, top_k - len(ids))]
    return results[:top_k]


# ---------- recipe embeddings (semantic ranking; also the RAG baseline in Experiment 2) ----------


class RecipeEmbeddings:
    """MiniLM embeddings of 'name. category. N min. Ingredients: ...' for every recipe in the index,
    cached next to the index file and rebuilt when the index changes."""

    def __init__(self, df: pd.DataFrame):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.np = np
        self.model = SentenceTransformer(cfg("canon", "embedding_model", default="all-MiniLM-L6-v2"))
        self.lock = threading.Lock()
        self.ids = df["recipe_id"].astype(int).to_numpy()
        emb_file, ids_file = path("recipes").with_name("recipe_emb.npy"), path("recipes").with_name("recipe_emb_ids.npy")
        if emb_file.exists() and ids_file.exists() and np.array_equal(np.load(ids_file), self.ids):
            self.emb = np.load(emb_file)
        else:
            texts = [f"{r['name']}. {r.get('category', '')}. {r['minutes']} min. Ingredients: {', '.join(r['ingredients'])}."
                     for r in df.to_dict("records")]
            print(f"embedding {len(texts)} recipes (once)...")
            self.emb = self.model.encode(texts, normalize_embeddings=True, batch_size=64)
            np.save(emb_file, self.emb)
            np.save(ids_file, self.ids)

    def scores(self, text: str):
        """Cosine similarity of `text` to every recipe, in index order."""
        with self.lock:
            q = self.model.encode([text], normalize_embeddings=True)[0]
        return self.emb @ q

    def similarity(self, text: str) -> dict[int, float]:
        return dict(zip(self.ids.tolist(), self.scores(text).tolist()))

    def top(self, text: str, k: int) -> list[int]:
        """Row positions of the k most similar recipes."""
        return [int(i) for i in self.np.argsort(-self.scores(text))[:k]]


_EMB: dict[int, RecipeEmbeddings] = {}


def recipe_embeddings(df: pd.DataFrame | None = None) -> RecipeEmbeddings | None:
    """Shared embeddings for an index (None if sentence-transformers is not installed)."""
    df = load_index() if df is None else df
    if id(df) not in _EMB:
        try:
            _EMB[id(df)] = RecipeEmbeddings(df)
        except ImportError:
            return None
    return _EMB[id(df)]


def main():
    ap = argparse.ArgumentParser(description="Recipe index tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="filter Food.com recipes and canonicalise ingredients")
    b.add_argument("src", help="data/raw/recipes.parquet (or .csv)")
    b.add_argument("--out", default=None)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--no-staples", action="store_true", help="keep salt, oil, etc. as real ingredients (ablation)")
    args = ap.parse_args()
    if args.cmd == "build":
        build_index(args.src, args.out, args.limit, use_staples=not args.no_staples)


if __name__ == "__main__":
    main()
