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
from functools import lru_cache
from pathlib import Path

import pandas as pd

from core import rules
from core.canon import get_canon, normalize
from core.config import cfg, load_yaml, path

# Columns of the Food.com "Recipes and Reviews" dataset (Kaggle: irkaal).
REQUIRED = ["RecipeId", "Name", "TotalTime", "RecipeIngredientParts", "RecipeServings", "Calories", "ReviewCount"]

_R_STRING = re.compile(r'"((?:[^"\\]|\\.)*)"')
_ISO_DURATION = re.compile(r"^PT(?:(\d+)H)?(?:(\d+)M)?")

NON_VEGETARIAN = {"meat", "seafood"}
NON_VEGAN = NON_VEGETARIAN | {"dairy", "egg"}
MEAT_WORDS = ("chicken", "beef", "pork", "bacon", "ham", "lamb", "mutton", "sausage", "turkey", "fish",
              "shrimp", "prawn", "salmon", "tuna", "anchov", "crab", "gelatin", "lard")
ANIMAL_WORDS = MEAT_WORDS + ("milk", "cheese", "butter", "cream", "yogurt", "egg", "honey")


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


def build_index(src: str | Path, out: str | Path | None = None, limit: int | None = None) -> pd.DataFrame:
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
    df["ReviewCount"] = pd.to_numeric(df["ReviewCount"], errors="coerce").fillna(0)
    df = df[
        df["minutes"].notna() & (df["minutes"] <= max_minutes)
        & (df["n_ingr"] > 0) & (df["n_ingr"] <= max_ingr)
        & (df["ReviewCount"] >= cfg("recipes", "min_reviews", default=1))
    ].sort_values("ReviewCount", ascending=False)

    canon, staples = get_canon(), rules.staples()
    rows = []
    for r in df.itertuples(index=False):
        names = parse_list(r.RecipeIngredientParts)
        ids = [canon.match(n).id or f"raw:{normalize(n)}" for n in names]
        core_ids = [i for i in dict.fromkeys(ids) if i not in staples]
        if not core_ids:
            continue
        rows.append({
            "recipe_id": int(r.RecipeId),
            "name": str(r.Name),
            "minutes": int(r.minutes),
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


@lru_cache(maxsize=1)
def load_index() -> pd.DataFrame:
    p = path("recipes")
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
    # Word-start match on the raw ingredient text catches what canonicalisation missed.
    # Known false positives to fix as you test: "peanut butter", "eggplant", "cream of tartar".
    text = " ".join(recipe["ingredients"]).lower()
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
) -> list[dict]:
    """Rank recipes for an inventory. method: 'pantrypal' (match score) or 'jaccard' (baseline)."""
    df = load_index() if df is None else df
    exclude_ids = {get_canon().match(e).id or normalize(e) for e in (exclude or [])}
    scorer = jaccard_score if method == "jaccard" else score_recipe
    results = []
    for rec in df.to_dict("records"):
        if max_minutes and rec["minutes"] > max_minutes:
            continue
        if violates_diet(rec, diet):
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
    return results[:top_k]


def main():
    ap = argparse.ArgumentParser(description="Recipe index tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="filter Food.com recipes and canonicalise ingredients")
    b.add_argument("src", help="data/raw/recipes.parquet (or .csv)")
    b.add_argument("--out", default=None)
    b.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.cmd == "build":
        build_index(args.src, args.out, args.limit)


if __name__ == "__main__":
    main()
