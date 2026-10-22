"""Experiment 2: recipe retrieval vs two baselines.

Test inventories: the fridge manifest's ground truth (or a recognition run's
predictions) plus N random ones with random days-left values.

  python -m eval.run_retrieval --fridge eval/manifests/fridge.csv --random 20
  python -m eval.run_retrieval --random 50 --methods pantrypal,jaccard      # no API cost
  python -m eval.run_retrieval --predicted eval/results/fridge__qwen3-vl-8b__all__test.jsonl
  python -m eval.run_retrieval --random 50 --seed 1 --no-staples --index data/recipes_nostaples.parquet

Expiry is OFF by default (alpha = 0, no expiry dates, no expiry metric); add --expiry to turn it on.

Metrics (top 5):
  cookable@5      share of recipes with at most 1 non-staple ingredient missing AND at least 3 in stock
                  (core.recipes.is_cookable; config recipes.cookable_*). Same rule for every method,
                  no substitute credit.
  cookable_subs@5 PantryPal only: same, but a listed substitute counts as having the ingredient
  expiry_use@5    (--expiry only) share of recipes using an item that expires within 3 days
  diet_violation  share of recipes that break the requested diet (run with --diet)
  hallucinated    LLM-only: share of claimed "uses" ingredients that are not in stock
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter

from core import recipes as rx
from core.canon import get_canon
from core.config import path
from core.llm import chat
from core.rules import staples
from eval.datasets import label_mapper, load_manifest

LLM_ONLY_PROMPT = """My kitchen has: {items}.
Suggest {k} recipes I can cook. Return JSON only:
{{"recipes": [{{"name": str, "uses": [ingredients I have], "missing": [ingredients I must buy]}}]}}"""


def random_inventories(n: int, df, seed: int = 0, min_recipes: int = 20, expiry: bool = False,
                       use_staples: bool = True) -> list[dict]:
    """Random pantries of 8-20 ids, drawn only from ids used by at least `min_recipes` recipes in the index
    (rare ids like adobo_sauce would make most inventories unrealistic). Days-to-expiry is None unless expiry."""
    rng = random.Random(seed)
    freq = Counter(i for ids in df["ingredient_ids"] for i in set(ids))
    st = staples(use_staples)
    pool = sorted(i for i, c in freq.items() if c >= min_recipes and i not in st and not i.startswith("raw:"))
    return [{"id": f"random{seed}_{k:02d}",
             "have": {i: (rng.randint(0, 14) if expiry else None) for i in rng.sample(pool, rng.randint(8, 20))}}
            for k in range(n)]


def fridge_inventories(manifest: str, seed: int = 0) -> list[dict]:
    rng, to_id = random.Random(seed), label_mapper("fridge")
    return [{"id": r["id"], "have": {to_id(l): rng.randint(0, 14) for l in r["labels"]}}
            for r in load_manifest(manifest)]


def predicted_inventories(results_file: str, seed: int = 0) -> list[dict]:
    rng, out = random.Random(seed), []
    for line in open(results_file, encoding="utf-8"):
        r = json.loads(line)
        out.append({"id": r["id"] + "_pred", "have": {p["id"]: rng.randint(0, 14) for p in r["pred"] if p["id"]}})
    return out


def score_ranked(recipes: list[dict], have: dict, diet: str | None, expiry: bool = False) -> dict:
    n = len(recipes) or 1
    out = {
        # Same rule for every method: no substitute credit.
        "cookable": sum(rx.is_cookable(r["ingredient_ids"], have) for r in recipes) / n,
        # Same rule, but an ingredient covered by a listed substitute counts as in stock.
        "cookable_subs": sum(rx.is_cookable(r["ingredient_ids"], set(have) | set(r.get("substituted") or {}))
                             for r in recipes) / n,
        "diet_violation": sum(rx.violates_diet(r, diet) for r in recipes) / n if diet else None,
    }
    if expiry:
        expiring = {i for i, d in have.items() if d is not None and d <= 3}
        out["expiry_use"] = sum(bool(expiring & set(r["ingredient_ids"])) for r in recipes) / n
    return out


def llm_only(have: dict, k: int, diet: str | None, expiry_on: bool = False, use_staples: bool = True) -> dict:
    canon, st = get_canon(), staples(use_staples)
    items = ", ".join(sorted(i.replace("_", " ") for i in have)) + (f". I am {diet}" if diet else "")
    out = chat([{"role": "user", "content": LLM_ONLY_PROMPT.format(items=items, k=k)}], use_cache=True)
    text = out["content"]
    try:
        recs = json.loads(text[text.find("{"): text.rfind("}") + 1])["recipes"][:k]
    except (ValueError, KeyError, TypeError):
        return {"cookable": 0.0, "hallucinated": None, "valid_json": False}
    expiring = {i for i, d in have.items() if d is not None and d <= 3}
    cookable = expiry = claimed = fake = 0
    for r in recs:
        uses = {canon.match(x).id or f"raw:{x}" for x in r.get("uses", [])} - st
        missing = {canon.match(x).id or f"raw:{x}" for x in r.get("missing", [])} - st
        fake += len(uses - set(have))
        claimed += len(uses)
        # A recipe's true needs are what it says it uses plus what it says is missing.
        cookable += rx.is_cookable(uses | missing, have)
        expiry += bool(expiring & uses)
    n = len(recs) or 1
    out = {"cookable": cookable / n, "hallucinated": fake / claimed if claimed else 0.0, "valid_json": True}
    if expiry_on:
        out["expiry_use"] = expiry / n
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fridge", help="fridge manifest (ground-truth inventories)")
    ap.add_argument("--predicted", help="recognition results .jsonl (predicted inventories)")
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--methods", default="pantrypal,jaccard,llm_only")
    ap.add_argument("--diet", choices=["vegetarian", "vegan"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="seed for random inventories (use a different one for tuning)")
    ap.add_argument("--alpha", type=float, default=0.0, help="expiry weight; 0 = expiry ignored")
    ap.add_argument("--beta", type=float, help="missing-ingredient penalty (default: config)")
    ap.add_argument("--expiry", action="store_true", help="give inventories expiry days and report expiry_use")
    ap.add_argument("--no-staples", action="store_true", help="treat salt, oil etc. as real ingredients")
    ap.add_argument("--index", help="recipe index parquet (default: config); use a --no-staples build with --no-staples")
    ap.add_argument("--min-recipes", type=int, default=20, help="random inventories only use ids in at least N recipes")
    a = ap.parse_args()
    use_staples = not a.no_staples
    df = rx.load_index(a.index)

    invs = []
    if a.fridge:
        invs += fridge_inventories(a.fridge)
    if a.predicted:
        invs += predicted_inventories(a.predicted)
    invs += random_inventories(a.random, df, a.seed, a.min_recipes, a.expiry, use_staples)
    if not invs:
        ap.error("give --fridge, --predicted and/or --random N")

    rows = []
    for inv in invs:
        for m in a.methods.split(","):
            if m == "llm_only":
                res = llm_only(inv["have"], a.k, a.diet, a.expiry, use_staples)
            else:
                ranked = rx.search(inv["have"], diet=a.diet, top_k=a.k, method=m, df=df, alpha=a.alpha, beta=a.beta)
                res = score_ranked(ranked, inv["have"], a.diet, a.expiry)
            rows.append({"inventory": inv["id"], "method": m, **res})

    out = path("results") / "retrieval.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"{len(invs)} inventories -> {out}\n")
    for m in a.methods.split(","):
        mr = [r for r in rows if r["method"] == m]
        line = {k: round(statistics.mean(r[k] for r in mr if r.get(k) is not None), 3)
                for k in ("cookable", "cookable_subs", "expiry_use", "diet_violation", "hallucinated")
                if any(r.get(k) is not None for r in mr)}
        print(f"{m:10s} {line}")


if __name__ == "__main__":
    main()
