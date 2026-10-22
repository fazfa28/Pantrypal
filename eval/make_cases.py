"""Build Experiment 2 test cases: a realistic pantry + a natural-language request with preferences.

  python -m eval.make_cases                 # 60 cases (10 dev, 50 test) -> eval/cases/exp2_cases.jsonl
  python -m eval.make_cases --n 60 --dev 10 --seed 0

Each case is generated from a structured SPEC, which is the ground truth for the automatic checks:
  diet         None | vegetarian | vegan                (hard: checked by code)
  max_minutes  None | 15 | 20 | 30 | 45                 (hard)
  exclude      an ingredient id to avoid, or None       (hard; usually something IN the pantry)
  use_up       a pantry id the user wants to use, or None (scored as use-up rate)
  preference   a soft wish ("something spicy"), or None (scored by the judge)
  servings     1-4                                      (shown in the request; amounts scale, not checked)

Pantry = the non-staple ingredients of 2-3 "anchor" recipes that satisfy the hard constraints (so a good
answer exists) + 3-5 common extras. When a preference has keyword hints, one anchor is chosen to match it.
Request text comes from templates with varied wording (no LLM, so it is free and reproducible).
Also writes eval/cases/exp2_cases_review.csv for a human read-through before any arm runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter

from core import recipes as rx
from core.config import ROOT
from core.rules import staples

OUT = ROOT / "eval" / "cases"

# Soft preferences, with words that suggest a recipe fits (used only to pick one anchor, never for scoring).
PREFERENCES = {
    "something spicy": ["chili", "jalapeno", "cayenne", "hot_sauce", "red_pepper_flakes", "curry", "salsa", "spicy"],
    "comfort food": ["casserole", "mac", "potato", "cheese", "soup", "stew", "meatloaf", "pie"],
    "something light and healthy": ["salad", "spinach", "lettuce", "cucumber", "yogurt", "steamed", "grilled"],
    "an Italian-style dish": ["pasta", "parmesan", "basil", "oregano", "italian", "mozzarella", "pizza"],
    "an Asian-style dish": ["soy_sauce", "ginger", "sesame", "rice_vinegar", "stir", "teriyaki", "noodle"],
    "a Mexican-style dish": ["salsa", "tortilla", "cumin", "taco", "enchilada", "bean", "cilantro"],
    "something high in protein": ["chicken", "beef", "egg", "tofu", "tuna", "salmon", "turkey", "pork"],
    "something kid-friendly": ["cheese", "pizza", "nugget", "mac", "pancake", "muffin", "banana"],
    "something cheesy": ["cheese", "cheddar", "mozzarella", "parmesan", "cream_cheese"],
    "a sweet dessert": ["chocolate", "cookie", "cake", "brownie", "vanilla", "pie", "pudding"],
    "a one-pan meal": ["skillet", "one pan", "sheet pan", "stir", "casserole", "one-pot"],
    "a quick breakfast": ["egg", "pancake", "oat", "toast", "muffin", "breakfast", "banana"],
}
MEAL_FOR = {"a sweet dessert": "dessert", "a quick breakfast": "breakfast"}

# Ingredient categories that plausibly "go off" or that people commonly avoid.
PERISHABLE = {"produce", "fruit", "dairy", "meat", "seafood", "egg"}
AVOIDABLE = PERISHABLE | {"plant_protein"}

OPENERS = ["What can I make for {meal} for {who}?", "Suggest a {meal} for {who}.",
           "I need {meal} ideas for {who}.", "Any {meal} recipes for {who}?"]
DIET = {"vegetarian": ["It has to be vegetarian.", "We're vegetarian."],
        "vegan": ["I'm vegan.", "It must be vegan."]}
TIME = ["I only have {m} minutes.", "Something that takes under {m} minutes.", "Max {m} minutes, please."]
PREF = ["I'm in the mood for {p}.", "Ideally {p}.", "I'd like {p}."]
EXCLUDE = ["No {x}, please.", "I don't like {x}.", "Please avoid {x}."]
USE_UP = ["I'd like to use up the {u} before it goes off.", "Try to use the {u}.", "The {u} needs using soon."]


def name(cid: str) -> str:
    return cid.replace("_", " ")


def who(n: int) -> str:
    return "one person" if n == 1 else f"{n} people"


def fits(rec: dict, spec: dict) -> bool:
    if spec["max_minutes"] and rec["minutes"] > spec["max_minutes"]:
        return False
    if rx.violates_diet(rec, spec["diet"]):
        return False
    return not (spec["exclude"] and spec["exclude"] in rec["ingredient_ids"])


def matches_pref(rec: dict, pref: str) -> bool:
    text = (rec["name"] + " " + " ".join(rec["ingredient_ids"])).lower()
    return any(h in text for h in PREFERENCES[pref])


def make_case(k: int, rng: random.Random, recs: list[dict], common: list[str], cats: dict,
              preference: str | None) -> dict | None:
    spec = {
        "diet": rng.choice([None, None, None, "vegetarian", "vegetarian", "vegan"]),
        "max_minutes": rng.choice([None, None, 15, 20, 30, 45]),
        "exclude": None,
        "use_up": None,
        "preference": preference,
        "servings": rng.randint(1, 4),
    }
    meal = MEAL_FOR.get(spec["preference"], "dinner")
    pool = [r for r in recs if fits(r, spec) and rx.suits_meal(r, meal) and len(r["ingredient_ids"]) <= 8]
    if len(pool) < 20:
        return None
    anchors = []
    if spec["preference"]:
        pref_pool = [r for r in pool if matches_pref(r, spec["preference"])]
        if len(pref_pool) >= 5:
            anchors.append(rng.choice(pref_pool))
    while len(anchors) < rng.choice([2, 3]):
        r = rng.choice(pool)
        if r not in anchors:
            anchors.append(r)
    anchor_ids = list(dict.fromkeys(i for r in anchors for i in r["ingredient_ids"] if not i.startswith("raw:")))
    extras = [i for i in rng.sample(common, 12) if i not in anchor_ids][: rng.randint(3, 5)]
    pantry = anchor_ids + extras

    # Exclusion: something IN the pantry but not needed by the anchors, so avoiding it is a real test.
    avoidable = [i for i in extras if cats.get(i) in AVOIDABLE]
    if rng.random() < 0.4 and avoidable:
        spec["exclude"] = rng.choice(avoidable)
    perishable = [i for i in anchor_ids if cats.get(i) in PERISHABLE]
    if rng.random() < 0.5 and perishable:
        spec["use_up"] = rng.choice(perishable)
    if sum(bool(spec[f]) for f in ("diet", "max_minutes", "exclude", "use_up", "preference")) < 2:
        return None   # every case tests at least two things

    parts = [rng.choice(OPENERS).format(meal=meal, who=who(spec["servings"]))]
    if spec["diet"]:
        parts.append(rng.choice(DIET[spec["diet"]]))
    if spec["max_minutes"]:
        parts.append(rng.choice(TIME).format(m=spec["max_minutes"]))
    if spec["preference"]:
        parts.append(rng.choice(PREF).format(p=spec["preference"]))
    if spec["exclude"]:
        parts.append(rng.choice(EXCLUDE).format(x=name(spec["exclude"])))
    if spec["use_up"]:
        parts.append(rng.choice(USE_UP).format(u=name(spec["use_up"])))

    return {"id": f"case{k:03d}", "request": " ".join(parts), "pantry": sorted(pantry), "spec": spec,
            "meal": meal,
            "anchors": [{"recipe_id": int(r["recipe_id"]), "name": r["name"], "category": r.get("category", "")}
                        for r in anchors]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--dev", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df = rx.load_index()
    recs = df.to_dict("records")
    freq = Counter(i for ids in df["ingredient_ids"] for i in set(ids))
    st = staples()
    common = sorted(i for i, c in freq.items() if c >= 50 and i not in st and not i.startswith("raw:"))

    from core.canon import get_canon
    cats = get_canon().category
    rng = random.Random(a.seed)
    # Preferences rotate evenly (each about n/15 times); about 1 in 5 cases has none.
    prefs = (list(PREFERENCES) + [None, None, None]) * (a.n // 15 + 1)
    rng.shuffle(prefs)
    cases, k = [], 0
    while len(cases) < a.n:
        c = make_case(k, rng, recs, common, cats, prefs[len(cases)])
        k += 1
        if c:
            c["id"] = f"case{len(cases):03d}"
            cases.append(c)
    dev = set(rng.sample(range(len(cases)), a.dev))
    for i, c in enumerate(cases):
        c["split"] = "dev" if i in dev else "test"

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp2_cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    with open(OUT / "exp2_cases_review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "request", "pantry", "diet", "max_minutes", "exclude", "use_up",
                    "preference", "anchors", "reviewer_ok", "reviewer_note"])
        for c in cases:
            s = c["spec"]
            w.writerow([c["id"], c["split"], c["request"], ", ".join(map(name, c["pantry"])), s["diet"] or "",
                        s["max_minutes"] or "", name(s["exclude"]) if s["exclude"] else "",
                        name(s["use_up"]) if s["use_up"] else "", s["preference"] or "",
                        " | ".join(x["name"] for x in c["anchors"]), "", ""])

    count = lambda f: sum(bool(c["spec"][f]) for c in cases)
    print(f"Wrote {len(cases)} cases ({a.dev} dev) to {OUT}/exp2_cases.jsonl and exp2_cases_review.csv")
    print("cases with each field:", {f: count(f) for f in ("diet", "max_minutes", "exclude", "use_up", "preference")})
    print("pantry size: min", min(len(c["pantry"]) for c in cases), "max", max(len(c["pantry"]) for c in cases),
          "mean", round(sum(len(c["pantry"]) for c in cases) / len(cases), 1))


if __name__ == "__main__":
    main()
