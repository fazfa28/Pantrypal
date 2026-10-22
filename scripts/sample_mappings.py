"""Draw a random sample of recipe ingredient strings and record how canon.py maps them.

  python -m scripts.sample_mappings                 # writes eval/mapping_check/before.csv
  python -m scripts.sample_mappings --score FILE    # error rates from a labelled sheet
  python -m scripts.sample_mappings --after         # re-run the matcher on the SAME strings -> after.csv

The sample is drawn from ingredient OCCURRENCES in the filtered recipes (same filters as
core.recipes build), so common ingredients are sampled more often and the error rate reflects
what recipe matching will actually see. The seed is fixed so before/after use the same strings.

Label the `verdict` column by hand:
  correct  matched id is the right food for this string ("cilantro" -> coriander)
  wrong    matched id is a different food ("fish sauce" -> fish)
  missed   no id returned, but a suitable id exists in data/aliases.yaml
  none     no id returned and nothing suitable exists in the list (a coverage gap)
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from core.canon import get_canon
from core.config import ROOT, cfg
from core.recipes import iso_minutes, parse_list

OUT = ROOT / "eval" / "mapping_check"


def occurrences(src: Path) -> list[str]:
    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
    df["minutes"] = df["TotalTime"].map(iso_minutes)
    df["names"] = df["RecipeIngredientParts"].map(parse_list)
    df["ReviewCount"] = pd.to_numeric(df["ReviewCount"], errors="coerce").fillna(0)
    df = df[
        df["minutes"].notna() & (df["minutes"] <= cfg("recipes", "max_minutes", default=60))
        & df["names"].map(lambda n: 0 < len(n) <= cfg("recipes", "max_ingredients", default=12))
        & (df["ReviewCount"] >= cfg("recipes", "min_reviews", default=1))
    ].sort_values("ReviewCount", ascending=False).head(cfg("recipes", "limit", default=5000))
    return [n.strip().lower() for names in df["names"] for n in names]


def sample(src: str, n: int, seed: int, out: Path) -> None:
    occ = occurrences(Path(src))
    rng = random.Random(seed)
    strings = list(dict.fromkeys(rng.sample(occ, n * 2)))[:n]   # distinct strings, occurrence-weighted
    canon = get_canon()
    rows = []
    for s in strings:
        m = canon.match(s)
        rows.append({"ingredient": s, "predicted_id": m.id or "", "method": m.method,
                     "score": round(m.score, 2), "verdict": ""})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(occ)} ingredient occurrences; wrote {len(rows)} strings to {out}")
    print("Now fill the `verdict` column (correct / wrong / missed / none); see this file's docstring.")


def after(before: Path, out: Path) -> None:
    """Re-run the current matcher on the strings in before.csv. Rows whose predicted id is unchanged keep
    their old verdict; changed rows are left blank for relabelling (same rule as before)."""
    canon = get_canon()
    # Keep verdicts already given in an existing after.csv when the prediction has not changed since.
    done = {}
    if out.exists():
        done = {(r["ingredient"], r["predicted_id"]): (r["verdict"], r.get("note", ""))
                for r in csv.DictReader(open(out, encoding="utf-8")) if r.get("verdict")}
    rows, changed = [], 0
    for r in csv.DictReader(open(before, encoding="utf-8")):
        m = canon.match(r["ingredient"])
        new = m.id or ""
        same = new == r["predicted_id"]
        changed += not same
        verdict, note = (r["verdict"], "") if same else done.get((r["ingredient"], new), ("", ""))
        rows.append({"ingredient": r["ingredient"], "before_id": r["predicted_id"], "before_verdict": r["verdict"],
                     "predicted_id": new, "method": m.method, "score": round(m.score, 2),
                     "verdict": verdict, "note": note})
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    blank = sum(not r["verdict"] for r in rows)
    print(f"{changed} of {len(rows)} predictions changed; {blank} need a verdict; wrote {out}")
    print("Relabel the blank `verdict` rows (changed predictions), then run --score on after.csv.")


def score(file: str) -> None:
    rows = list(csv.DictReader(open(file, encoding="utf-8")))
    c = Counter(r["verdict"].strip().lower() for r in rows)
    unlabeled = c.pop("", 0)
    n = sum(c.values())
    matched = c["correct"] + c["wrong"]
    print(f"{n} labelled ({unlabeled} still blank): {dict(c)}")
    if n:
        print(f"coverage (matched / all)      : {matched / n:.1%}")
        print(f"wrong rate (wrong / all)       : {c['wrong'] / n:.1%}")
        if matched:
            print(f"precision (correct / matched)  : {c['correct'] / matched:.1%}")
        print(f"fully right (correct+none / all): {(c['correct'] + c['none']) / n:.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(ROOT / "data" / "raw" / "recipes.csv"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "before.csv"))
    ap.add_argument("--score", help="labelled CSV to compute error rates from")
    ap.add_argument("--after", action="store_true", help="re-run the matcher on before.csv's strings")
    a = ap.parse_args()
    if a.score:
        score(a.score)
    elif a.after:
        after(OUT / "before.csv", OUT / "after.csv")
    else:
        sample(a.src, a.n, a.seed, Path(a.out))


if __name__ == "__main__":
    main()
