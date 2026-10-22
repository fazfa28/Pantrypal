"""Experiment 2: LLM-judge scores for the recommended recipes, plus the human check.

  python -m eval.judge_recommend --split dev                       # primary judge (GPT-6.1 Sol), all cases
  python -m eval.judge_recommend --split test --judge secondary --subset 0.2   # second judge on 20% of cases
  python -m eval.judge_recommend --split test --export-human 40     # blind sheet for teammates
  python -m eval.judge_recommend --split test --agreement eval/cases/human_ratings.csv

One judge call per case: every recipe any arm recommended for that case is pooled, de-duplicated and
shuffled, so the judge cannot tell which arm produced it and rates all arms on one scale.
Per recipe:
  fit       1-5, how well it matches what the user asked for (kind of meal, preference, spirit of the request).
            Diet, time, exclusions and pantry coverage are NOT part of fit: code checks those.
  sensible  yes/no, a coherent dish that these ingredients can make (catches odd invented recipes).
Per arm and case: mean fit over the 5 slots (a missing pick scores 1), share of sensible picks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics

from core.config import ROOT, cfg, path
from core.llm import chat, cost_usd
from eval.run_recommend import ALWAYS, CASES, K, name, parse_json, pick_checks

ARMS = ("llm_only", "rag", "pantrypal")

JUDGE_PROMPT = """You are evaluating recipe recommendations for a cooking app.

User request: "{request}"
User's pantry: {pantry}. {always}

Below are recipes suggested by different systems, in random order. Rate EACH recipe on its own.

fit (1-5): how well the recipe satisfies what the user asked for: the kind of meal, any stated preference (cuisine, mood, flavour, kid-friendly...), and the overall spirit of the request.
  5 = clearly what the user wanted; 4 = good, minor mismatch; 3 = partly fits; 2 = weak fit; 1 = does not fit (wrong kind of meal, or ignores the preference).
  Do NOT judge diet, time limit, excluded ingredients or whether the pantry has the ingredients; those are checked separately.
sensible (true/false): true if this is a coherent dish that the listed ingredients can make; false if the combination is strange or the ingredients cannot make the named dish.

Recipes:
{recipes}

Return JSON only, one entry per recipe key:
{{"ratings": [{{"key": "<key>", "fit": <1-5>, "sensible": <true|false>}}]}}"""


def recipe_key(p: dict) -> str:
    """Same recipe from two arms -> same key (judged once)."""
    raw = str(p["recipe_id"]) if p.get("recipe_id") is not None else p["name"].strip().lower() + "|" + "|".join(sorted(p["ingredients"]))
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def load(split: str):
    cases = {c["id"]: c for c in map(json.loads, open(CASES, encoding="utf-8")) if c["split"] == split}
    runs = {}
    for arm in ARMS:
        f = path("results") / f"exp2__{arm}__{split}.jsonl"
        runs[arm] = {r["id"]: r for r in map(json.loads, f.read_text().splitlines()) if r} if f.exists() else {}
    return cases, runs


def pooled(case_id: str, runs: dict) -> dict[str, dict]:
    items = {}
    for arm in ARMS:
        for p in runs[arm].get(case_id, {}).get("picks", []):
            items.setdefault(recipe_key(p), p)
    return items


def judge_case(case: dict, items: dict[str, dict], model: str, seed: int) -> tuple[dict, dict]:
    keys = sorted(items)
    random.Random(f"{seed}-{case['id']}").shuffle(keys)
    block = "\n".join(json.dumps({"key": k, "name": items[k]["name"], "minutes": items[k]["minutes"],
                                  "ingredients": items[k]["ingredients"]}) for k in keys)
    prompt = JUDGE_PROMPT.format(request=case["request"], pantry=", ".join(map(name, case["pantry"])),
                                 always=ALWAYS, recipes=block)
    messages = [{"role": "user", "content": prompt}]
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for _ in range(2):
        out = chat(messages, model=model, use_cache=True)
        for k in usage:
            usage[k] += out["usage"][k]
        try:
            ratings = {r["key"]: {"fit": int(r["fit"]), "sensible": bool(r["sensible"])}
                       for r in parse_json(out["content"])["ratings"] if r.get("key") in items}
            return ratings, usage
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            messages += [{"role": "assistant", "content": out["content"]},
                         {"role": "user", "content": "Not valid JSON in the required shape. Reply with the JSON object only."}]
    return {}, usage


def arm_scores(case_id: str, runs: dict, ratings: dict, case: dict | None = None) -> dict:
    """fit: mean over K slots (missing = 1); sensible: share of K; useful: share of K slots that are
    cookable AND respect every hard constraint AND have judge fit >= 4 (same checks as run_recommend)."""
    out = {}
    for arm in ARMS:
        picks = runs[arm].get(case_id, {}).get("picks", [])
        if case_id not in runs[arm]:
            continue
        fits = [ratings.get(recipe_key(p), {}).get("fit") for p in picks]
        fits = [f for f in fits if f is not None] + [1] * (K - len(picks))   # missing pick = 1
        sens = [ratings.get(recipe_key(p), {}).get("sensible", False) for p in picks]
        out[arm] = {"fit": statistics.mean(fits) if fits else None, "sensible": sum(sens) / K}
        if case is not None:
            useful = [all(pick_checks(case, p).values()) and ratings.get(recipe_key(p), {}).get("fit", 0) >= 4
                      for p in picks]
            out[arm]["useful"] = sum(useful) / K
            out[arm]["any_useful"] = float(any(useful))
    return out


def export_human(cases: dict, runs: dict, n: int, seed: int, out_dir) -> None:
    """Blind sheet: request + one recipe per row, balanced across arms; the key file says which arm."""
    rng = random.Random(seed)
    rows = []
    for arm in ARMS:
        pool = [(cid, p) for cid, r in runs[arm].items() if cid in cases for p in r["picks"]]
        for cid, p in rng.sample(pool, min(len(pool), n // len(ARMS) + (1 if arm == ARMS[0] else 0))):
            rows.append({"case": cid, "arm": arm, "key": recipe_key(p), "request": cases[cid]["request"],
                         "recipe": p["name"], "minutes": p["minutes"], "ingredients": ", ".join(p["ingredients"])})
    rng.shuffle(rows)
    rows = rows[:n]
    with open(out_dir / "human_ratings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "request", "recipe", "minutes", "ingredients", "fit_rater1", "fit_rater2", "fit_rater3",
                    "sensible_rater1", "sensible_rater2", "sensible_rater3"])
        for i, r in enumerate(rows):
            w.writerow([i, r["request"], r["recipe"], r["minutes"], r["ingredients"], "", "", "", "", "", ""])
    (out_dir / "human_ratings_key.json").write_text(json.dumps(
        [{"item": i, "case": r["case"], "arm": r["arm"], "key": r["key"]} for i, r in enumerate(rows)], indent=1))
    print(f"wrote {out_dir / 'human_ratings.csv'} ({len(rows)} items) and the hidden key human_ratings_key.json")
    print("Raters fill fit (1-5) and sensible (yes/no) using the rubric in this script's JUDGE_PROMPT.")


def agreement(human_csv: str, split: str) -> None:
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score

    key = {k["item"]: k for k in json.loads((ROOT / "eval" / "cases" / "human_ratings_key.json").read_text())}
    judged = {}
    for jf in path("results").glob(f"exp2_judge__*__{split}.jsonl"):
        jname = jf.name.split("__")[1]
        for r in map(json.loads, jf.read_text().splitlines()):
            for k, v in r["ratings"].items():
                judged.setdefault(jname, {})[(r["id"], k)] = v["fit"]
    rows = list(csv.DictReader(open(human_csv, encoding="utf-8")))
    raters = [c for c in rows[0] if c.startswith("fit_rater") and any(r[c].strip() for r in rows)]
    series = {rt: [] for rt in raters}
    items = []
    for r in rows:
        if all(r[rt].strip() for rt in raters):
            items.append(key[int(r["item"])])
            for rt in raters:
                series[rt].append(int(r[rt]))
    human_mean = [statistics.mean(v) for v in zip(*series.values())]
    print(f"{len(items)} items rated by all of {raters}")
    for i, a in enumerate(raters):
        for b in raters[i + 1:]:
            print(f"  {a} vs {b}: weighted kappa {cohen_kappa_score(series[a], series[b], weights='quadratic'):.2f}")
    for jname, scores in judged.items():
        pairs = [(h, scores.get((it["case"], it["key"]))) for h, it in zip(human_mean, items)]
        pairs = [(h, j) for h, j in pairs if j is not None]
        if len(pairs) > 2:
            rho = spearmanr(*zip(*pairs)).statistic
            print(f"  judge {jname} vs mean human: Spearman {rho:.2f} on {len(pairs)} items")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--judge", default="primary", help="key under judges: in config.yaml")
    ap.add_argument("--subset", type=float, default=1.0, help="share of cases to judge (fixed seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--export-human", type=int, metavar="N")
    ap.add_argument("--agreement", metavar="CSV")
    a = ap.parse_args()

    cases, runs = load(a.split)
    if a.export_human:
        return export_human(cases, runs, a.export_human, a.seed, ROOT / "eval" / "cases")
    if a.agreement:
        return agreement(a.agreement, a.split)

    j = cfg("judges", a.judge)
    ids = sorted(cid for cid in cases if all(cid in runs[arm] for arm in ARMS))
    if a.subset < 1:
        ids = sorted(random.Random(a.seed).sample(ids, max(1, round(len(ids) * a.subset))))
    out_file = path("results") / f"exp2_judge__{a.judge}__{a.split}.jsonl"
    done = {r["id"]: r for r in map(json.loads, out_file.read_text().splitlines()) if r} if out_file.exists() else {}
    todo = [cid for cid in ids if cid not in done]
    print(f"judge {j['id']}: {len(ids)} cases, {len(done)} done, running {len(todo)}")

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(a.workers) as pool, open(out_file, "a", encoding="utf-8") as f:
        for cid, (ratings, usage) in zip(todo, pool.map(lambda c: judge_case(cases[c], pooled(c, runs), j["id"], a.seed), todo)):
            rec = {"id": cid, "judge": j["id"], "ratings": ratings, "usage": usage, "cost_usd": cost_usd(usage, j.get("price"))}
            done[cid] = rec
            f.write(json.dumps(rec) + "\n")

    per_arm = {arm: {"fit": [], "sensible": [], "useful": [], "any_useful": []} for arm in ARMS}
    for cid in ids:
        for arm, s in arm_scores(cid, runs, done[cid]["ratings"], cases[cid]).items():
            for k in per_arm[arm]:
                if s.get(k) is not None:
                    per_arm[arm][k].append(s[k])
    cost = sum(done[c]["cost_usd"] or 0 for c in ids)
    print(f"\nJudge scores, {a.split} split, {len(ids)} cases (fit 1-5; sensible = share of {K} slots). Judge cost US${cost:.3f}")
    for arm, v in per_arm.items():
        if v["fit"]:
            print(f"  {arm:10s} fit {statistics.mean(v['fit']):.2f}   sensible {statistics.mean(v['sensible']):.2f}   "
                  f"useful@5 {statistics.mean(v['useful']):.2f}   cases with >=1 useful {sum(v['any_useful']):.0f}/{len(v['any_useful'])}")


if __name__ == "__main__":
    main()
