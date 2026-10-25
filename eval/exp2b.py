"""Experiment 2b: follow-up to Experiment 2 (plan: docs/experiment2b_plan.md).

  python -m eval.exp2b arms  --split test     # rag_pantry, pantrypal_noai, pantrypal_labels
  python -m eval.exp2b steps --split test     # steps for LLM-only recipes (GLM, as in the app)
  python -m eval.exp2b audit --split test     # GPT-6.1 Sol: essential ingredients missing from LLM-only lists
  python -m eval.exp2b judge --split test     # fit, workable, diet/avoid double-check (all six methods)
  python -m eval.exp2b judge --split test --judge secondary --subset 0.2
  python -m eval.exp2b stats --split test     # -> eval/results/exp2b_report_<split>.md + per-recipe csv

Reuses the frozen Experiment 2 code and outputs without changing them. Every step writes a jsonl
(finished cases are skipped on rerun) and model replies are cached, so reruns cost nothing.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from core import recipes as rx
from core.canon import get_canon
from core.config import cfg, path
from core.llm import chat, cost_usd
from core.recommend import write_steps
from eval import metrics
from eval.judge_recommend import recipe_key
from eval.run_recommend import (ALWAYS, CASES, K, N_CANDIDATES, SELECT_PROMPT, ask_json, auto_metrics, canon_ids,
                                name, parse_json, pick_checks, pick_record, view)

ARMS = ("llm_only", "rag", "rag_pantry", "pantrypal", "pantrypal_noai", "pantrypal_labels")
NEW_ARMS = ("rag_pantry", "pantrypal_noai", "pantrypal_labels")
RAG_POOL = 100          # rag_pantry: similar recipes retrieved before re-ranking by coverage
MAX_STEPS = 12          # database steps shown to the judge
MODEL = cfg("eval_models")["glm-5.3-flash"]

LABEL_PROMPT = SELECT_PROMPT.replace(
    "prefer recipes the pantry can cover,",
    "prefer recipes the pantry can cover (each candidate lists what is missing_from_pantry; prefer those with "
    "nothing or little missing),")
assert LABEL_PROMPT != SELECT_PROMPT

AUDIT_PROMPT = """You are checking AI-written recipes for completeness. For each recipe below, list the essential ingredients a cook would need to make the dish as named that are NOT in its ingredient list. Ignore salt, pepper, oil, water and sugar, and optional garnishes. Use short generic ingredient names. Use an empty list if nothing essential is missing.

Recipes:
{recipes}

Return JSON only, no other text:
{{"recipes": [{{"key": "<key>", "missing": ["<ingredient>", ...]}}]}}"""

JUDGE_PROMPT = """You are evaluating recipe recommendations for a cooking app.

User request: "{request}"
User's pantry: {pantry}. {always}

Below are recipes suggested by different systems, in random order. Rate EACH recipe on its own.

fit (1-5): how well the recipe satisfies what the user asked for: the kind of meal, any stated preference (cuisine, mood, flavour, kid-friendly...), and the overall spirit of the request.
  5 = clearly what the user wanted; 4 = good, minor mismatch; 3 = partly fits; 2 = weak fit; 1 = does not fit (wrong kind of meal, or ignores the preference).
  Do NOT judge diet, time limit, excluded ingredients or whether the pantry has the ingredients for fit; those are checked separately.
workable (1-5): would this recipe work as written: are the ingredients sufficient for the dish, and do the steps produce it? Judge whether it would work, not how well it is written; short steps are fine if they are correct.
  5 = clearly works; 4 = works, minor gaps; 3 = probably works but something important is unclear or missing; 2 = likely to fail or missing key ingredients or steps; 1 = would not work.{extra}

Recipes:
{recipes}

Return JSON only, one entry per recipe key:
{{"ratings": [{{"key": "<key>", "fit": <1-5>, "workable": <1-5>{extra_fields}}}]}}"""


# ---------- data ----------


def load_cases(split: str) -> dict:
    return {c["id"]: c for c in map(json.loads, open(CASES, encoding="utf-8")) if c["split"] == split}


def read_jsonl(file) -> dict:
    return {r["id"]: r for r in map(json.loads, file.read_text().splitlines()) if r} if file.exists() else {}


def arm_file(arm: str, split: str):
    old = arm not in NEW_ARMS
    return path("results") / (f"exp2__{arm}__{split}.jsonl" if old else f"exp2b__{arm}__{split}.jsonl")


def load_runs(split: str) -> dict:
    return {arm: read_jsonl(arm_file(arm, split)) for arm in ARMS}


def run_parallel(todo: list, fn, out_file, workers: int) -> dict:
    done = read_jsonl(out_file)
    todo = [c for c in todo if c["id"] not in done]
    with ThreadPoolExecutor(workers) as pool, open(out_file, "a", encoding="utf-8") as f:
        for rec in pool.map(fn, todo):
            if rec:
                done[rec["id"]] = rec
                f.write(json.dumps(rec) + "\n")
                f.flush()
    return done


def robust_chat(messages, model, **kw):
    """chat() already retries briefly; judges also hit per-minute limits, so wait longer between tries."""
    for attempt in range(4):
        try:
            return chat(messages, model=model, use_cache=True, **kw)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(20 * (attempt + 1))


# ---------- new arms ----------


def select_labelled(case: dict, candidates: list[dict], usage: dict) -> tuple[list[dict], dict]:
    """Experiment 2's select(), with missing_from_pantry shown per candidate and the label prompt."""
    have = set(case["pantry"])
    by_id = {int(r["recipe_id"]): r for r in candidates}
    shown = [{**view(r), "missing_from_pantry": sorted(name(i.removeprefix("raw:")) for i in set(r["ingredient_ids"]) - have)}
             for r in candidates]
    prompt = LABEL_PROMPT.format(request=case["request"], pantry=", ".join(map(name, case["pantry"])),
                                 always=ALWAYS, candidates=json.dumps(shown), k=K)
    data, latency, valid = ask_json(prompt, MODEL["id"], usage)
    picks, invalid = [], 0
    for p in (data or {}).get("picks", [])[:K]:
        try:
            rid = int(p.get("id"))
        except (TypeError, ValueError):
            invalid += 1
            continue
        if rid in by_id and rid not in {x["recipe_id"] for x in picks}:
            picks.append(pick_record(by_id[rid], str(p.get("reason", ""))))
        else:
            invalid += 1
    return picks, {"latency_s": latency, "valid_json": valid, "invalid_picks": invalid}


def run_arms(split: str, limit: int | None, workers: int) -> None:
    cases = load_cases(split)
    pp = read_jsonl(arm_file("pantrypal", split))
    df = rx.load_index()
    by_id = {int(r["recipe_id"]): r for r in df.to_dict("records")}
    emb = rx.recipe_embeddings(df)
    todo = [cases[c] for c in sorted(cases) if c in pp][:limit]
    db_names = set(df["name"].str.strip().str.lower())

    def finish(case, arm, rec):
        rec = {"id": case["id"], "arm": arm, "model": MODEL["id"], **rec}
        rec["cost_usd"] = cost_usd(rec["usage"], MODEL["price"])
        rec["metrics"] = auto_metrics(case, rec, db_names)
        return rec

    def rag_pantry(case):
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        query = f"{case['request']} I have: {', '.join(map(name, case['pantry']))}."
        scores = emb.scores(query)
        pool = [df.iloc[i].to_dict() for i in emb.top(query, RAG_POOL)]
        have = set(case["pantry"])
        sim = {int(df.iloc[i]["recipe_id"]): float(scores[i]) for i in emb.top(query, RAG_POOL)}
        cov = lambda r: len(set(r["ingredient_ids"]) & have) / (len(set(r["ingredient_ids"])) or 1)
        pool.sort(key=lambda r: (-cov(r), -sim[int(r["recipe_id"])]))
        cands = pool[:N_CANDIDATES]
        from eval.run_recommend import select
        picks, info = select(case, cands, MODEL["id"], usage)
        return finish(case, "rag_pantry", {"picks": picks, "candidates": [int(r["recipe_id"]) for r in cands],
                                            "usage": usage, **info})

    def noai(case):
        cands = [by_id[i] for i in pp[case["id"]]["candidates"]]
        return finish(case, "pantrypal_noai", {"picks": [pick_record(r) for r in cands[:K]],
                                                "candidates": pp[case["id"]]["candidates"], "filters": pp[case["id"]]["filters"],
                                                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                                                "latency_s": 0.0, "valid_json": True})

    def labels(case):
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        cands = [by_id[i] for i in pp[case["id"]]["candidates"]]
        picks, info = select_labelled(case, cands, usage)
        return finish(case, "pantrypal_labels", {"picks": picks, "candidates": pp[case["id"]]["candidates"],
                                                  "filters": pp[case["id"]]["filters"], "usage": usage, **info})

    for arm, fn in (("rag_pantry", rag_pantry), ("pantrypal_noai", noai), ("pantrypal_labels", labels)):
        done = run_parallel(todo, fn, arm_file(arm, split), workers)
        rows = [done[c["id"]] for c in todo if c["id"] in done]
        print(f"{arm}: {len(rows)} cases | cookable {statistics.mean(r['metrics']['cookable'] for r in rows):.2f} | "
              f"cost US${sum(r['cost_usd'] or 0 for r in rows):.4f}")


# ---------- LLM-only steps and completeness audit ----------


def run_steps(split: str, limit: int | None, workers: int) -> None:
    cases = load_cases(split)
    lo = read_jsonl(arm_file("llm_only", split))
    todo = [cases[c] for c in sorted(cases) if c in lo][:limit]
    fn = lambda case: {"id": case["id"], "steps": [write_steps(p, case["request"]) for p in lo[case["id"]]["picks"]]}
    done = run_parallel(todo, fn, path("results") / f"exp2b_steps__{split}.jsonl", workers)
    empty = sum(not s for r in done.values() for s in r["steps"])
    print(f"steps: {len(done)} cases, {sum(len(r['steps']) for r in done.values())} recipes, {empty} without steps")


def run_audit(split: str, limit: int | None, workers: int) -> None:
    cases = load_cases(split)
    lo = read_jsonl(arm_file("llm_only", split))
    judge = cfg("judges", "primary")
    todo = [cases[c] for c in sorted(cases) if c in lo][:limit]

    def fn(case):
        picks = lo[case["id"]]["picks"]
        block = "\n".join(json.dumps({"key": recipe_key(p), "name": p["name"], "ingredients": p["ingredients"]}) for p in picks)
        out = robust_chat([{"role": "user", "content": AUDIT_PROMPT.format(recipes=block)}], judge["id"])
        try:
            missing = {r["key"]: [str(x) for x in r.get("missing", [])] for r in parse_json(out["content"])["recipes"]}
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            print(f"  {case['id']}: unreadable audit reply")
            return None
        return {"id": case["id"], "missing": missing, "usage": out["usage"], "cost_usd": cost_usd(out["usage"], judge.get("price"))}

    done = run_parallel(todo, fn, path("results") / f"exp2b_audit__{split}.jsonl", workers)
    n = sum(len(v) for r in done.values() for v in r["missing"].values())
    print(f"audit: {len(done)} cases, {n} essential ingredients added, cost US${sum(r['cost_usd'] or 0 for r in done.values()):.3f}")


# ---------- judge ----------


def recipe_steps(p: dict, steps_lo: dict, by_id: dict, case_id: str, rank: int) -> list[str]:
    if p.get("recipe_id") is not None:
        return list(by_id[int(p["recipe_id"])]["instructions"])[:MAX_STEPS]
    return list((steps_lo.get(case_id, {}).get("steps") or [[]] * K)[rank])


def run_judge(split: str, which: str, subset: float, seed: int, limit: int | None, workers: int) -> None:
    cases, runs = load_cases(split), load_runs(split)
    steps_lo = read_jsonl(path("results") / f"exp2b_steps__{split}.jsonl")
    by_id = {int(r["recipe_id"]): r for r in rx.load_index().to_dict("records")}
    judge = cfg("judges", which)
    ids = sorted(c for c in cases if all(c in runs[a] for a in ARMS) and c in steps_lo)
    if subset < 1:
        ids = sorted(random.Random(seed).sample(ids, max(1, round(len(ids) * subset))))
    todo = [cases[c] for c in ids][:limit]

    def fn(case):
        items = {}
        for arm in ARMS:
            for rank, p in enumerate(runs[arm][case["id"]]["picks"]):
                items.setdefault(recipe_key(p), {"name": p["name"], "minutes": p["minutes"], "ingredients": p["ingredients"],
                                                 "steps": recipe_steps(p, steps_lo, by_id, case["id"], rank)})
        keys = sorted(items)
        random.Random(f"{seed}-{case['id']}").shuffle(keys)
        spec, extra, fields = case["spec"], "", ""
        if spec["diet"]:
            extra += f"\ndiet_ok (true/false): true if the recipe is {spec['diet']} (judge from its ingredients, steps and name)."
            fields += ', "diet_ok": <true|false>'
        if spec["exclude"]:
            extra += f"\ncontains_avoided (true/false): true if the recipe contains {name(spec['exclude'])} in any form."
            fields += ', "contains_avoided": <true|false>'
        prompt = JUDGE_PROMPT.format(request=case["request"], pantry=", ".join(map(name, case["pantry"])), always=ALWAYS,
                                     extra=extra, extra_fields=fields,
                                     recipes="\n".join(json.dumps({"key": k, **items[k]}) for k in keys))
        messages, usage = [{"role": "user", "content": prompt}], {"prompt_tokens": 0, "completion_tokens": 0}
        for _ in range(2):
            out = robust_chat(messages, judge["id"])
            for k in usage:
                usage[k] += out["usage"][k]
            try:
                ratings = {r["key"]: {"fit": int(r["fit"]), "workable": int(r["workable"]),
                                      **({"diet_ok": bool(r["diet_ok"])} if "diet_ok" in r else {}),
                                      **({"contains_avoided": bool(r["contains_avoided"])} if "contains_avoided" in r else {})}
                           for r in parse_json(out["content"])["ratings"] if r.get("key") in items}
                break
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                messages += [{"role": "assistant", "content": out["content"]},
                             {"role": "user", "content": "Not valid JSON in the required shape. Reply with the JSON object only."}]
        else:
            print(f"  {case['id']}: unreadable judge reply")
            return None
        missing = set(items) - set(ratings)
        if missing:
            print(f"  {case['id']}: judge skipped {len(missing)} of {len(items)} recipes")
        return {"id": case["id"], "judge": judge["id"], "n_items": len(items), "ratings": ratings, "usage": usage,
                "cost_usd": cost_usd(usage, judge.get("price"))}

    done = run_parallel(todo, fn, path("results") / f"exp2b_judge__{which}__{split}.jsonl", workers)
    print(f"judge {which}: {len(done)} cases, cost US${sum(r['cost_usd'] or 0 for r in done.values()):.3f}")


# ---------- stats ----------


def per_case(case: dict, rec: dict, ratings: dict, audit: dict | None) -> dict:
    """Per case, a share or mean over the K slots (a missing pick fails; fit and workable count as 1)."""
    spec, picks = case["spec"], rec["picks"]
    have = set(case["pantry"])
    m = {}
    if audit is not None:   # llm_only: cookable with the judge-completed ingredient list
        cook = [rx.is_cookable(list(dict.fromkeys(p["ingredient_ids"] + canon_ids(audit.get(recipe_key(p), [])))), have)
                for p in picks]
        m["cookable_self"] = sum(pick_checks(case, p)["cookable"] for p in picks) / K
    else:
        cook = [pick_checks(case, p)["cookable"] for p in picks]
    m["cookable"] = sum(cook) / K
    checks = [pick_checks(case, p) for p in picks]
    for k, field in (("diet_ok", "diet"), ("time_ok", "max_minutes"), ("exclude_ok", "exclude")):
        if spec[field]:
            m[k] = sum(c[k] for c in checks) / K
    if spec["use_up"]:
        m["use_up"] = sum(spec["use_up"] in p["ingredient_ids"] for p in picks) / K
    for k in ("fit", "workable"):
        vals = [ratings.get(recipe_key(p), {}).get(k) for p in picks]
        vals = [v for v in vals if v is not None] + [1] * (K - len(picks))
        m[k] = statistics.mean(vals)
    return m


def run_stats(split: str) -> None:
    from scipy.stats import spearmanr, wilcoxon

    cases, runs = load_cases(split), load_runs(split)
    audit = read_jsonl(path("results") / f"exp2b_audit__{split}.jsonl")
    primary = read_jsonl(path("results") / f"exp2b_judge__primary__{split}.jsonl")
    secondary = read_jsonl(path("results") / f"exp2b_judge__secondary__{split}.jsonl")
    ids = sorted(c for c in primary if c in audit)
    per = {arm: {} for arm in ARMS}
    for cid in ids:
        for arm in ARMS:
            m = per_case(cases[cid], runs[arm][cid], primary[cid]["ratings"],
                         audit[cid]["missing"] if arm == "llm_only" else None)
            for k, v in m.items():
                per[arm].setdefault(k, {})[cid] = v

    label = {"cookable": "Cookable", "cookable_self": "Cookable, own list only (Exp 2 measure)",
             "diet_ok": "Diet respected", "time_ok": "Time respected", "exclude_ok": "Avoided food respected",
             "use_up": "Use-up item used", "fit": "Fit to request (1-5)", "workable": "Workable (1-5)"}
    source = {"llm_only": "AI-written", "rag": "database", "rag_pantry": "database", "pantrypal": "database",
              "pantrypal_noai": "database", "pantrypal_labels": "database"}

    def cell(arm, k):
        v = list(per[arm].get(k, {}).values())
        if not v:
            return "–"
        lo, hi = metrics.bootstrap_ci(v, statistics.mean)
        return f"{statistics.mean(v):.2f} ({lo:.2f}–{hi:.2f})"

    lines = [f"## Experiment 2b results, {split} split ({len(ids)} cases)", "",
             "Mean per case (bootstrap 95% CI). Shares are of the 5 recommendation slots. Plan: docs/experiment2b_plan.md.", "",
             "| Measure | " + " | ".join(ARMS) + " |", "|---|" + "---|" * len(ARMS),
             "| Source | " + " | ".join(source[a] for a in ARMS) + " |"]
    for k in label:
        if any(per[a].get(k) for a in ARMS):
            lines.append(f"| {label[k]} | " + " | ".join(cell(a, k) for a in ARMS) + " |")
    cost = {a: statistics.mean(runs[a][c].get("cost_usd") or 0 for c in ids) for a in ARMS}
    lines.append("| US$ per request (method only) | " + " | ".join(f"{cost[a]:.5f}" for a in ARMS) + " |")

    lines += ["", "### PantryPal vs each other method (paired Wilcoxon over cases)", "",
              "| Measure | vs | PantryPal | Other | p |", "|---|---|---|---|---|"]
    for k in ("cookable", "diet_ok", "time_ok", "exclude_ok", "fit", "workable"):
        for other in ARMS:
            if other == "pantrypal":
                continue
            common = sorted(set(per["pantrypal"].get(k, {})) & set(per[other].get(k, {})))
            if len(common) < 5:
                continue
            x = [per["pantrypal"][k][c] for c in common]
            y = [per[other][k][c] for c in common]
            p = 1.0 if all(i == j for i, j in zip(x, y)) else wilcoxon(x, y, zero_method="wilcox").pvalue
            lines.append(f"| {label[k]} | {other} | {statistics.mean(x):.2f} | {statistics.mean(y):.2f} | {p:.2g} |")

    # Judge vs code on diet and avoided food (all recipes of all methods, once per recipe)
    agree = {"diet": [], "avoid": []}
    for cid in ids:
        c, seen = cases[cid], set()
        for arm in ARMS:
            for p in runs[arm][cid]["picks"]:
                k = recipe_key(p)
                if k in seen or k not in primary[cid]["ratings"]:
                    continue
                seen.add(k)
                r, chk = primary[cid]["ratings"][k], pick_checks(c, p)
                if "diet_ok" in r and "diet_ok" in chk:
                    agree["diet"].append((chk["diet_ok"], r["diet_ok"], p["name"], cid))
                if "contains_avoided" in r and "exclude_ok" in chk:
                    agree["avoid"].append((chk["exclude_ok"], not r["contains_avoided"], p["name"], cid))
    lines += ["", "### Code checks vs GPT-6.1 Sol (per unique recipe)", ""]
    for what, pairs in agree.items():
        if not pairs:
            continue
        both_ok = sum(a and b for a, b, *_ in pairs)
        both_bad = sum(not a and not b for a, b, *_ in pairs)
        code_only_bad = [(n, c) for a, b, n, c in pairs if not a and b]
        judge_only_bad = [(n, c) for a, b, n, c in pairs if a and not b]
        lines.append(f"- {what}: {len(pairs)} recipes, agreement {(both_ok + both_bad) / len(pairs):.0%} "
                     f"(both ok {both_ok}, both violation {both_bad}); code flags but judge accepts {len(code_only_bad)}; "
                     f"judge flags but code accepts {len(judge_only_bad)}"
                     + (f" (e.g. {'; '.join(f'{n} [{c}]' for n, c in judge_only_bad[:5])})" if judge_only_bad else "") + ".")

    if secondary:
        for k in ("fit", "workable"):
            pairs = [(primary[c]["ratings"][key][k], secondary[c]["ratings"][key][k])
                     for c in secondary if c in primary for key in secondary[c]["ratings"] if key in primary[c]["ratings"]]
            if len(pairs) > 2:
                a, b = zip(*pairs)
                lines.append(f"- second judge (Qwen3-VL 235B), {k}: {len(pairs)} recipes in {len(secondary)} cases, "
                             f"Spearman {spearmanr(a, b).statistic:.2f}, within 1 point {sum(abs(i - j) <= 1 for i, j in pairs) / len(pairs):.0%}.")
        sec = {a: [] for a in ARMS}
        for c in secondary:
            for a in ARMS:
                sec[a].append(per_case(cases[c], runs[a][c], secondary[c]["ratings"], None)["fit"])
        lines.append("- mean fit by the second judge on its cases: " + ", ".join(f"{a} {statistics.mean(v):.2f}" for a, v in sec.items() if v) + ".")

    spend = (sum(runs[a][c].get("cost_usd") or 0 for a in NEW_ARMS for c in ids)
             + sum(r.get("cost_usd") or 0 for r in audit.values())
             + sum(r.get("cost_usd") or 0 for r in primary.values()) + sum(r.get("cost_usd") or 0 for r in secondary.values()))
    lines += ["", f"API cost of Experiment 2b on this split (new methods, audit, judges; steps not tracked): US${spend:.2f}."]

    out_csv = path("results") / f"exp2b_{split}_per_recipe.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "request", "arm", "rank", "recipe", "source", "minutes", "ingredients", "added_by_audit",
                    "cookable", "diet_ok", "time_ok", "exclude_ok", "fit", "workable", "judge_diet_ok", "judge_contains_avoided"])
        for cid in ids:
            c = cases[cid]
            for arm in ARMS:
                for rank, p in enumerate(runs[arm][cid]["picks"], 1):
                    k = recipe_key(p)
                    added = audit[cid]["missing"].get(k, []) if arm == "llm_only" else []
                    cook = rx.is_cookable(list(dict.fromkeys(p["ingredient_ids"] + canon_ids(added))), set(c["pantry"]))
                    chk, r = pick_checks(c, p), primary[cid]["ratings"].get(k, {})
                    w.writerow([cid, c["request"], arm, rank, p["name"], source[arm], p["minutes"], "; ".join(p["ingredients"]),
                                "; ".join(added), cook, chk.get("diet_ok", ""), chk.get("time_ok", ""), chk.get("exclude_ok", ""),
                                r.get("fit", ""), r.get("workable", ""), r.get("diet_ok", ""), r.get("contains_avoided", "")])
    text = "\n".join(lines)
    (path("results") / f"exp2b_report_{split}.md").write_text(text)
    print(text)
    print(f"\nwrote eval/results/exp2b_report_{split}.md and {out_csv.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["arms", "steps", "audit", "judge", "stats"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--judge", default="primary")
    ap.add_argument("--subset", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.step == "arms":
        run_arms(a.split, a.limit, a.workers)
    elif a.step == "steps":
        run_steps(a.split, a.limit, a.workers)
    elif a.step == "audit":
        run_audit(a.split, a.limit, a.workers)
    elif a.step == "judge":
        run_judge(a.split, a.judge, a.subset, a.seed, a.limit, a.workers)
    else:
        run_stats(a.split)


if __name__ == "__main__":
    main()
