"""Experiment 2 results: per-arm means with bootstrap 95% CIs, paired tests, judge agreement.

  python -m eval.exp2_stats --split test        # -> eval/results/exp2_report_<split>.md

Everything is recomputed from the saved arm outputs and judge ratings (no API calls).
Per case, each metric is a share of the 5 slots; CIs bootstrap over cases; PantryPal is compared with
each baseline by a paired Wilcoxon signed-rank test over cases.
"""
from __future__ import annotations

import argparse
import json
import statistics

from core import recipes as rx
from core.config import path
from eval import metrics
from eval.judge_recommend import ARMS, arm_scores, load, recipe_key
from eval.run_recommend import auto_metrics

AUTO = ("grounded", "cookable", "diet_ok", "time_ok", "exclude_ok", "use_up", "hallucinated", "filters_correct")
JUDGED = ("fit", "sensible", "useful", "any_useful")
LABEL = {"grounded": "Grounded (real recipe)", "cookable": "Cookable@5", "diet_ok": "Diet respected",
         "time_ok": "Time respected", "exclude_ok": "Exclusion respected", "use_up": "Use-up item used",
         "hallucinated": "Hallucinated pantry items", "filters_correct": "Filters extracted correctly",
         "fit": "Preference fit (1-5)", "sensible": "Sensible dish", "useful": "useful@5",
         "any_useful": "Cases with >=1 useful recipe"}


def judge_file(name: str, split: str) -> dict:
    f = path("results") / f"exp2_judge__{name}__{split}.jsonl"
    return {r["id"]: r for r in map(json.loads, f.read_text().splitlines()) if r} if f.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    cases, runs = load(a.split)
    db_names = set(rx.load_index()["name"].str.strip().str.lower())
    primary, secondary = judge_file("primary", a.split), judge_file("secondary", a.split)
    ids = sorted(c for c in cases if all(c in runs[arm] for arm in ARMS) and c in primary)

    per = {arm: {} for arm in ARMS}   # arm -> metric -> {case: value}
    for cid in ids:
        js = arm_scores(cid, runs, primary[cid]["ratings"], cases[cid])
        for arm in ARMS:
            m = auto_metrics(cases[cid], runs[arm][cid], db_names)
            for k in AUTO:
                if k in m:
                    per[arm].setdefault(k, {})[cid] = m[k]
            for k in JUDGED:
                per[arm].setdefault(k, {})[cid] = js[arm][k]
    cost = {arm: statistics.mean(runs[arm][c]["cost_usd"] or 0 for c in ids) for arm in ARMS}
    lat = {arm: statistics.median(runs[arm][c]["latency_s"] for c in ids) for arm in ARMS}

    def cell(arm, k):
        v = per[arm].get(k)
        if not v:
            return "n/a"
        vals = list(v.values())
        if k == "any_useful":
            return f"{sum(vals):.0f}/{len(vals)}"
        lo, hi = metrics.bootstrap_ci(vals, statistics.mean)
        fmt = "{:.2f}"
        return f"{fmt.format(statistics.mean(vals))} ({fmt.format(lo)}-{fmt.format(hi)})"

    lines = [f"## Experiment 2 results, {a.split} split ({len(ids)} cases)", "",
             "Mean per case (bootstrap 95% CI). Shares are of the 5 recommendation slots.", "",
             "| Metric | n cases | " + " | ".join(ARMS) + " |", "|---|---|" + "---|" * len(ARMS)]
    for k in AUTO + JUDGED:
        n = max(len(per[arm].get(k, {})) for arm in ARMS)
        if n:
            lines.append(f"| {LABEL[k]} | {n} | " + " | ".join(cell(arm, k) for arm in ARMS) + " |")
    lines.append("| US$ per request (arm only) | | " + " | ".join(f"{cost[arm]:.5f}" for arm in ARMS) + " |")
    lines.append("| Median latency (s) | | " + " | ".join(f"{lat[arm]:.1f}" for arm in ARMS) + " |")

    from scipy.stats import wilcoxon
    lines += ["", "### Paired tests: PantryPal vs each baseline (Wilcoxon signed-rank over cases)", "",
              "| Metric | vs | PantryPal mean | Baseline mean | p |", "|---|---|---|---|---|"]
    for k in ("cookable", "diet_ok", "fit", "useful"):
        for base in ("llm_only", "rag"):
            common = sorted(set(per["pantrypal"].get(k, {})) & set(per[base].get(k, {})))
            x = [per["pantrypal"][k][c] for c in common]
            y = [per[base][k][c] for c in common]
            if len(common) < 5:
                continue
            diffs = [i - j for i, j in zip(x, y)]
            p = 1.0 if all(d == 0 for d in diffs) else wilcoxon(x, y, zero_method="wilcox").pvalue
            lines.append(f"| {LABEL[k]} | {base} | {statistics.mean(x):.2f} | {statistics.mean(y):.2f} | {p:.2g} |")

    if secondary:
        from scipy.stats import spearmanr
        from sklearn.metrics import cohen_kappa_score
        pairs = [(primary[c]["ratings"][k]["fit"], secondary[c]["ratings"][k]["fit"])
                 for c in secondary if c in primary for k in secondary[c]["ratings"] if k in primary[c]["ratings"]]
        if len(pairs) > 2:
            p1, p2 = zip(*pairs)
            lines += ["", "### Judge agreement (preference fit, per recipe)", "",
                      f"{len(pairs)} recipes in {len(secondary)} cases rated by both judges: "
                      f"Spearman {spearmanr(p1, p2).statistic:.2f}, quadratic-weighted kappa "
                      f"{cohen_kappa_score(p1, p2, weights='quadratic'):.2f}, exact agreement "
                      f"{sum(i == j for i, j in pairs) / len(pairs):.0%}, within 1 point "
                      f"{sum(abs(i - j) <= 1 for i, j in pairs) / len(pairs):.0%}."]
        sec_scores = {arm: [] for arm in ARMS}
        for c in secondary:
            for arm, s in arm_scores(c, runs, secondary[c]["ratings"], cases[c]).items():
                sec_scores[arm].append(s["fit"])
        lines.append("Mean fit by the second judge on its cases: " + ", ".join(
            f"{arm} {statistics.mean(v):.2f}" for arm, v in sec_scores.items() if v) + ".")

    judge_cost = sum((primary[c].get("cost_usd") or 0) for c in ids) + sum((r.get("cost_usd") or 0) for r in secondary.values())
    arm_cost = sum(runs[arm][c]["cost_usd"] or 0 for arm in ARMS for c in ids)
    lines += ["", f"Total API cost of this split: arms US${arm_cost:.3f}, judges US${judge_cost:.3f}."]

    text = "\n".join(lines)
    out = path("results") / f"exp2_report_{a.split}.md"
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
