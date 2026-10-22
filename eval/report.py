"""Aggregate Experiment 1 runs into one table, per-class F1 and a confusion list.

  python -m eval.report                      # all eval/results/freiburg__*__closed__test.jsonl
  python -m eval.report --best glm-5.3-flash # confusion pairs for one model
Writes eval/results/report.md. Everything is recomputed from the saved JSONL, so it costs nothing.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

from core.config import path
from eval import metrics
from eval.compare import load, paired_hits


def model_name(f: Path) -> str:
    return f.name.split("__")[1]


def stats(rows: list[dict]) -> dict:
    y_true = [next(iter(r["true"])) for r in rows]
    y_pred = [r["top1"] for r in rows]
    ci = metrics.bootstrap_ci(list(zip(y_true, y_pred)), lambda x: metrics.accuracy(*zip(*x)))
    costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    f1 = metrics.macro_f1(y_true, y_pred)
    per_img = sum(costs) / len(costs) if costs else None
    return {
        "n": len(rows),
        "acc": metrics.accuracy(y_true, y_pred), "ci": ci, "f1": f1,
        "json": sum(r["valid_json"] for r in rows) / len(rows),
        "usd_img": per_img, "usd_1k": per_img * 1000 if per_img else None,
        "f1_per_usd": f1 / (per_img * 1000) if per_img else None,   # macro-F1 per US$ spent on 1,000 images
        "tokens": statistics.mean(sum(r["usage"].values()) for r in rows),
        "lat": statistics.median(r["latency_s"] for r in rows),
        "y_true": y_true, "y_pred": y_pred,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="freiburg__*__closed__test.jsonl")
    ap.add_argument("--best", help="model key for the confusion list (default: highest macro-F1)")
    a = ap.parse_args()

    files = sorted(path("results").glob(a.pattern))
    runs = {model_name(f): load(str(f)) for f in files}
    S = {m: stats(list(r.values())) for m, r in runs.items()}
    order = sorted(S, key=lambda m: -S[m]["f1"])
    best = a.best or order[0]

    out = ["## Experiment 1: Freiburg, closed-set, test split\n",
           "| Model | n | Top-1 acc (95% CI) | Macro-F1 | Valid JSON | US$/1k img | Macro-F1 per US$ | Tokens/img | Median latency (s) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        s = S[m]
        usd = f"{s['usd_1k']:.3f}" if s["usd_1k"] else "n/a"
        fpd = f"{s['f1_per_usd']:.1f}" if s["f1_per_usd"] else "n/a"
        out.append(f"| {m} | {s['n']} | {s['acc']:.3f} ({s['ci'][0]:.3f}-{s['ci'][1]:.3f}) | {s['f1']:.3f} | "
                   f"{s['json']:.1%} | {usd} | {fpd} | {s['tokens']:.0f} | {s['lat']:.2f} |")

    out += ["\n## Paired McNemar tests (top-1 correct, same images)\n", "| A | B | A acc | B acc | p |", "|---|---|---|---|---|"]
    for i, m1 in enumerate(order):
        for m2 in order[i + 1:i + 2]:   # adjacent models in the ranking
            ha, hb = paired_hits(runs[m1], runs[m2])
            r = metrics.mcnemar(ha, hb)
            p = r.get("p_value", r.get("p"))
            out.append(f"| {m1} | {m2} | {sum(ha)/len(ha):.3f} | {sum(hb)/len(hb):.3f} | {p} |")

    s = S[best]
    f1c = metrics.per_class_f1(s["y_true"], s["y_pred"])
    out += [f"\n## Per-class F1, {best}\n", "| Class | F1 |", "|---|---|"]
    for c, v in sorted(f1c.items(), key=lambda kv: (kv[1] if isinstance(kv[1], float) else kv[1].get("f1", 0))):
        out.append(f"| {c} | {(v if isinstance(v, float) else v.get('f1', 0)):.3f} |")

    conf = collections.Counter((t, p) for t, p in zip(s["y_true"], s["y_pred"]) if t != p)
    out += [f"\n## Most common confusions, {best} (true -> predicted)\n", "| True | Predicted | Count |", "|---|---|---|"]
    out += [f"| {t} | {p} | {n} |" for (t, p), n in conf.most_common(15)]

    text = "\n".join(out)
    (path("results") / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
