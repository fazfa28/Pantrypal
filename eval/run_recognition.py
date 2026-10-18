"""Experiment 1: recognition accuracy and cost.

Examples:
  # 50 test images, one model
  python -m eval.run_recognition eval/manifests/freiburg.csv --model qwen3-vl-8b --limit 50
  # fridge scenes, first photo only vs all three photos
  python -m eval.run_recognition eval/manifests/fridge.csv --model qwen3-vl-8b --photos first
  python -m eval.run_recognition eval/manifests/fridge.csv --model qwen3-vl-8b --photos all

Every model reply is cached (eval/cache/), and finished rows are skipped on rerun,
so you can stop and resume at any time. Results go to eval/results/.
"""
from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from core.config import ROOT, cfg, path
from core.llm import cost_usd
from core.vlm import _norm, closed_prompt, recognise
from eval import metrics
from eval.datasets import label_mapper, load_manifest


def resolve_model(name: str) -> dict:
    """A key from config.yaml eval_models, or a raw model id."""
    m = (cfg("eval_models", default={}) or {}).get(name)
    return {"key": name, **m} if m else {"key": name.replace("/", "_"), "id": name, "price": None}


def run_row(row: dict, model: dict, photos: str, to_id, closed: tuple | None = None) -> dict:
    files = row["photos"][:1] if photos == "first" else row["photos"]
    files = [f if Path(f).is_absolute() else str(ROOT / f) for f in files]
    prompt_text, label_ids = closed or (None, None)
    out = recognise(files, model=model["id"], use_cache=True,
                    base_url=model.get("base_url"), api_key_env=model.get("api_key_env"),
                    prompt_text=prompt_text, label_ids=label_ids)
    pred = [{"id": it.canonical_id, "name": it.name, "qty": it.quantity, "confidence": it.confidence}
            for it in out.items]
    return {
        "id": row["id"],
        "true": {to_id(k): v for k, v in row["labels"].items()},
        "pred": pred,
        "top1": pred[0]["id"] if pred else None,
        "valid_json": out.valid_json,
        "usage": out.usage,
        "cost_usd": cost_usd(out.usage, model.get("price")),
        "latency_s": out.latency_s,
    }


def summarize(records: list[dict]) -> dict:
    s = {"n": len(records)}
    if not records:
        return s
    s["valid_json_rate"] = sum(r["valid_json"] for r in records) / len(records)
    costs = [r["cost_usd"] for r in records if r["cost_usd"] is not None]
    s["cost_usd_total"] = round(sum(costs), 4) if costs else None
    s["cost_usd_per_image"] = round(sum(costs) / len(costs), 6) if costs else None
    s["tokens_per_image"] = statistics.mean(sum(r["usage"].values()) for r in records)
    s["median_latency_s"] = statistics.median(r["latency_s"] for r in records)

    single = [r for r in records if len(r["true"]) == 1]
    if single:  # classification datasets: one class per image
        y_true = [next(iter(r["true"])) for r in single]
        y_pred = [r["top1"] for r in single]
        s["top1_accuracy"] = metrics.accuracy(y_true, y_pred)
        s["top1_accuracy_ci95"] = metrics.bootstrap_ci(
            list(zip(y_true, y_pred)), lambda rows: metrics.accuracy(*zip(*rows)))
        s["macro_f1"] = metrics.macro_f1(y_true, y_pred)
        s["hit_rate"] = sum(t in {p["id"] for p in r["pred"]} for t, r in zip(y_true, single)) / len(single)

    multi = [r for r in records if len(r["true"]) > 1]
    if multi:  # scenes with several items (fridge set, Freiburg clutter)
        true_sets = [set(r["true"]) for r in multi]
        pred_sets = [{p["id"] for p in r["pred"] if p["id"]} for r in multi]
        s["scene_prf"] = metrics.set_prf(true_sets, pred_sets)
        s["scene_recall_ci95"] = metrics.bootstrap_ci(
            list(zip(true_sets, pred_sets)), lambda rows: metrics.set_prf(*zip(*rows))["recall"])
        s["count_mae"] = metrics.count_mae(
            [r["true"] for r in multi], [{p["id"]: p["qty"] for p in r["pred"]} for r in multi])
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--model", required=True, help="key in config.yaml eval_models, or a raw model id")
    ap.add_argument("--dataset", help="label map name (default: manifest file name)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--photos", choices=["all", "first"], default="all")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--closed-set", action="store_true",
                    help="ask the model to pick from the dataset's classes; names map straight to the label map")
    a = ap.parse_args()

    dataset = a.dataset or Path(a.manifest).stem
    model = resolve_model(a.model)
    to_id = label_mapper(dataset)
    rows = load_manifest(a.manifest, a.split)[: a.limit]

    closed = None
    if a.closed_set:
        classes = sorted({k for r in load_manifest(a.manifest) for k in r["labels"]})
        names = [c.lower().replace("_", " ") for c in classes]
        closed = (closed_prompt(names), {_norm(c): to_id(c) for c in classes})
        print(f"closed-set mode, {len(classes)} classes: {', '.join(names)}")

    out_dir = path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "__closed" if closed else ""
    out_file = out_dir / f"{dataset}__{model['key']}__{a.photos}{tag}__{a.split}.jsonl"
    done = {}
    if out_file.exists():
        done = {json.loads(l)["id"]: json.loads(l) for l in out_file.read_text().splitlines() if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} rows, {len(done)} already done, running {len(todo)} with {model['id']}")

    with open(out_file, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as pool:
        futures = {pool.submit(run_row, r, model, a.photos, to_id, closed): r["id"] for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception as e:
                print(f"  {futures[fut]} failed: {e}")
                continue
            done[rec["id"]] = rec
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}")

    summary = summarize([done[r["id"]] for r in rows if r["id"] in done])
    summary_file = out_file.with_suffix(".summary.json")
    summary_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
