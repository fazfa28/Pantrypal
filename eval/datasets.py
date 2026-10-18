"""Build and load evaluation manifests.

A manifest is one CSV per dataset with columns:
  id      unique row id
  photos  ';'-separated image paths (relative to the repo root or absolute)
  labels  ';'-separated labels; for counted scenes use 'label:qty' (e.g. 'milk:1;egg:6')
  split   'dev' (prompt tuning, max 200 rows) or 'test'

Usage:
  python -m eval.datasets freiburg data/raw/freiburg/images
  python -m eval.datasets grocery  data/raw/grocery_store/dataset --split test
The fridge manifest is written by hand; copy eval/manifests/fridge_template.csv.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import yaml

from core.canon import get_canon
from core.config import ROOT, path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MANIFESTS = ROOT / "eval" / "manifests"


def _write(rows: list[dict], name: str, dev_size: int = 200, seed: int = 0) -> Path:
    rng = random.Random(seed)
    dev = set(rng.sample(range(len(rows)), min(dev_size, len(rows))))
    for i, r in enumerate(rows):
        r.setdefault("split", "dev" if i in dev else "test")
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    out = MANIFESTS / f"{name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "photos", "labels", "split"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")
    return out


def make_freiburg(images_dir: str) -> Path:
    """Freiburg layout: images/<CLASS>/<file>.png; the folder name is the label."""
    root = Path(images_dir)
    rows = [{"id": f.stem, "photos": str(f), "labels": f.parent.name}
            for f in sorted(root.rglob("*")) if f.suffix.lower() in IMAGE_EXT]
    return _write(rows, "freiburg")


def make_grocery(dataset_dir: str, split: str = "test") -> Path:
    """Grocery Store layout: <split>.txt lines 'path, fine_id, coarse_id' + classes.csv.
    Labels use the coarse class name (e.g. 'Apple'). Check the column names if this errors."""
    root = Path(dataset_dir)
    import pandas as pd

    classes = pd.read_csv(root / "classes.csv")
    name_col = next(c for c in classes.columns if "coarse" in c.lower() and "name" in c.lower())
    id_col = next(c for c in classes.columns if "coarse" in c.lower() and "id" in c.lower())
    coarse = dict(zip(classes[id_col].astype(int), classes[name_col]))
    rows = []
    for line in (root / f"{split}.txt").read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        rows.append({"id": Path(parts[0]).stem, "photos": str(root / parts[0]), "labels": coarse[int(parts[2])]})
    return _write(rows, "grocery")


def load_manifest(file: str | Path, split: str | None = None) -> list[dict]:
    with open(file, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if split:
        rows = [r for r in rows if r.get("split") == split]
    for r in rows:
        r["photos"] = [p for p in r["photos"].split(";") if p]
        labels = {}
        for tok in filter(None, r["labels"].split(";")):
            name, _, qty = tok.partition(":")
            labels[name.strip()] = float(qty) if qty else None
        r["labels"] = labels  # label -> count (None if not counted)
    return rows


def label_mapper(dataset: str):
    """Returns f(label) -> canonical id, using data/label_maps/<dataset>.yaml then the canonicalizer."""
    f = path("label_maps") / f"{dataset}.yaml"
    table = yaml.safe_load(f.read_text()) if f.exists() else {}
    canon = get_canon()
    return lambda label: (table or {}).get(label) or canon.match(label.replace("_", " ")).id or f"?{label}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["freiburg", "grocery"])
    ap.add_argument("root")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    make_freiburg(a.root) if a.dataset == "freiburg" else make_grocery(a.root, a.split)


if __name__ == "__main__":
    main()
