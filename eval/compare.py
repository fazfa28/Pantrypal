"""Paired comparison of two recognition runs on the same manifest (McNemar).

Example: did three photos find more items than one?
  python -m eval.compare eval/results/fridge__qwen3-vl-8b__first__test.jsonl \
                         eval/results/fridge__qwen3-vl-8b__all__test.jsonl

Pairs are (scene, true item): 'found' means the item's canonical id was predicted.
For single-label datasets the pair is the image and 'found' means top-1 correct.
"""
from __future__ import annotations

import argparse
import json

from eval.metrics import mcnemar


def load(file: str) -> dict:
    return {r["id"]: r for r in map(json.loads, open(file, encoding="utf-8")) if r}


def paired_hits(a: dict, b: dict) -> tuple[list[bool], list[bool]]:
    hits_a, hits_b = [], []
    for scene in sorted(a.keys() & b.keys()):
        ra, rb = a[scene], b[scene]
        if len(ra["true"]) == 1:
            t = next(iter(ra["true"]))
            hits_a.append(ra["top1"] == t)
            hits_b.append(rb["top1"] == t)
            continue
        found_a = {p["id"] for p in ra["pred"]}
        found_b = {p["id"] for p in rb["pred"]}
        for item in ra["true"]:
            hits_a.append(item in found_a)
            hits_b.append(item in found_b)
    return hits_a, hits_b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    a = ap.parse_args()
    ha, hb = paired_hits(load(a.run_a), load(a.run_b))
    res = mcnemar(ha, hb)
    print(json.dumps({"pairs": len(ha), "a_rate": sum(ha) / max(len(ha), 1),
                      "b_rate": sum(hb) / max(len(hb), 1), **res}, indent=2))


if __name__ == "__main__":
    main()
