"""Metrics for Experiments 1 and 2. Pure Python, no sklearn needed."""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Callable, Sequence


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    return sum(t == p for t, p in zip(y_true, y_pred)) / (len(y_true) or 1)


def per_class_f1(y_true: Sequence, y_pred: Sequence) -> dict:
    labels = sorted(set(y_true), key=str)
    tp, fp, fn = Counter(), Counter(), Counter()
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fn[t] += 1
            fp[p] += 1
    out = {}
    for c in labels:
        prec = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        out[c] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return out


def macro_f1(y_true: Sequence, y_pred: Sequence) -> float:
    """Averaged over the true classes only, so stray predicted labels count as errors, not classes."""
    f1 = per_class_f1(y_true, y_pred)
    return sum(f1.values()) / (len(f1) or 1)


def set_prf(true_sets: Sequence[set], pred_sets: Sequence[set]) -> dict:
    """Micro precision / recall / F1 over scenes with several items each."""
    tp = sum(len(t & p) for t, p in zip(true_sets, pred_sets))
    n_pred = sum(len(p) for p in pred_sets)
    n_true = sum(len(t) for t in true_sets)
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_true if n_true else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def count_mae(true_counts: Sequence[dict], pred_counts: Sequence[dict]) -> float | None:
    """Mean absolute count error over items that were both labelled with a count and detected."""
    errs = [abs(t[k] - p[k]) for t, p in zip(true_counts, pred_counts)
            for k in t if k in p and t[k] is not None and p[k] is not None]
    return sum(errs) / len(errs) if errs else None


def bootstrap_ci(rows: Sequence, stat: Callable[[Sequence], float], n: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI: resample rows (e.g. images) with replacement."""
    rng = random.Random(seed)
    vals = sorted(stat([rows[rng.randrange(len(rows))] for _ in rows]) for _ in range(n))
    return vals[int(n * alpha / 2)], vals[int(n * (1 - alpha / 2)) - 1]


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict:
    """Exact McNemar test for paired outcomes (e.g. 1 photo vs 3 photos on the same items)."""
    b = sum(x and not y for x, y in zip(a_correct, b_correct))  # A right, B wrong
    c = sum(y and not x for x, y in zip(a_correct, b_correct))  # B right, A wrong
    n = b + c
    if n == 0:
        return {"a_only": b, "b_only": c, "p_value": 1.0}
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return {"a_only": b, "b_only": c, "p_value": min(1.0, 2 * tail)}
