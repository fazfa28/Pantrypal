from eval import metrics


def test_accuracy_and_macro_f1():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    assert metrics.accuracy(y_true, y_pred) == 0.75
    # a: P=1, R=0.5 -> 0.667; b: P=0.667, R=1 -> 0.8
    assert abs(metrics.macro_f1(y_true, y_pred) - (2 / 3 + 0.8) / 2) < 1e-9


def test_set_prf():
    r = metrics.set_prf([{"milk", "egg"}], [{"milk", "cheese"}])
    assert r["precision"] == 0.5 and r["recall"] == 0.5


def test_count_mae():
    assert metrics.count_mae([{"egg": 6, "milk": None}], [{"egg": 4, "milk": 1}]) == 2


def test_mcnemar_no_difference():
    assert metrics.mcnemar([True, False], [True, False])["p_value"] == 1.0


def test_mcnemar_clear_difference():
    a = [False] * 10
    b = [True] * 10
    assert metrics.mcnemar(a, b)["p_value"] < 0.01


def test_bootstrap_ci_contains_point_estimate():
    rows = [1, 0, 1, 1, 0, 1, 1, 1]
    lo, hi = metrics.bootstrap_ci(rows, lambda r: sum(r) / len(r), n=500)
    assert lo <= 0.75 <= hi
