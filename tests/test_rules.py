from datetime import date

from core import rules


def test_mifflin_st_jeor_male():
    # 10*70 + 6.25*175 - 5*30 + 5
    assert rules.mifflin_st_jeor("male", 70, 175, 30) == 1648.75


def test_mifflin_st_jeor_female():
    assert rules.mifflin_st_jeor("female", 60, 165, 25) == 10 * 60 + 6.25 * 165 - 5 * 25 - 161


def test_calorie_target_respects_floor():
    tiny = {"sex": "female", "weight_kg": 35, "height_cm": 140, "age": 80, "activity": "sedentary", "goal": "lose"}
    assert rules.daily_calorie_target(tiny) == 1200


def test_calorie_target_incomplete_profile():
    assert rules.daily_calorie_target({"sex": "male"}) is None


def test_printed_expiry_wins():
    d = date(2026, 10, 20)
    assert rules.estimate_expiry("milk", "fridge", date(2026, 10, 1), d) == (d, "printed")


def test_shelf_life_fallback():
    exp, src = rules.estimate_expiry("milk", "fridge", date(2026, 10, 1))
    assert src == "shelf_life" and exp > date(2026, 10, 1)


def test_unknown_expiry():
    assert rules.estimate_expiry("not_a_food", "fridge", date(2026, 10, 1)) == (None, "unknown")


def test_urgency_bands():
    assert rules.urgency(0) == 1.0
    assert rules.urgency(2) == 1.0
    assert rules.urgency(4) == 0.5
    assert rules.urgency(10) == 0.0
    assert rules.urgency(None) == 0.0


def test_storage_advice_meat_includes_bottom_shelf():
    texts = " ".join(r["text"] for r in rules.storage_advice("chicken"))
    assert "bottom shelf" in texts


def test_shopping_list_skips_staples_and_stock():
    assert rules.shopping_list(["egg", "salt", "milk", "raw:fish sauce"], {"egg"}) == ["milk", "fish sauce"]


def test_restock_list():
    items = [{"name": "milk", "quantity": 0, "min_quantity": 1}, {"name": "egg", "quantity": 6, "min_quantity": 4}]
    assert [i["name"] for i in rules.restock_list(items)] == ["milk"]
