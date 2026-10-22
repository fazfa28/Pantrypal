import pandas as pd

from core import recipes as rx

RECIPES = pd.DataFrame([
    {"recipe_id": 1, "name": "Egg Fried Rice", "minutes": 20, "servings": 2, "calories_per_serving": 450,
     "review_count": 50, "ingredients": ["eggs", "rice", "scallions", "soy sauce"],
     "ingredient_ids": ["egg", "rice", "spring_onion", "soy_sauce"]},
    {"recipe_id": 2, "name": "Chicken Stir Fry", "minutes": 25, "servings": 2, "calories_per_serving": 500,
     "review_count": 80, "ingredients": ["chicken breast", "bell pepper", "soy sauce", "garlic"],
     "ingredient_ids": ["chicken", "bell_pepper", "soy_sauce", "garlic"]},
    {"recipe_id": 3, "name": "Slow Beef Stew", "minutes": 180, "servings": 4, "calories_per_serving": 600,
     "review_count": 10, "ingredients": ["beef", "potato", "carrot"],
     "ingredient_ids": ["beef", "potato", "carrot"]},
])


def test_parse_list_r_string():
    assert rx.parse_list('c("eggs", "rice")') == ["eggs", "rice"]
    assert rx.parse_list(float("nan")) == []


def test_iso_minutes():
    assert rx.iso_minutes("PT1H30M") == 90
    assert rx.iso_minutes("PT45M") == 45
    assert rx.iso_minutes(None) is None


def test_score_full_coverage_with_urgency():
    s = rx.score_recipe(["egg", "rice"], {"egg": 1, "rice": 30}, alpha=0.3, beta=0.05)
    assert s["coverage"] == 1.0 and s["missing"] == []
    assert abs(s["score"] - (1.0 + 0.3 * 0.5)) < 1e-9


def test_substitute_gives_half_credit():
    s = rx.score_recipe(["spring_onion"], {"onion": 10}, alpha=0, beta=0)
    assert s["coverage"] == 0.5 and s["substituted"] == {"spring_onion": "onion"}


def test_missing_penalty():
    s = rx.score_recipe(["egg", "beef"], {"egg": 10}, alpha=0, beta=0.05)
    assert s["missing"] == ["beef"] and abs(s["score"] - (0.5 - 0.05)) < 1e-9


def test_search_ranks_and_filters():
    have = {"egg": 1, "rice": 20, "onion": 5, "soy_sauce": 60}
    hits = rx.search(have, max_minutes=60, df=RECIPES)
    assert hits[0]["name"] == "Egg Fried Rice"
    assert all(h["minutes"] <= 60 for h in hits)


def test_vegetarian_filter():
    hits = rx.search({"chicken": 1}, diet="vegetarian", df=RECIPES)
    assert "Chicken Stir Fry" not in [h["name"] for h in hits]


def test_exclude():
    hits = rx.search({"egg": 1}, exclude=["soy sauce"], df=RECIPES)
    assert not any("soy_sauce" in h["ingredient_ids"] for h in hits)


def test_cookable_is_the_same_rule_for_every_method():
    from eval.run_retrieval import score_ranked
    rec = {"ingredient_ids": ["milk", "cream", "egg", "rice", "onion"], "missing": [], "ingredients": [],
           "substituted": {"milk": "soy_milk", "cream": "milk"}}
    have = {"soy_milk": None, "egg": None, "rice": None, "onion": None}
    r = score_ranked([rec], have, None)
    assert r["cookable"] == 0.0 and r["cookable_subs"] == 1.0   # strict: milk+cream missing; with substitutes: none


def test_random_inventories_use_common_ids_only():
    import pandas as pd
    from eval.run_retrieval import random_inventories
    common = [f"x{i}" for i in range(25)]
    df = pd.DataFrame({"ingredient_ids": [common + ["rare"]] * 1 + [common + ["salt"]] * 30})
    invs = random_inventories(5, df, seed=1, min_recipes=20)
    assert all(set(i["have"]) <= set(common) for i in invs)   # no rare id, no staple (salt)
    assert random_inventories(5, df, seed=1, min_recipes=20) == invs      # reproducible


def test_staples_switch(monkeypatch):
    from core import rules
    assert "salt" in rules.staples() and rules.staples(False) == set()


def test_is_cookable_needs_enough_matched():
    assert rx.is_cookable(["egg", "rice", "onion"], {"egg", "rice", "onion"}, 1, 3)
    assert rx.is_cookable(["egg", "rice", "onion", "beef"], {"egg", "rice", "onion"}, 1, 3)
    assert not rx.is_cookable(["egg"], set(), 1, 3)                       # 1 missing but nothing matched
    assert not rx.is_cookable(["egg", "rice", "beef", "pork"], {"egg", "rice"}, 1, 3)


def test_diet_check_uses_recipe_name():
    rec = {"name": "Veal Marsala", "ingredient_ids": ["butter", "mushroom"], "ingredients": ["butter", "mushroom"]}
    assert rx.violates_diet(rec, "vegetarian")


def test_suits_meal():
    assert rx.suits_meal({"name": "Chicken Stir Fry", "category": "One Dish Meal"}, "dinner")
    assert not rx.suits_meal({"name": "Stir-Fry Sauce", "category": "Lunch/Snacks"}, "dinner")
    assert not rx.suits_meal({"name": "Fluffy Omelet", "category": "Breakfast"}, "dinner")
    assert rx.suits_meal({"name": "Brownies", "category": "Dessert"}, "dessert")
