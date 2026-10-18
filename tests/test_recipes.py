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
