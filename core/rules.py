"""Deterministic rules: expiry, urgency, calories, storage advice, shopping and restock lists.

Nothing in this file calls a model. Anything a user relies on being correct lives here.
"""
from __future__ import annotations

from datetime import date, timedelta

from core.canon import get_canon
from core.config import cfg, load_yaml

# ---------- Expiry ----------


def shelf_life_days(canonical_id: str | None, location: str) -> int | None:
    row = load_yaml("shelf_life").get(canonical_id or "", {}) or {}
    return row.get(location)


def estimate_expiry(
    canonical_id: str | None,
    location: str,
    added_on: date,
    printed_expiry: date | None = None,
) -> tuple[date | None, str]:
    """Printed date wins; otherwise added_on + shelf-life days. Returns (date, source)."""
    if printed_expiry:
        return printed_expiry, "printed"
    days = shelf_life_days(canonical_id, location)
    if days is None:
        return None, "unknown"
    return added_on + timedelta(days=days), "shelf_life"


def days_left(expiry: date | str | None, today: date | None = None) -> int | None:
    if expiry is None or expiry == "":
        return None
    if isinstance(expiry, str):
        expiry = date.fromisoformat(expiry)
    return (expiry - (today or date.today())).days


def urgency(days: int | None) -> float:
    """u(r) in the match score: 1 within use_soon_days, 0.5 within soon_days, else 0."""
    if days is None:
        return 0.0
    if days <= cfg("rules", "use_soon_days", default=2):
        return 1.0
    if days <= cfg("rules", "soon_days", default=5):
        return 0.5
    return 0.0


def use_soon(days: int | None) -> bool:
    return days is not None and days <= cfg("rules", "use_soon_days", default=2)


# ---------- Nutrition ----------

ACTIVITY_FACTORS = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "very_active": 1.9}
GOAL_ADJUSTMENT = {"lose": -500, "maintain": 0, "gain": 300}  # kcal/day; cite your source


def mifflin_st_jeor(sex: str, weight_kg: float, height_cm: float, age: float) -> float:
    """Resting energy expenditure, kcal/day (Mifflin et al., 1990)."""
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    return base + 5 if sex == "male" else base - 161


def daily_calorie_target(profile: dict) -> float | None:
    """None if the profile is incomplete. Never below the configured floor."""
    try:
        sex = profile["sex"]
        ree = mifflin_st_jeor(sex, float(profile["weight_kg"]), float(profile["height_cm"]), float(profile["age"]))
    except (KeyError, TypeError, ValueError):
        return None
    target = ree * ACTIVITY_FACTORS.get(profile.get("activity", "light"), 1.375)
    target += GOAL_ADJUSTMENT.get(profile.get("goal", "maintain"), 0)
    floor = cfg("rules", "calorie_floor", default={}).get(sex, 0)
    return round(max(target, floor))


# ---------- Storage advice ----------


def storage_advice(canonical_id: str | None, location: str | None = None) -> list[dict]:
    """Rules from data/storage_rules.yaml that apply to this item. Returns [{text, source}]."""
    category = get_canon().category.get(canonical_id or "", "other")
    out = []
    for rule in load_yaml("storage_rules").get("rules", []):
        a = rule.get("applies_to", {})
        if (
            a.get("all")
            or canonical_id in a.get("ids", [])
            or category in a.get("categories", [])
            or (location and location in a.get("locations", []))
        ):
            out.append({"text": rule["text"], "source": rule.get("source", "")})
    return out


# ---------- Shopping and restock ----------


def staples(enabled: bool | None = None) -> set[str]:
    """Ids assumed to be in every kitchen. enabled=False (or config rules.use_staples: false) returns an
    empty set, so nothing is assumed: used for the "no staples" ablation."""
    on = cfg("rules", "use_staples", default=True) if enabled is None else enabled
    return set(load_yaml("staples").get("staples", [])) if on else set()


def shopping_list(recipe_ingredient_ids: list[str], have_ids: set[str]) -> list[str]:
    """Recipe ingredients not in stock, excluding staples. Unmatched ones keep a readable name."""
    missing = []
    for i in recipe_ingredient_ids:
        if i in have_ids or i in staples():
            continue
        missing.append(i.removeprefix("raw:").replace("_", " "))
    return missing


def restock_list(items: list[dict]) -> list[dict]:
    """Items whose quantity has fallen below the user's min_quantity (MVP: fixed thresholds)."""
    return [
        it
        for it in items
        if it.get("min_quantity") is not None and (it.get("quantity") or 0) < it["min_quantity"]
    ]
