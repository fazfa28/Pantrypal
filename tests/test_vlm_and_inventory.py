"""No API calls: these test parsing, post-processing and the SQLite inventory."""
from datetime import date

import pytest

from core.inventory import Inventory
from core.vlm import Item, parse_items, post_process

REPLY = """```json
{"items": [
  {"name": "whole milk", "category": "dairy", "quantity": 1, "unit": "carton",
   "location": "fridge", "printed_expiry": "2026-10-15", "confidence": "high"},
  {"name": "eggs", "quantity": "six?", "location": "Fridge", "printed_expiry": "unknown", "confidence": "medium"},
  {"name": "eggs", "quantity": 6, "location": "fridge", "confidence": "high"},
  {"name": "mystery jar", "quantity": 1, "location": "fridge", "confidence": "low"}
]}
```"""


def test_parse_tolerates_fences_and_bad_fields():
    items = parse_items(REPLY)
    assert len(items) == 4
    assert items[0].printed_expiry == date(2026, 10, 15)
    assert items[1].quantity is None and items[1].printed_expiry is None
    assert items[1].location == "fridge"


def test_parse_rejects_non_json():
    with pytest.raises(ValueError):
        parse_items("Sorry, I can't see anything.")


def test_post_process_merges_and_flags():
    items = post_process(parse_items(REPLY))
    by_id = {i.canonical_id: i for i in items}
    assert by_id["egg"].quantity == 6 and by_id["egg"].confidence == "high"
    mystery = next(i for i in items if i.canonical_id is None)
    assert mystery.needs_review and "unknown food name" in mystery.review_reasons
    assert not by_id["milk"].needs_review


@pytest.fixture
def inv(tmp_path):
    return Inventory(tmp_path / "test.db")


def test_add_estimates_expiry(inv):
    inv.add("whole milk", quantity=1, added_on=date.today())
    item = inv.items()[0]
    assert item["canonical_id"] == "milk" and item["expiry_source"] == "shelf_life"
    assert item["days_left"] is not None


def test_scan_diff_and_apply(inv):
    inv.add("eggs", quantity=12)
    rows = inv.diff_scan([Item(name="eggs", quantity=6, canonical_id="egg"),
                          Item(name="milk", quantity=1, canonical_id="milk")])
    assert {r["canonical_id"]: r["status"] for r in rows} == {"egg": "in stock", "milk": "new"}
    inv.apply_scan(rows)
    qty = {i["canonical_id"]: i["quantity"] for i in inv.items()}
    assert qty == {"egg": 6, "milk": 1}
    assert inv.events("scan")


def test_consume_and_have(inv):
    inv.add("eggs", quantity=2)
    assert inv.consume("egg") and inv.items()[0]["quantity"] == 1
    assert inv.consume("egg") and inv.items() == []
    assert not inv.consume("egg")


def test_profile_roundtrip(inv):
    inv.set_profile({"sex": "male", "age": 30})
    assert inv.profile() == {"sex": "male", "age": 30}
