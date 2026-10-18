"""Convert USDA FoodKeeper JSON into data/shelf_life.yaml (days per location).

  python -m scripts.build_shelf_life data/raw/foodkeeper.json

FoodKeeper stores ranges like Refrigerate_Min / Refrigerate_Max / Refrigerate_Metric
("Days", "Weeks", ...). We keep the MINIMUM (the conservative end). Field names below
follow the FoodKeeper data dictionary; check them against your download, since the
file has changed format over the years. Names that do not map to a canonical id are
written to data/shelf_life_unmatched.txt for you to add to aliases.yaml.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from core.canon import get_canon
from core.config import ROOT, path

UNIT_DAYS = {"day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30, "year": 365, "years": 365}
# FoodKeeper field prefix -> our location. "DOP_" = after opening/purchase variants.
PREFIXES = {"Refrigerate": "fridge", "DOP_Refrigerate": "fridge", "Freeze": "freezer",
            "DOP_Freeze": "freezer", "Pantry": "pantry", "DOP_Pantry": "pantry"}


def _rows(obj):
    """FoodKeeper nests rows as lists of single-key dicts; flatten every dict that has a 'Name'."""
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) and len(x) == 1 for x in obj):
            merged = {k: v for x in obj for k, v in x.items()}
            if "Name" in merged:
                yield merged
                return
        for x in obj:
            yield from _rows(x)
    elif isinstance(obj, dict):
        if "Name" in obj:
            yield obj
        for v in obj.values():
            yield from _rows(v)


def to_days(value, metric) -> int | None:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    factor = UNIT_DAYS.get(str(metric or "").strip().lower())
    return int(n * factor) if factor else None


def main(src: str):
    data = json.loads(Path(src).read_text(encoding="utf-8"))
    canon, table, unmatched = get_canon(), {}, set()
    for row in _rows(data):
        name = " ".join(str(row.get(k) or "") for k in ("Name", "Name_subtitle")).strip()
        cid = canon.match(row.get("Name", "")).id
        if not cid:
            unmatched.add(name)
            continue
        for prefix, loc in PREFIXES.items():
            days = to_days(row.get(f"{prefix}_Min"), row.get(f"{prefix}_Metric"))
            if days:
                current = table.setdefault(cid, {}).get(loc)
                table[cid][loc] = min(days, current) if current else days
    out = path("shelf_life")
    header = "# Generated from USDA FoodKeeper by scripts/build_shelf_life.py; minimum of each range.\n"
    out.write_text(header + yaml.safe_dump(dict(sorted(table.items())), sort_keys=False), encoding="utf-8")
    (ROOT / "data" / "shelf_life_unmatched.txt").write_text("\n".join(sorted(unmatched)), encoding="utf-8")
    print(f"{len(table)} items written to {out}; {len(unmatched)} names unmatched")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/foodkeeper.json")
