"""SQLite inventory: items, the user profile, and an event log (scan, cook, purchase)."""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

from core import rules
from core.canon import get_canon
from core.config import path

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id            INTEGER PRIMARY KEY,
    canonical_id  TEXT,
    name          TEXT NOT NULL,
    category      TEXT,
    quantity      REAL,
    unit          TEXT,
    location      TEXT DEFAULT 'fridge',
    added_on      TEXT,
    expiry_date   TEXT,
    expiry_source TEXT,
    min_quantity  REAL
);
CREATE TABLE IF NOT EXISTS profile (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, ts TEXT, kind TEXT, payload TEXT
);
"""

EDITABLE = {"name", "canonical_id", "quantity", "unit", "location", "expiry_date", "expiry_source", "min_quantity"}


def _iso(d) -> str | None:
    if d is None or d == "" or (isinstance(d, float) and d != d):  # None, "", NaN
        return None
    return d.isoformat() if isinstance(d, (date, datetime)) else str(d)[:10]


class Inventory:
    def __init__(self, db_path: str | Path | None = None):
        self.path = Path(db_path or path("db"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # ----- items -----

    def items(self, location: str | None = None, today: date | None = None) -> list[dict]:
        sql, args = "SELECT * FROM items", ()
        if location:
            sql, args = sql + " WHERE location = ?", (location,)
        rows = [dict(r) for r in self.conn.execute(sql + " ORDER BY expiry_date IS NULL, expiry_date", args)]
        for r in rows:
            r["days_left"] = rules.days_left(r["expiry_date"], today)
            r["use_soon"] = rules.use_soon(r["days_left"])
        return rows

    def add(
        self,
        name: str,
        canonical_id: str | None = None,
        quantity: float | None = None,
        unit: str | None = None,
        location: str = "fridge",
        printed_expiry: date | str | None = None,
        added_on: date | None = None,
        min_quantity: float | None = None,
    ) -> int:
        canon = get_canon()
        canonical_id = canonical_id or canon.match(name).id
        added_on = added_on or date.today()
        printed = date.fromisoformat(_iso(printed_expiry)) if _iso(printed_expiry) else None
        expiry, source = rules.estimate_expiry(canonical_id, location, added_on, printed)
        cur = self.conn.execute(
            """INSERT INTO items (canonical_id, name, category, quantity, unit, location,
                                  added_on, expiry_date, expiry_source, min_quantity)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (canonical_id, name, canon.category.get(canonical_id or "", "other"), quantity, unit,
             location, added_on.isoformat(), _iso(expiry), source, min_quantity),
        )
        self.conn.commit()
        return cur.lastrowid

    def update(self, item_id: int, **fields) -> None:
        fields = {k: (_iso(v) if k == "expiry_date" else v) for k, v in fields.items() if k in EDITABLE}
        if not fields:
            return
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(f"UPDATE items SET {sets} WHERE id = ?", (*fields.values(), item_id))
        self.conn.commit()

    def remove(self, item_id: int) -> None:
        self.conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
        self.conn.commit()

    def have(self) -> dict[str, int | None]:
        """canonical_id -> fewest days left among items in stock. Used by recipe matching."""
        out: dict[str, int | None] = {}
        for it in self.items():
            cid = it["canonical_id"]
            if not cid or (it["quantity"] is not None and it["quantity"] <= 0):
                continue
            d = it["days_left"]
            if cid not in out or (d is not None and (out[cid] is None or d < out[cid])):
                out[cid] = d
        return out

    # ----- scans -----

    def diff_scan(self, scanned: list) -> list[dict]:
        """Compare a scan with the current inventory. Each row gets status 'new' or 'in stock'."""
        current = {(it["canonical_id"], it["location"]): it for it in self.items() if it["canonical_id"]}
        rows = []
        for s in scanned:
            s = s.model_dump() if hasattr(s, "model_dump") else dict(s)
            existing = current.get((s.get("canonical_id"), s.get("location")))
            rows.append({
                **s,
                "keep": True,
                "status": "in stock" if existing else "new",
                "existing_id": existing["id"] if existing else None,
                "flag": ", ".join(s.get("review_reasons") or []),
            })
        return rows

    def apply_scan(self, rows: list[dict]) -> None:
        """Add confirmed new items; for items already in stock, take the scanned quantity."""
        for r in rows:
            if r.get("existing_id"):
                self.update(int(r["existing_id"]), quantity=r.get("quantity"))
            else:
                self.add(
                    name=r["name"], canonical_id=r.get("canonical_id") or None, quantity=r.get("quantity"),
                    unit=r.get("unit"), location=r.get("location") or "fridge",
                    printed_expiry=r.get("printed_expiry"),
                )
        self.log("scan", {"n_items": len(rows)})

    def consume(self, canonical_id: str, amount: float = 1.0) -> bool:
        """Use up `amount` of the soonest-expiring item with this id. Returns False if none."""
        for it in self.items():
            if it["canonical_id"] != canonical_id:
                continue
            q = it["quantity"]
            if q is None or q - amount <= 0:
                self.remove(it["id"])
            else:
                self.update(it["id"], quantity=q - amount)
            return True
        return False

    # ----- profile and events -----

    def profile(self) -> dict:
        return {r["key"]: json.loads(r["value"]) for r in self.conn.execute("SELECT * FROM profile")}

    def set_profile(self, values: dict) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO profile (key, value) VALUES (?, ?)",
            [(k, json.dumps(v)) for k, v in values.items()],
        )
        self.conn.commit()

    def log(self, kind: str, payload: dict) -> None:
        self.conn.execute(
            "INSERT INTO events (ts, kind, payload) VALUES (?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), kind, json.dumps(payload, default=str)),
        )
        self.conn.commit()

    def events(self, kind: str | None = None) -> list[dict]:
        sql = "SELECT * FROM events" + (" WHERE kind = ?" if kind else "") + " ORDER BY id"
        return [dict(r) for r in self.conn.execute(sql, (kind,) if kind else ())]
