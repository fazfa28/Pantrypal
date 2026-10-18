"""Photos -> validated list of inventory items.

One request per scan with all photos attached. The model returns JSON; pydantic
validates it; code (not the model) maps names to canonical ids and decides
which items need the user's review.
"""
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from core.canon import get_canon
from core.config import cfg
from core.llm import chat

LOCATIONS = ("fridge", "freezer", "pantry")
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}

PROMPT = """You are a kitchen inventory assistant. Look at ALL the photos together. They may show the same fridge, freezer or pantry from different angles. List every distinct food or drink item you can see.

Return JSON only, no other text, in exactly this shape:
{"items": [{"name": str, "category": str, "quantity": number or null, "unit": str or null, "location": "fridge" | "freezer" | "pantry", "printed_expiry": "YYYY-MM-DD" or null, "confidence": "high" | "medium" | "low"}]}

Rules:
- name: a short generic English food name ("whole milk", "eggs", "spring onion"), not a brand.
- category: one of dairy, egg, meat, seafood, produce, fruit, grain, bakery, condiment, snack, beverage, baking, plant_protein, other.
- An item seen in several photos is listed ONCE.
- quantity: the number of units you can see. Use null if you cannot tell. Never guess.
- printed_expiry: only if a date is clearly readable on the package, otherwise null.
- confidence: "low" if the item is partly hidden, blurry, or you are unsure what it is.
- List the most prominent item first."""


CLOSED_PROMPT = """You are a grocery product classifier. Each image shows one packaged grocery product, usually on a shelf or in a kitchen.

Choose the single best category for the main product from this list:
{classes}

Guidelines:
- Classify what the product IS, not its brand or packaging colour.
- If several products are visible, choose the most prominent one.
- If nothing fits well, still choose the closest category. Never answer with a name outside the list.

Return JSON only, no other text, in exactly this shape:
{"items": [{"name": "<one category from the list>", "category": "<same category>", "quantity": 1, "unit": null, "location": "pantry", "printed_expiry": null, "confidence": "high" | "medium" | "low"}]}

Return exactly one item."""


def closed_prompt(classes: list[str]) -> str:
    return CLOSED_PROMPT.replace("{classes}", ", ".join(classes))


def _norm(name: str) -> str:
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


class Item(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    category: str = "other"
    quantity: float | None = None
    unit: str | None = None
    location: Literal["fridge", "freezer", "pantry"] = "fridge"
    printed_expiry: date | None = None
    confidence: Literal["high", "medium", "low"] = "low"
    # Filled in by code, never by the model:
    canonical_id: str | None = None
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)

    @field_validator("quantity", mode="before")
    @classmethod
    def _quantity(cls, v):
        try:
            return None if v in (None, "") else float(v)
        except (TypeError, ValueError):
            return None

    @field_validator("printed_expiry", mode="before")
    @classmethod
    def _expiry(cls, v):
        if v in (None, "", "null") or isinstance(v, date):
            return v or None
        try:
            return date.fromisoformat(str(v)[:10])
        except ValueError:
            return None

    @field_validator("location", mode="before")
    @classmethod
    def _location(cls, v):
        v = str(v or "").lower()
        return v if v in LOCATIONS else "fridge"

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence(cls, v):
        v = str(v or "").lower()
        return v if v in CONFIDENCE_RANK else "low"


class ScanResult(BaseModel):
    items: list[Item]


@dataclass
class ScanOutput:
    items: list[Item]
    model: str
    valid_json: bool
    raw_text: str = ""
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})
    latency_s: float = 0.0


def encode_image(img, max_px: int | None = None) -> str:
    """Path, bytes, file-like (e.g. Streamlit upload) or PIL image -> JPEG data URL."""
    max_px = max_px or cfg("llm", "max_image_px", default=1024)
    if isinstance(img, (str, Path)):
        im = Image.open(img)
    elif isinstance(img, bytes):
        im = Image.open(io.BytesIO(img))
    elif hasattr(img, "read"):
        if hasattr(img, "seek"):
            img.seek(0)
        im = Image.open(img)
    else:
        im = img
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_items(text: str) -> list[Item]:
    """Pull the JSON object out of a model reply (tolerates ``` fences) and validate it."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in reply")
    return ScanResult.model_validate(json.loads(text[start : end + 1])).items


def post_process(items: list[Item], label_ids: dict[str, str] | None = None) -> list[Item]:
    """Canonicalise names, flag items for review, merge duplicates across photos.

    label_ids (closed-set mode): normalised class name -> canonical id. A name in the
    table is mapped directly, so a correct answer is never lost to fuzzy matching.
    """
    canon = get_canon()
    merged: dict[str, Item] = {}
    for it in items:
        it.canonical_id = (label_ids or {}).get(_norm(it.name)) or canon.match(it.name).id
        reasons = []
        if it.confidence == "low":
            reasons.append("low confidence")
        if it.quantity is None:
            reasons.append("quantity unclear")
        if it.canonical_id is None:
            reasons.append("unknown food name")
        it.review_reasons, it.needs_review = reasons, bool(reasons)

        key = f"{it.canonical_id or it.name.lower()}|{it.location}"
        if key not in merged:
            merged[key] = it
            continue
        # Same item seen twice: keep the larger count, the higher confidence, the earliest date.
        a = merged[key]
        if it.quantity is not None and (a.quantity is None or it.quantity > a.quantity):
            a.quantity, a.unit = it.quantity, it.unit or a.unit
        if CONFIDENCE_RANK[it.confidence] > CONFIDENCE_RANK[a.confidence]:
            a.confidence = it.confidence
        dates = [d for d in (a.printed_expiry, it.printed_expiry) if d]
        a.printed_expiry = min(dates) if dates else None
    return list(merged.values())


def recognise(
    images: list,
    model: str | None = None,
    location_hint: str | None = None,
    *,
    use_cache: bool = False,
    base_url: str | None = None,
    api_key_env: str | None = None,
    prompt_text: str | None = None,
    label_ids: dict[str, str] | None = None,
) -> ScanOutput:
    """Send all photos in one request; retry once with a correction if the JSON is invalid."""
    model = model or cfg("llm", "vision_model")
    prompt = (prompt_text or PROMPT) + (f"\n- All photos are from the {location_hint}." if location_hint else "")
    content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": encode_image(img)}} for img in images
    ]
    messages = [{"role": "user", "content": content}]
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    latency, text = 0.0, ""

    for attempt in range(2):
        out = chat(messages, model=model, use_cache=use_cache, base_url=base_url, api_key_env=api_key_env)
        text = out["content"]
        latency += out["latency_s"]
        for k in usage:
            usage[k] += out["usage"][k]
        try:
            items = parse_items(text)
            return ScanOutput(post_process(items, label_ids), model, attempt == 0, text, usage, latency)
        except (ValueError, ValidationError):
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "That was not valid JSON in the required shape. Reply with the JSON object only."},
            ]
    return ScanOutput([], model, False, text, usage, latency)
