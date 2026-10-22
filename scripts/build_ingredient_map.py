"""Map every distinct recipe ingredient string to a canonical food id, once, offline.

  python -m scripts.build_ingredient_map --limit 50      # trial on the 50 most common strings
  python -m scripts.build_ingredient_map                 # all strings (a few cents); OVERWRITES the extra file
  python -m scripts.build_ingredient_map --only-new      # only index strings that match nothing yet; merges
                                                         # into the extra file, keeping hand edits

An LLM sees the current id list and decides, per string, an existing id or a NEW id (strings are
processed most-common first, and each batch sees the ids created so far, so names stay consistent).
Outputs:
  data/aliases_extra.yaml            new ids + every mapped string as an alias; loaded by core/canon.py
  eval/mapping_check/review.csv      string, count, id, new_id?  -> a teammate reviews this
Model replies are cached (eval/cache), so a rerun with the same inputs costs nothing.
At runtime nothing calls a model: matching is a dictionary lookup.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import yaml

from core.canon import normalize
from core.config import ROOT, cfg, path
from core.llm import chat
from scripts.sample_mappings import occurrences

CATEGORIES = ["dairy", "egg", "meat", "seafood", "produce", "fruit", "grain", "bakery",
              "condiment", "snack", "beverage", "baking", "plant_protein", "other"]
ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")

PROMPT = """You map recipe ingredient strings to canonical food ids for a pantry app.

An id names ONE generic food someone keeps in stock. Two strings share an id only if a cook could use one in place of the other in most recipes.

Rules:
1. Ignore words that do not change the food: size, freshness, brand, cut and trim (fresh, large, chopped, diced, minced, sliced, skinless, boneless, low-fat, unsalted, kosher, lean, cooked...). "boneless skinless chicken breast half" -> chicken.
2. Keep words that change the food. These are DIFFERENT ids: garlic powder vs garlic, lemon juice vs lemon, chicken broth vs chicken, coconut milk vs milk, baking soda vs soda, cream cheese vs cheese, sour cream vs cream, powdered sugar vs sugar, rice wine vs rice, red pepper flakes vs bell pepper, cornmeal vs corn.
3. A ground or dried herb or spice maps to the spice itself: "ground cinnamon" -> cinnamon, "dried oregano" -> oregano. Each specific spice or herb gets its own id. The id "spices" is only for the generic words "spice" or "spices".
4. Existing ids such as cheese, nuts, oil, sugar, salt, chicken, beef, pasta are deliberately broad: use them for all their varieties (parmesan -> cheese, walnuts -> nuts, olive oil -> oil, brown sugar -> sugar).
5. Prefer an existing id. Create a new id only if none fits: lowercase snake_case, singular, generic (cinnamon, coconut_milk, baking_powder). Reuse a new id created earlier in this task instead of inventing a synonym.
6. Use null if the string is not a food or is too vague to decide.

Existing ids:
{ids}

Categories for NEW ids: {cats}

Return JSON only, no other text:
{{"items": [{{"s": "<the string exactly as given>", "id": "<id or null>", "category": "<category, only if the id is new>"}}]}}

Strings:
{strings}"""


def load_base(file: Path) -> dict[str, str]:
    data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    return {cid: (info or {}).get("category", "other") for cid, info in data.items()}


def parse(text: str) -> list[dict]:
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
        raise ValueError("no JSON object")
    return json.loads(text[a : b + 1])["items"]


def map_batch(batch: list[str], ids: dict[str, str], model: str) -> tuple[dict[str, str | None], dict[str, str]]:
    prompt = PROMPT.format(ids=", ".join(sorted(ids)), cats=", ".join(CATEGORIES),
                           strings="\n".join(f"- {s}" for s in batch))
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(2):
        out = chat(messages, model=model, use_cache=True)
        try:
            items = parse(out["content"])
            break
        except (ValueError, KeyError, json.JSONDecodeError):
            messages += [{"role": "assistant", "content": out["content"]},
                         {"role": "user", "content": "Not valid JSON in the required shape. Reply with the JSON object only."}]
    else:
        return {}, {}
    result, new_cat = {}, {}
    for it in items:
        s, cid = it.get("s"), it.get("id")
        if s not in batch:
            continue
        if cid in (None, "", "null"):
            result[s] = None
        elif isinstance(cid, str) and ID_RE.match(cid):
            result[s] = cid
            if cid not in ids:
                new_cat[cid] = it.get("category") if it.get("category") in CATEGORIES else "other"
        else:
            result[s] = None
    return result, new_cat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(ROOT / "data" / "raw" / "recipes.csv"))
    ap.add_argument("--model", default="glm-5.3-flash", help="key in config.yaml eval_models, or a raw model id")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--limit", type=int, help="only the N most common strings (trial run)")
    ap.add_argument("--only-new", action="store_true",
                    help="map only ingredient strings in the current recipe index that canon cannot match")
    a = ap.parse_args()

    model = ((cfg("eval_models", default={}) or {}).get(a.model) or {}).get("id", a.model)
    base_ids = load_base(path("aliases"))
    out = path("aliases").parent / "aliases_extra.yaml"
    if a.only_new:
        from core.canon import get_canon
        from core.recipes import load_index
        canon = get_canon()
        ids = dict(canon.category)          # base + existing extra ids
        counts = Counter(s.strip().lower() for names in load_index()["ingredients"] for s in names
                         if canon.match(s).id is None)
    else:
        ids = dict(base_ids)
        counts = Counter(occurrences(Path(a.src)))
    strings = [s for s, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][: a.limit]
    print(f"{len(strings)} strings, model {model}")

    mapping: dict[str, str | None] = {}
    for i in range(0, len(strings), a.batch):
        batch = strings[i : i + a.batch]
        res, new = map_batch(batch, ids, model)
        mapping.update(res)
        ids.update(new)
        print(f"  {min(i + a.batch, len(strings))}/{len(strings)}  ids so far: {len(ids)}")
    unanswered = [s for s in strings if s not in mapping]
    if unanswered:
        print(f"WARNING: {len(unanswered)} strings got no answer (left unmapped): {unanswered[:5]}...")

    # aliases_extra.yaml: every mapped string becomes an alias; new ids carry their category.
    extra: dict[str, dict] = {}
    if a.only_new and out.exists():
        extra = yaml.safe_load(out.read_text(encoding="utf-8")) or {}
    for s, cid in mapping.items():
        if cid:
            e = extra.setdefault(cid, {"category": ids[cid], "aliases": []})
            if s not in e["aliases"]:
                e["aliases"].append(s)
    header = ("# Generated by scripts/build_ingredient_map.py. Edit by hand freely after review;\n"
              "# rerunning the script overwrites this file. Base ids/aliases live in aliases.yaml and win on conflict.\n")
    out.write_text(header + yaml.safe_dump(extra, sort_keys=True, allow_unicode=True, width=120), encoding="utf-8")

    review = ROOT / "eval" / "mapping_check" / ("review_new.csv" if a.only_new else "review.csv")
    review.parent.mkdir(parents=True, exist_ok=True)
    with open(review, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ingredient", "count", "id", "new_id", "reviewer_ok"])
        for s in strings:
            cid = mapping.get(s)
            w.writerow([s, counts[s], cid or "", "yes" if cid and cid not in base_ids else "", ""])
    if a.only_new:
        print("Rebuild the recipe index so the new aliases take effect: python -m core.recipes build data/raw/recipes.csv")

    new_ids = [c for c in ids if c not in base_ids]
    print(f"mapped {sum(1 for v in mapping.values() if v)}/{len(strings)}; {len(new_ids)} new ids")
    print(f"wrote {out}\nwrote {review}")


if __name__ == "__main__":
    main()
