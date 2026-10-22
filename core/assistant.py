"""Conversational assistant: a small LLM that chooses tools and writes the reply.

Grounding rule: recipes may only come from search_recipes, and the model must
write each recipe name as [[Recipe Name]]. After the reply, code checks every
[[name]] against the recipes the tools actually returned; on a miss it asks the
model to rewrite once. `grounded` in the result is what Experiment 2 measures.
"""
from __future__ import annotations

import json
import re
from datetime import date

from core import recipes as rx
from core import rules
from core.canon import get_canon
from core.config import cfg
from core.inventory import Inventory
from core.llm import chat

SYSTEM_PROMPT = """You are PantryPal, a helpful kitchen assistant. Today is {today}.
- Use the tools to see the user's inventory and to find recipes.
- Recommend ONLY recipes returned by search_recipes. Write every recipe name exactly as returned, inside double brackets, like [[Easy Fried Rice]].
- Quote numbers (calories, minutes, days left) only from tool results. Never estimate them yourself.
- Prefer recipes that use items expiring soon, and say which ones.
- Storage advice must come from the storage_advice tool.
- Keep replies short and practical."""

_NAME_TAG = re.compile(r"\[\[(.+?)\]\]")


def _fn(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {}, "required": required or []},
    }}


TOOLS = [
    _fn("get_inventory", "List items in the fridge, freezer and pantry with days until expiry."),
    _fn("search_recipes", "Find recipes cookable from the inventory, ranked by relevance to the user's request. "
        "Fill every filter: use null, \"none\", \"any\" or [] when the user did not ask for it.", {
        "max_minutes": {"type": ["integer", "null"], "description": "time limit the user stated, else null"},
        "diet": {"type": "string", "enum": ["none", "vegetarian", "vegan"]},
        "meal": {"type": "string", "enum": ["any", "dinner", "breakfast", "dessert"],
                 "description": "the meal the user named (lunch counts as dinner); dinner = a main or side dish"},
        "exclude": {"type": "array", "items": {"type": "string"}, "description": "ingredients the user wants to avoid"},
        "top_k": {"type": "integer", "description": "number of recipes, default 5"},
    }, ["max_minutes", "diet", "meal", "exclude"]),
    _fn("get_recipe", "Full ingredients and steps for one recipe.",
        {"recipe_id": {"type": "integer"}}, ["recipe_id"]),
    _fn("nutrition_for", "Calories for a recipe at a number of servings, and the user's daily target.",
        {"recipe_id": {"type": "integer"}, "servings": {"type": "integer"}}, ["recipe_id"]),
    _fn("shopping_list", "Ingredients missing from the inventory for a recipe.",
        {"recipe_id": {"type": "integer"}}, ["recipe_id"]),
    _fn("storage_advice", "How to store a food item, from curated food-safety rules.",
        {"item": {"type": "string"}}, ["item"]),
    _fn("log_cooked", "Record that the user cooked a recipe and remove the used items from the inventory. "
        "Only call this when the user says they cooked it.",
        {"recipe_id": {"type": "integer"}}, ["recipe_id"]),
]


class Assistant:
    def __init__(self, inventory: Inventory, model: str | None = None):
        self.inv = inventory
        self.model = model or cfg("llm", "chat_model")
        self.returned: dict[str, int] = {}  # recipe name (lower) -> id, from tool results this turn

    # ----- tools -----

    def _recipe(self, recipe_id: int) -> dict:
        df = rx.load_index()
        hit = df[df["recipe_id"] == int(recipe_id)]
        if hit.empty:
            raise ValueError(f"no recipe with id {recipe_id}")
        return hit.iloc[0].to_dict()

    def call_tool(self, name: str, args: dict) -> dict:
        try:
            if name == "get_inventory":
                keep = ("name", "canonical_id", "quantity", "unit", "location", "days_left")
                return {"items": [{k: it[k] for k in keep} for it in self.inv.items()]}

            if name == "search_recipes":
                hits = rx.search(
                    self.inv.have(), max_minutes=args.get("max_minutes"), diet=args.get("diet"),
                    meal=args.get("meal"), query=getattr(self, "last_user_text", None),
                    exclude=args.get("exclude"), top_k=min(int(args.get("top_k") or 5), 10),
                    calorie_target=rules.daily_calorie_target(self.inv.profile()),
                )
                for h in hits:
                    self.returned[h["name"].lower()] = h["recipe_id"]
                return {"recipes": [{
                    "recipe_id": h["recipe_id"], "name": h["name"], "minutes": h["minutes"],
                    "coverage": round(h["coverage"], 2), "missing": h["missing"],
                    "substituted": h["substituted"], "calories_per_serving": h["calories_per_serving"],
                } for h in hits]}

            if name == "get_recipe":
                r = self._recipe(args["recipe_id"])
                self.returned[r["name"].lower()] = r["recipe_id"]
                return {k: r[k] for k in ("recipe_id", "name", "minutes", "servings", "ingredients", "instructions")}

            if name == "nutrition_for":
                r = self._recipe(args["recipe_id"])
                servings = int(args.get("servings") or 1)
                per = r["calories_per_serving"]
                target = rules.daily_calorie_target(self.inv.profile())
                return {"recipe": r["name"], "servings": servings, "calories_per_serving": per,
                        "calories_total": round(per * servings) if per is not None else None,
                        "daily_target_per_person": target}

            if name == "shopping_list":
                r = self._recipe(args["recipe_id"])
                return {"recipe": r["name"], "missing": rules.shopping_list(r["ingredient_ids"], set(self.inv.have()))}

            if name == "storage_advice":
                cid = get_canon().match(args["item"]).id
                shelf = {loc: rules.shelf_life_days(cid, loc) for loc in ("fridge", "freezer", "pantry")}
                return {"item": args["item"], "canonical_id": cid, "rules": rules.storage_advice(cid),
                        "shelf_life_days": {k: v for k, v in shelf.items() if v is not None}}

            if name == "log_cooked":
                r = self._recipe(args["recipe_id"])
                # TODO: ask the user for quantities; for now each used item drops by one unit.
                used = [i for i in r["ingredient_ids"] if self.inv.consume(i)]
                self.inv.log("cook", {"recipe_id": r["recipe_id"], "used": used})
                return {"recipe": r["name"], "removed_from_inventory": used}

            return {"error": f"unknown tool {name}"}
        except Exception as e:  # the model sees the error and can recover
            return {"error": str(e)}

    # ----- conversation -----

    def ungrounded(self, text: str) -> list[str]:
        return [n for n in _NAME_TAG.findall(text) if n.strip().lower() not in self.returned]

    def reply(self, history: list[dict], user_text: str) -> dict:
        """history: earlier [{'role': 'user'|'assistant', 'content': str}] turns."""
        self.returned = {}
        self.last_user_text = user_text   # search_recipes ranks feasible recipes by relevance to this
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(today=date.today().isoformat())},
                    *history, {"role": "user", "content": user_text}]
        trace, calls_left = [], cfg("llm", "max_tool_calls", default=4)

        while True:
            out = chat(messages, model=self.model, tools=TOOLS, tool_choice=None if calls_left > 0 else "none")
            if not out["tool_calls"] or calls_left <= 0:
                break
            messages.append({"role": "assistant", "content": out["content"] or None, "tool_calls": [
                {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in out["tool_calls"]]})
            for tc in out["tool_calls"]:
                try:
                    args = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self.call_tool(tc["name"], args) if calls_left > 0 else {"error": "tool limit reached"}
                calls_left -= 1
                trace.append({"tool": tc["name"], "args": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result, default=str)})

        text = out["content"]
        bad = self.ungrounded(text)
        if bad:  # one corrective rewrite
            messages += [{"role": "assistant", "content": text}, {"role": "user", "content":
                         f"These recipes were not returned by search_recipes: {bad}. "
                         "Rewrite your answer using only recipes the tools returned."}]
            text = chat(messages, model=self.model, tools=TOOLS, tool_choice="none")["content"]
            bad = self.ungrounded(text)
        return {"text": text, "grounded": not bad, "ungrounded": bad, "trace": trace, "recipes": dict(self.returned)}


def pretty(text: str) -> str:
    """[[Recipe]] -> **Recipe** for display."""
    return _NAME_TAG.sub(r"**\1**", text)
