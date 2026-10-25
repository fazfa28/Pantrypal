"""Recipe recommendation for the app: the two Experiment 2 methods, run on the user's inventory.

  llm_only   inventory + request -> the model writes 5 recipes (no database)
  pantrypal  the model turns the request into filters; code filters, keeps cookable recipes, ranks them
             by similarity to the request and offers 20; the model picks 5 by id

Both reuse the frozen Experiment 2 code (eval/run_recommend.py) unchanged, so the demo runs exactly
the methods that were measured. Code then adds what the user needs to see: what each recipe is
missing from the inventory, whether it is cookable, and (for database recipes) the steps.
"""
from __future__ import annotations

from core import recipes as rx
from core.config import cfg
from core.llm import cost_usd

METHODS = {"pantrypal": "PantryPal", "llm_only": "LLM-only"}
DEMO_MODEL = "glm-5.3-flash"   # key in config.yaml eval_models: the model used in Experiments 1 and 2


def demo_model() -> dict:
    """{'id': ..., 'price': [...]} of the demo model; falls back to the app's chat model."""
    return (cfg("eval_models", default={}) or {}).get(DEMO_MODEL) or {"id": cfg("llm", "chat_model"), "price": None}


STEPS_PROMPT = """Write the cooking steps for this recipe.
Recipe: {name}
Ingredients: {ingredients}
The user asked: "{request}"

Use only the listed ingredients (salt, pepper, oil, water and sugar are also available). Write 4 to 8 short steps, with amounts where useful.

Return JSON only, no other text:
{{"steps": ["<step>", ...]}}"""


def write_steps(pick: dict, request: str) -> list[str]:
    """Steps for an LLM-only recipe, written on demand (the Experiment 2 prompt asks for none).
    These are generated text, not a tested recipe."""
    from eval.run_recommend import ask_json

    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    prompt = STEPS_PROMPT.format(name=pick["name"], ingredients=", ".join(pick["ingredients"]), request=request)
    data, _, _ = ask_json(prompt, demo_model()["id"], usage)
    return [str(s) for s in (data or {}).get("steps", []) if str(s).strip()]


def recommend(request: str, pantry: list[str], method: str) -> dict:
    """pantry: canonical ids in stock. Returns the run record plus annotated picks and cost."""
    from eval.run_recommend import run_llm_only, run_pantrypal

    m = demo_model()
    case = {"id": "app", "request": request, "pantry": sorted(set(pantry))}
    if method == "llm_only":
        rec = run_llm_only(case, m["id"])
    elif method == "pantrypal":
        rec = run_pantrypal(case, m["id"], rx.load_index())
    else:
        raise ValueError(f"unknown method {method!r}")

    have = set(case["pantry"])
    steps = {}
    if method == "pantrypal" and rec["picks"]:
        df = rx.load_index()
        hits = df[df["recipe_id"].isin([p["recipe_id"] for p in rec["picks"]])]
        steps = dict(zip(hits["recipe_id"].astype(int), hits["instructions"]))
    for p in rec["picks"]:
        p["missing"] = sorted(i.removeprefix("raw:") for i in set(p["ingredient_ids"]) - have)
        p["cookable"] = rx.is_cookable(p["ingredient_ids"], have)
        p["real"] = p["recipe_id"] is not None
        p["steps"] = list(steps.get(p["recipe_id"], []))
    rec["method"] = method
    rec["request"] = request
    rec["model"] = m["id"]
    rec["cost_usd"] = cost_usd(rec["usage"], m.get("price"))
    return rec
