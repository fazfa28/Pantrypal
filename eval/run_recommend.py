"""Experiment 2: recipe recommendation from a pantry + a natural-language request.

  python -m eval.run_recommend --split dev --limit 2          # smoke test
  python -m eval.run_recommend --split dev                    # tune prompts here, then freeze
  python -m eval.run_recommend --split test

Arms (same chat model for all; default glm-5.3-flash):
  llm_only   pantry + request -> the model writes 5 recipes (no database)
  rag        embed request + pantry, take the 20 most similar database recipes (MiniLM), the model picks 5
  pantrypal  the model turns the request into search filters (forced search_recipes tool call); code applies
             them, keeps only recipes cookable from the pantry, ranks those by similarity to the request
             (same MiniLM embeddings as rag) and returns 20 candidates; the model picks 5
rag and pantrypal use the SAME selection prompt and candidate count, so only retrieval differs.
No hard filters are applied in rag (typical RAG); translating the request into filters is PantryPal's job.

Writes eval/results/exp2__<arm>__<split>.jsonl (one line per case; finished cases are skipped on rerun)
and prints the automatic metrics. Model replies are cached, so reruns cost nothing.
Preference fit is scored separately by the judge (eval/judge_recommend.py).
"""
from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

from core import recipes as rx
from core.assistant import TOOLS
from core.canon import get_canon
from core.config import ROOT, cfg, path
from core.llm import chat, cost_usd
from core.rules import staples

CASES = ROOT / "eval" / "cases" / "exp2_cases.jsonl"
K, N_CANDIDATES = 5, 20
ALWAYS = "Salt, pepper, oil, water and sugar are always available."

SELECT_PROMPT = """You are a recipe assistant. A user asked: "{request}"
Their pantry: {pantry}. {always}

Candidate recipes from our database (JSON):
{candidates}

Choose the {k} candidates that best satisfy the request: respect every stated requirement (diet, time limit, ingredients to avoid, kind of meal), prefer recipes the pantry can cover, and match the stated preferences. Choose only from the candidates above, by id.

Return JSON only, no other text:
{{"picks": [{{"id": <candidate id>, "reason": "<one short sentence>"}}]}}"""

LLM_ONLY_PROMPT = """You are a recipe assistant. A user asked: "{request}"
Their pantry: {pantry}. {always}

Suggest {k} recipes that best satisfy the request: respect every stated requirement (diet, time limit, ingredients to avoid, kind of meal), prefer recipes the pantry can cover, and match the stated preferences.

Return JSON only, no other text:
{{"recipes": [{{"name": str, "minutes": <total minutes>, "ingredients": [every ingredient the recipe needs, short generic names], "from_pantry": [the ingredients you are taking from the pantry], "reason": "<one short sentence>"}}]}}"""

FILTER_SYSTEM = ("You translate a cooking request into recipe-search filters. Call search_recipes once. "
                 "Set only the filters the user actually asked for. If the request names a meal "
                 "(dinner, lunch, breakfast, dessert), set meal (lunch counts as dinner).")


# ---------- helpers ----------


def parse_json(text: str) -> dict:
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
        raise ValueError("no JSON object")
    return json.loads(text[a : b + 1])


def ask_json(prompt: str, model: str, usage: dict) -> tuple[dict | None, float, bool]:
    """One prompt, JSON reply, one retry on bad JSON. Returns (data, latency, valid_first_time)."""
    messages, latency = [{"role": "user", "content": prompt}], 0.0
    for attempt in range(2):
        out = chat(messages, model=model, use_cache=True)
        latency += out["latency_s"]
        for k in usage:
            usage[k] += out["usage"][k]
        try:
            return parse_json(out["content"]), latency, attempt == 0
        except (ValueError, json.JSONDecodeError):
            messages = messages + [{"role": "assistant", "content": out["content"]},
                                   {"role": "user", "content": "Not valid JSON in the required shape. Reply with the JSON object only."}]
    return None, latency, False


def name(cid: str) -> str:
    return cid.replace("_", " ")


def canon_ids(ingredients: list[str]) -> list[str]:
    canon, st = get_canon(), staples()
    ids = [canon.match(x).id or f"raw:{x.strip().lower()}" for x in ingredients]
    return [i for i in dict.fromkeys(ids) if i not in st]


def view(rec: dict) -> dict:
    """What the selection prompt shows about a candidate."""
    return {"id": int(rec["recipe_id"]), "name": rec["name"], "minutes": int(rec["minutes"]),
            "category": rec.get("category", ""), "ingredients": list(rec["ingredients"])}


def pick_record(rec: dict, reason: str = "") -> dict:
    return {"recipe_id": int(rec["recipe_id"]), "name": rec["name"], "minutes": int(rec["minutes"]),
            "category": rec.get("category", ""), "ingredients": list(rec["ingredients"]),
            "ingredient_ids": list(rec["ingredient_ids"]), "reason": reason}


def select(case: dict, candidates: list[dict], model: str, usage: dict) -> tuple[list[dict], dict]:
    """Shared by rag and pantrypal: the model picks K of the candidates by id."""
    by_id = {int(r["recipe_id"]): r for r in candidates}
    prompt = SELECT_PROMPT.format(request=case["request"], pantry=", ".join(map(name, case["pantry"])),
                                  always=ALWAYS, candidates=json.dumps([view(r) for r in candidates]), k=K)
    data, latency, valid = ask_json(prompt, model, usage)
    picks, invalid = [], 0
    for p in (data or {}).get("picks", [])[:K]:
        try:
            rid = int(p.get("id"))
        except (TypeError, ValueError):
            invalid += 1
            continue
        if rid in by_id and rid not in {x["recipe_id"] for x in picks}:
            picks.append(pick_record(by_id[rid], str(p.get("reason", ""))))
        else:
            invalid += 1   # an id that was not offered = a grounding failure
    return picks, {"latency_s": latency, "valid_json": valid, "invalid_picks": invalid}


# ---------- arms ----------


class Rag:
    """Typical RAG retrieval: the 20 recipes most similar to 'request + I have: pantry' (no filters)."""

    def __init__(self, df):
        self.df = df
        self.emb = rx.recipe_embeddings(df)

    def candidates(self, case: dict) -> list[dict]:
        query = f"{case['request']} I have: {', '.join(map(name, case['pantry']))}."
        return [self.df.iloc[i].to_dict() for i in self.emb.top(query, N_CANDIDATES)]


def run_llm_only(case: dict, model: str) -> dict:
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    prompt = LLM_ONLY_PROMPT.format(request=case["request"], pantry=", ".join(map(name, case["pantry"])),
                                    always=ALWAYS, k=K)
    data, latency, valid = ask_json(prompt, model, usage)
    picks = []
    for r in (data or {}).get("recipes", [])[:K]:
        ingredients = [str(x) for x in r.get("ingredients", []) if x]
        try:
            minutes = int(r.get("minutes"))
        except (TypeError, ValueError):
            minutes = None
        picks.append({"recipe_id": None, "name": str(r.get("name", "")), "minutes": minutes, "category": "",
                      "ingredients": ingredients, "ingredient_ids": canon_ids(ingredients),
                      "from_pantry": [str(x) for x in r.get("from_pantry", []) if x],
                      "reason": str(r.get("reason", ""))})
    return {"picks": picks, "usage": usage, "latency_s": latency, "valid_json": valid}


def run_rag(case: dict, model: str, rag: Rag) -> dict:
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    cands = rag.candidates(case)
    picks, info = select(case, cands, model, usage)
    return {"picks": picks, "candidates": [int(r["recipe_id"]) for r in cands], "usage": usage, **info}


SEARCH_TOOL = next(t for t in TOOLS if t["function"]["name"] == "search_recipes")


def run_pantrypal(case: dict, model: str, df) -> dict:
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    out = chat([{"role": "system", "content": FILTER_SYSTEM}, {"role": "user", "content": case["request"]}],
               model=model, tools=[SEARCH_TOOL],
               tool_choice={"type": "function", "function": {"name": "search_recipes"}}, use_cache=True)
    for k in usage:
        usage[k] += out["usage"][k]
    args = {}
    if out["tool_calls"]:
        try:
            args = json.loads(out["tool_calls"][0]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
    filters = {k: args.get(k) for k in ("max_minutes", "diet", "meal", "exclude")}
    diet = filters["diet"] if filters["diet"] in ("vegetarian", "vegan") else None
    cands = rx.search({i: None for i in case["pantry"]}, max_minutes=filters["max_minutes"] or None, diet=diet,
                      exclude=filters["exclude"] or None, meal=filters["meal"], top_k=N_CANDIDATES, df=df,
                      query=case["request"])
    picks, info = select(case, cands, model, usage)
    info["latency_s"] += out["latency_s"]
    return {"picks": picks, "filters": filters, "tool_called": bool(out["tool_calls"]),
            "candidates": [int(r["recipe_id"]) for r in cands], "usage": usage, **info}


# ---------- automatic metrics ----------


def pick_checks(case: dict, p: dict) -> dict:
    """Per-recipe checks against the case spec (only the constraints the case has)."""
    spec = case["spec"]
    out = {"cookable": rx.is_cookable(p["ingredient_ids"], set(case["pantry"]))}
    if spec["diet"]:
        out["diet_ok"] = not rx.violates_diet(p, spec["diet"])
    if spec["max_minutes"]:
        out["time_ok"] = p["minutes"] is not None and p["minutes"] <= spec["max_minutes"]
    if spec["exclude"]:
        word = name(spec["exclude"])
        out["exclude_ok"] = (spec["exclude"] not in p["ingredient_ids"]
                             and not any(word in x.lower() for x in p["ingredients"]))
    return out


def auto_metrics(case: dict, rec: dict, db_names: set[str]) -> dict:
    """Per case, averaged over K slots (a missing pick counts as a failure)."""
    spec, have, picks = case["spec"], set(case["pantry"]), rec["picks"]
    canon = get_canon()

    def share(flags):
        return sum(flags) / K

    checks = [pick_checks(case, p) for p in picks]
    m = {
        "n_picks": len(picks),
        "grounded": share(p["recipe_id"] is not None or p["name"].strip().lower() in db_names for p in picks),
        "cookable": share(c["cookable"] for c in checks),
    }
    for k, field in (("diet_ok", "diet"), ("time_ok", "max_minutes"), ("exclude_ok", "exclude")):
        if spec[field]:
            m[k] = share(c[k] for c in checks)
    if spec["use_up"]:
        m["use_up"] = share(spec["use_up"] in p["ingredient_ids"] for p in picks)
    if "filters" in rec:   # pantrypal: did the model extract the request's hard constraints correctly?
        f = rec["filters"]
        want_ex = {spec["exclude"]} if spec["exclude"] else set()
        got_ex = {canon.match(x).id or x for x in (f.get("exclude") or [])}
        m["filters_correct"] = float(
            (f.get("max_minutes") or None) == spec["max_minutes"]
            and (f.get("diet") if f.get("diet") in ("vegetarian", "vegan") else None) == spec["diet"]
            and (f.get("meal") or "any") == case.get("meal", "any")
            and got_ex == want_ex)
    claimed = [canon.match(x).id for p in picks for x in p.get("from_pantry", [])]
    if claimed:   # llm_only: claimed pantry items that are not in the pantry
        m["hallucinated"] = sum(c not in have for c in claimed) / len(claimed)
    return m


METRICS = ("n_picks", "grounded", "cookable", "diet_ok", "time_ok", "exclude_ok", "use_up", "hallucinated",
           "filters_correct")


def summarize(rows: list[dict]) -> dict:
    s = {"cases": len(rows)}
    for k in METRICS:
        vals = [r["metrics"][k] for r in rows if k in r["metrics"]]
        if vals:
            s[k] = round(statistics.mean(vals), 3)
    s["valid_json"] = round(statistics.mean(r["valid_json"] for r in rows), 3)
    costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    s["usd_per_request"] = round(sum(costs) / len(costs), 6) if costs else None
    s["median_latency_s"] = round(statistics.median(r["latency_s"] for r in rows), 2)
    return s


# ---------- main ----------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--arms", default="llm_only,rag,pantrypal")
    ap.add_argument("--model", default="glm-5.3-flash", help="key in config.yaml eval_models, or a raw model id")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    m = (cfg("eval_models", default={}) or {}).get(a.model) or {"id": a.model, "price": None}
    model, price = m["id"], m.get("price")
    cases = [json.loads(l) for l in open(CASES, encoding="utf-8") if l.strip()]
    cases = [c for c in cases if c["split"] == a.split][: a.limit]
    df = rx.load_index()
    db_names = set(df["name"].str.strip().str.lower())
    arms = a.arms.split(",")
    rag = Rag(df) if "rag" in arms else None

    out_dir = path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for arm in arms:
        out_file = out_dir / f"exp2__{arm}__{a.split}.jsonl"
        done = {}
        if out_file.exists():
            done = {r["id"]: r for r in map(json.loads, out_file.read_text().splitlines()) if r}
        todo = [c for c in cases if c["id"] not in done]
        print(f"{arm}: {len(cases)} cases, {len(done)} done, running {len(todo)} with {model}")

        def run(case):
            if arm == "llm_only":
                rec = run_llm_only(case, model)
            elif arm == "rag":
                rec = run_rag(case, model, rag)
            else:
                rec = run_pantrypal(case, model, df)
            rec = {"id": case["id"], "arm": arm, "model": model, **rec}
            rec["cost_usd"] = cost_usd(rec["usage"], price)
            rec["metrics"] = auto_metrics(case, rec, db_names)
            return rec

        with open(out_file, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as pool:
            futures = {pool.submit(run, c): c["id"] for c in todo}
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                except Exception as e:
                    print(f"  {futures[fut]} failed: {e}")
                    continue
                done[rec["id"]] = rec
                f.write(json.dumps(rec) + "\n")
                f.flush()
        # Metrics are recomputed from the saved picks every time, so fixing a check costs no API calls.
        by_id = {c["id"]: c for c in cases}
        for r in done.values():
            if r["id"] in by_id:
                r["metrics"] = auto_metrics(by_id[r["id"]], r, db_names)
        summaries[arm] = summarize([done[c["id"]] for c in cases if c["id"] in done])

    print(f"\nAutomatic metrics, {a.split} split (shares of {K} slots; higher is better except hallucinated):")
    for arm, s in summaries.items():
        print(f"  {arm:10s} {s}")


if __name__ == "__main__":
    main()
