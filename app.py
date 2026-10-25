"""PantryPal Streamlit app: Scan | Inventory | Recommend | Chat | Shopping.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st

from core import rules
from core.assistant import Assistant, pretty
from core.canon import get_canon
from core.config import cfg
from core.inventory import Inventory
from core.recommend import METHODS, demo_model, recommend, write_steps
from core.vlm import recognise

st.set_page_config(page_title="PantryPal", page_icon="🥕", layout="wide")
MODEL = demo_model()["id"]   # GLM 5.3 Flash, as in Experiments 1 and 2


@st.cache_resource
def get_inventory() -> Inventory:
    return Inventory()


inv = get_inventory()

# ---------------- Sidebar: profile ----------------
with st.sidebar:
    st.header("Your profile")
    p = inv.profile()
    with st.form("profile"):
        sex = st.selectbox("Sex", ["female", "male"], index=["female", "male"].index(p.get("sex", "female")))
        age = st.number_input("Age", 10, 100, int(p.get("age", 25)))
        height = st.number_input("Height (cm)", 100, 230, int(p.get("height_cm", 165)))
        weight = st.number_input("Weight (kg)", 30, 250, int(p.get("weight_kg", 60)))
        activities = list(rules.ACTIVITY_FACTORS)
        activity = st.selectbox("Activity", activities, index=activities.index(p.get("activity", "light")))
        goals = list(rules.GOAL_ADJUSTMENT)
        goal = st.selectbox("Goal", goals, index=goals.index(p.get("goal", "maintain")))
        diet = st.selectbox("Diet", ["none", "vegetarian", "vegan"],
                            index=["none", "vegetarian", "vegan"].index(p.get("diet", "none")))
        household = st.number_input("Household size", 1, 12, int(p.get("household", 1)))
        if st.form_submit_button("Save"):
            inv.set_profile({"sex": sex, "age": age, "height_cm": height, "weight_kg": weight,
                             "activity": activity, "goal": goal, "diet": diet, "household": household})
            st.rerun()
    target = rules.daily_calorie_target(inv.profile())
    if target:
        st.metric("Daily energy target", f"{target} kcal")

scan_tab, inv_tab, rec_tab, chat_tab, shop_tab = st.tabs(["Scan", "Inventory", "Recommend", "Chat", "Shopping"])

# ---------------- Scan ----------------
with scan_tab:
    scan_models = {"GLM 5.3 Flash": MODEL, "Qwen3-VL 235B": cfg("eval_models")["qwen3-vl-235b"]["id"]}
    scan_model = scan_models[st.selectbox(
        "Scan model", list(scan_models),
        help="GLM was best on single-product close-ups (Experiment 1). On a real shelf with many items "
             "it can miss or invent items; Qwen3-VL 235B is slower (about 35 s) but reads cluttered photos better.")]
    location = st.radio("Where are these photos from?", ["fridge", "freezer", "pantry"], horizontal=True)
    uploads = st.file_uploader("Photos (1 to 3 angles)", type=["jpg", "jpeg", "png", "webp"],
                               accept_multiple_files=True)
    shot = st.camera_input("Or take a photo now")
    photos = list(uploads or []) + ([shot] if shot else [])
    if photos:
        st.image(photos, width=260)

    if st.button("Recognise items", type="primary", disabled=not photos):
        with st.spinner("Reading your photos..."):
            out = recognise(photos, model=scan_model, location_hint=location)
        if not out.items:
            st.error("The model did not return a usable item list. Try clearer or fewer photos.")
        st.session_state.scan = inv.diff_scan(out.items)
        st.session_state.scan_json = {"model": out.model, "items": [it.model_dump(mode="json") for it in out.items]}
        st.session_state.scan_raw = out.raw_text
        st.session_state.scan_meta = f"{out.model} · {out.latency_s:.1f}s · {sum(out.usage.values())} tokens"

    if st.session_state.get("scan_json"):
        st.caption(st.session_state.scan_meta)
        with st.expander("Model output as JSON"):
            st.markdown("**1. Exactly what the model replied**")
            st.code(st.session_state.get("scan_raw") or "", language="json")
            st.markdown("**2. After code:** validated, plus `canonical_id` from the lookup table and review flags")
            st.json(st.session_state.scan_json)
            st.download_button("Download JSON", json.dumps(st.session_state.scan_json, indent=2),
                               file_name="scan.json", mime="application/json")

    if st.session_state.get("scan"):
        st.write("**Please verify.** Untick anything wrong, fix names or quantities, then add to your inventory. "
                 "Rows in *Check* need a look.")
        cols = ["keep", "name", "canonical_id", "quantity", "unit", "location", "printed_expiry",
                "status", "flag", "existing_id"]
        edited = st.data_editor(
            pd.DataFrame(st.session_state.scan)[cols], hide_index=True, key="scan_editor",
            column_config={
                "keep": st.column_config.CheckboxColumn("Add"),
                "location": st.column_config.SelectboxColumn(options=["fridge", "freezer", "pantry"]),
                "printed_expiry": st.column_config.DateColumn("Expiry on pack"),
                "status": st.column_config.TextColumn(disabled=True),
                "flag": st.column_config.TextColumn("Check", disabled=True),
                "existing_id": None,  # hidden
            },
        )
        if st.button("Add to inventory"):
            # The editor turns empty cells into NaN/NaT; make them None so "no value" stays falsy.
            rows = [{k: (None if not isinstance(v, (list, dict)) and pd.isna(v) else v) for k, v in r.items()}
                    for r in edited[edited["keep"]].to_dict("records")]
            originals = {r["name"]: r for r in st.session_state.scan}
            for r in rows:   # a renamed or unmapped item gets its food id from the (corrected) name
                if r["name"] not in originals or not r.get("canonical_id"):
                    r["canonical_id"] = get_canon().match(r["name"]).id
            inv.apply_scan(rows)
            st.session_state.scan = st.session_state.scan_json = None
            st.success(f"Saved {len(rows)} items.")
            st.rerun()

# ---------------- Inventory ----------------
with inv_tab:
    with st.expander("Add items by hand", expanded=not inv.items()):
        with st.form("manual_add", clear_on_submit=True):
            names = st.text_input("Food names", placeholder="milk, butter, onion",
                                  help="One item, or several separated by commas.")
            c1, c2, c3 = st.columns(3)
            qty = c1.number_input("Quantity (each)", min_value=0.0, value=1.0, step=1.0)
            unit = c2.text_input("Unit (optional)", placeholder="pcs, g, ml")
            loc = c3.selectbox("Location", ["fridge", "freezer", "pantry"])
            expiry = st.date_input("Expiry date on pack (optional)", value=None)
            if st.form_submit_button("Add", type="primary"):
                added, unknown = [], []
                for n in (x.strip() for x in names.split(",")):
                    if not n:
                        continue
                    cid = get_canon().match(n).id
                    inv.add(name=n, canonical_id=cid, quantity=qty, unit=unit or None, location=loc,
                            printed_expiry=expiry)
                    (added if cid else unknown).append(f"{n} → {cid}" if cid else n)
                if added:
                    st.success("Added: " + ", ".join(added))
                if unknown:
                    st.warning("Added, but not recognised as a food, so recipes won't use them: " + ", ".join(unknown)
                               + ". Try a more generic name (e.g. \"cheese\" instead of a brand).")

    items = inv.items()
    if not items:
        st.info("Nothing here yet. Scan a photo, or add items by hand above.")
    else:
        soon = [i["name"] for i in items if i["use_soon"]]
        if soon:
            st.warning("Use soon: " + ", ".join(soon))
        df = pd.DataFrame(items)
        df["remove"] = False
        edited = st.data_editor(
            df[["id", "name", "quantity", "unit", "location", "expiry_date", "days_left", "min_quantity", "remove"]],
            hide_index=True, key="inv_editor", disabled=["id", "days_left"],
            column_config={"id": None, "min_quantity": st.column_config.NumberColumn("Restock below")},
        )
        if st.button("Save changes"):
            for row in edited.to_dict("records"):
                if row["remove"]:
                    inv.remove(int(row["id"]))
                else:
                    inv.update(int(row["id"]), name=row["name"], quantity=row["quantity"], unit=row["unit"],
                               location=row["location"], expiry_date=row["expiry_date"],
                               min_quantity=row["min_quantity"])
            st.rerun()

# ---------------- Recommend ----------------


def show_recipe(n: int, p: dict, rec: dict) -> None:
    with st.container(border=True):
        st.markdown(f"**{n}. {p['name']}**")
        source = "Real recipe (Food.com)" if p["real"] else "Written by the AI, not a database recipe"
        st.caption(" · ".join(x for x in (f"{p['minutes']} min" if p.get("minutes") else "", source) if x))
        missing = ", ".join(m.replace("_", " ") for m in p["missing"])
        if not p["missing"]:
            st.success("You have everything", icon="✅")
        elif p["cookable"]:
            st.info(f"Cookable · you only need: {missing}", icon="🛒")
        else:
            st.warning(f"Not cookable from your inventory · missing: {missing}", icon="⚠️")
        if p.get("reason"):
            st.markdown(f"_{p['reason']}_")
        if not p["real"] and not p["steps"]:
            if st.button("Show steps", key=f"steps-{rec['method']}-{n}"):
                with st.spinner("Writing steps..."):
                    try:
                        p["steps"] = write_steps(p, rec["request"])   # saved on the pick, so it survives reruns
                    except Exception as e:
                        st.error(f"Could not write steps: {e}")
                if p.get("steps"):
                    st.rerun()   # redraw the card without the button
                st.warning("The model returned no steps. Try again.")
        with st.expander("Ingredients and steps" if p["steps"] else "Ingredients", expanded=bool(p["steps"]) and not p["real"]):
            st.write(", ".join(p["ingredients"]))
            if p["steps"] and not p["real"]:
                st.caption("Steps written by the AI on request; not a tested recipe.")
            for i, s in enumerate(p["steps"], 1):
                st.write(f"{i}. {s}")


def show_run(rec: dict) -> None:
    st.subheader(METHODS[rec["method"]])
    if rec["method"] == "pantrypal":
        f = rec.get("filters") or {}
        st.caption(f"Filters read from your request: diet {f.get('diet') or 'none'} · meal {f.get('meal') or 'any'} · "
                   f"time {f.get('max_minutes') or 'no limit'} · avoid {', '.join(f.get('exclude') or []) or 'nothing'}")
    else:
        st.caption("Cookability is checked against the ingredient list the AI wrote.")
    if not rec["picks"]:
        st.error("No recipes came back. Try rewording the request.")
    for n, p in enumerate(rec["picks"], 1):
        show_recipe(n, p, rec)
    cost = f" · US${rec['cost_usd']:.4f}" if rec.get("cost_usd") is not None else ""
    st.caption(f"{rec['model']} · {rec['latency_s']:.1f}s{cost}")


with rec_tab:
    have = sorted(inv.have())
    if not have:
        st.info("Your inventory is empty. Scan a photo in the Scan tab first.")
    else:
        st.caption(f"Using {len(have)} items from your inventory: {', '.join(h.replace('_', ' ') for h in have)}. "
                   "Salt, pepper, oil, water and sugar are always assumed.")
        request = st.text_area("What would you like to cook?", "Suggest a quick dinner for 2 people.", height=80,
                               help="Mention diet, time limit, foods to avoid or a mood, e.g. "
                                    "\"vegetarian, under 30 minutes, something spicy\".")
        mode = st.radio("Method", ["PantryPal", "LLM-only", "Compare both"], horizontal=True,
                        captions=["Real recipes, checked by code", "The AI writes recipes freely", "Side by side"])
        if st.button("Recommend recipes", type="primary", disabled=not request.strip()):
            methods = {"PantryPal": ["pantrypal"], "LLM-only": ["llm_only"], "Compare both": ["llm_only", "pantrypal"]}[mode]
            with st.spinner("Finding recipes..."):
                try:
                    with ThreadPoolExecutor(len(methods)) as pool:
                        st.session_state.recs = list(pool.map(lambda m: recommend(request, have, m), methods))
                except Exception as e:   # network or API errors: show them instead of crashing the page
                    st.session_state.recs = None
                    st.error(f"Recommendation failed: {e}")
        runs = st.session_state.get("recs")
        if runs:
            for col, rec in zip(st.columns(len(runs)), runs):
                with col:
                    show_run(rec)

# ---------------- Chat ----------------


def show_chat(text: str) -> str:
    """[[Recipe]] -> bold; the model sometimes bolds the brackets too (**[[x]]**), which would give ****x****."""
    return pretty(text).replace("****", "**")

with chat_tab:
    st.session_state.setdefault("chat", [])
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(show_chat(m["content"]))

    if prompt := st.chat_input("What can I cook for two people in 20 minutes?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            result = Assistant(inv, model=MODEL).reply(st.session_state.chat, prompt)
            st.markdown(show_chat(result["text"]))
            if not result["grounded"]:
                st.caption("Note: some recipe names could not be verified against the database.")
        st.session_state.chat += [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": result["text"]}]
        if result["recipes"]:
            st.session_state.last_recipes = result["recipes"]

# ---------------- Shopping ----------------
with shop_tab:
    st.subheader("Restock")
    low = rules.restock_list(inv.items())
    st.write(", ".join(i["name"] for i in low) if low else "Nothing below your restock levels.")

    st.subheader("For a recipe")
    recent = st.session_state.get("last_recipes", {})
    if not recent:
        st.caption("Ask the assistant for recipes first; they will show up here.")
    else:
        from core.recipes import load_index

        df = load_index()
        name = st.selectbox("Recipe", list(recent), format_func=str.title)
        rec = df[df["recipe_id"] == recent[name]].iloc[0]
        missing = rules.shopping_list(list(rec["ingredient_ids"]), set(inv.have()))
        st.write("You need: " + (", ".join(missing) if missing else "nothing, you have everything."))
