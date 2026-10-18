"""PantryPal Streamlit app: Scan | Inventory | Chat | Shopping.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import rules
from core.assistant import Assistant, pretty
from core.inventory import Inventory
from core.vlm import recognise

st.set_page_config(page_title="PantryPal", page_icon="🥕", layout="centered")


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

scan_tab, inv_tab, chat_tab, shop_tab = st.tabs(["Scan", "Inventory", "Chat", "Shopping"])

# ---------------- Scan ----------------
with scan_tab:
    location = st.radio("Where are these photos from?", ["fridge", "freezer", "pantry"], horizontal=True)
    uploads = st.file_uploader("Photos (1 to 3 angles)", type=["jpg", "jpeg", "png", "webp"],
                               accept_multiple_files=True)
    shot = st.camera_input("Or take a photo now")
    photos = list(uploads or []) + ([shot] if shot else [])

    if st.button("Recognise items", type="primary", disabled=not photos):
        with st.spinner("Reading your photos..."):
            out = recognise(photos, location_hint=location)
        if not out.items:
            st.error("The model did not return a usable item list. Try clearer or fewer photos.")
        st.session_state.scan = inv.diff_scan(out.items)
        st.caption(f"{out.model} · {out.latency_s:.1f}s · {sum(out.usage.values())} tokens")

    if st.session_state.get("scan"):
        st.write("Check the list. Untick anything wrong and fix flagged rows before adding.")
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
            rows = edited[edited["keep"]].to_dict("records")
            inv.apply_scan(rows)
            st.session_state.scan = None
            st.success(f"Saved {len(rows)} items.")
            st.rerun()

# ---------------- Inventory ----------------
with inv_tab:
    items = inv.items()
    if not items:
        st.info("Nothing here yet. Scan your fridge first.")
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

# ---------------- Chat ----------------
with chat_tab:
    st.session_state.setdefault("chat", [])
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(pretty(m["content"]))

    if prompt := st.chat_input("What can I cook for two people in 20 minutes?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            result = Assistant(inv).reply(st.session_state.chat, prompt)
            st.markdown(pretty(result["text"]))
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
