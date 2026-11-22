"""
Snack Check — Streamlit front end.

Collects a per-100g nutrition label, POSTs it to the deployed FastAPI /predict endpoint,
and shows whether the snack is likely ultra-processed (NOVA 4).

The API URL comes from the API_URL environment variable (set it as a Space secret/variable
on Hugging Face). Run locally:  API_URL=http://localhost:8080 streamlit run frontend/app.py
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8080").rstrip("/")

FIELDS = [  # (key, label, unit, max, step)
    ("energy_kcal", "Energy", "kcal", 900.0, 1.0),
    ("fat", "Fat", "g", 100.0, 0.1),
    ("saturated_fat", "  of which saturates", "g", 100.0, 0.1),
    ("carbohydrates", "Carbohydrate", "g", 100.0, 0.1),
    ("sugars", "  of which sugars", "g", 100.0, 0.1),
    ("fiber", "Fibre", "g", 100.0, 0.1),
    ("proteins", "Protein", "g", 100.0, 0.1),
    ("salt", "Salt", "g", 100.0, 0.01),
]

# Approximate per-100g values for typical products, for one-click demo fill.
EXAMPLES = {
    "🍫 Chocolate bar": dict(energy_kcal=535, fat=30, saturated_fat=18, carbohydrates=57, sugars=52,
                             fiber=2.5, proteins=7, salt=0.2, additives_n=3),
    "🥔 Potato chips": dict(energy_kcal=536, fat=34, saturated_fat=3, carbohydrates=53, sugars=0.5,
                            fiber=4.5, proteins=6.5, salt=1.3, additives_n=0),
    "🍪 Sandwich cookies": dict(energy_kcal=480, fat=20, saturated_fat=6, carbohydrates=69, sugars=38,
                                fiber=3, proteins=5, salt=0.8, additives_n=4),
    "🥜 Plain roasted peanuts": dict(energy_kcal=600, fat=50, saturated_fat=7, carbohydrates=12, sugars=4,
                                     fiber=8, proteins=26, salt=0.0, additives_n=0),
    "🥣 Plain oats": dict(energy_kcal=375, fat=8, saturated_fat=1.5, carbohydrates=60, sugars=1,
                          fiber=10, proteins=13, salt=0.0, additives_n=0),
}

st.set_page_config(page_title="Snack Check", page_icon="🥨", layout="centered")

# ---------- state ----------
defaults = EXAMPLES["🍪 Sandwich cookies"]
for k, v in defaults.items():
    st.session_state.setdefault(k, float(v) if k != "additives_n" else int(v))


def fill(example: dict) -> None:
    for k, v in example.items():
        st.session_state[k] = float(v) if k != "additives_n" else int(v)


# ---------- header ----------
st.title("🥨 Snack Check")
st.caption("Is this snack **ultra-processed** (NOVA group 4)? Type in the nutrition label from the back "
           "of the package and a model trained on Open Food Facts will guess.")

st.write("**Try an example:**")
cols = st.columns(len(EXAMPLES))
for col, (name, ex) in zip(cols, EXAMPLES.items()):
    col.button(name, on_click=fill, args=(ex,), use_container_width=True)

# ---------- nutrition-label style form ----------
with st.container(border=True):
    st.markdown("#### Nutrition Facts &nbsp; <span style='font-weight:400;font-size:0.9em'>per 100 g</span>",
                unsafe_allow_html=True)
    for key, label, unit, mx, step in FIELDS:
        c1, c2 = st.columns([3, 2])
        c1.markdown(f"{'&nbsp;&nbsp;&nbsp;' if label.startswith('  ') else '**'}{label.strip()}"
                    f"{'' if label.startswith('  ') else '**'}", unsafe_allow_html=True)
        c2.number_input(f"{label.strip()} ({unit})", min_value=0.0, max_value=mx, step=step,
                        key=key, label_visibility="collapsed", format="%.2f" if step < 0.1 else "%.1f")
    st.divider()
    c1, c2 = st.columns([3, 2])
    c1.markdown("**Number of additives**  \n<small>count the E-numbers in the ingredient list</small>",
                unsafe_allow_html=True)
    c2.number_input("Additives", min_value=0, max_value=100, step=1, key="additives_n",
                    label_visibility="collapsed")

payload = {k: st.session_state[k] for k, *_ in FIELDS} | {"additives_n": int(st.session_state["additives_n"])}

# Client-side sanity checks (the API enforces the same rules)
problems = []
if payload["saturated_fat"] > payload["fat"]:
    problems.append("Saturates can't exceed total fat.")
if payload["sugars"] > payload["carbohydrates"]:
    problems.append("Sugars can't exceed total carbohydrate.")
for p in problems:
    st.warning(p)

# ---------- predict ----------
if st.button("Check this snack", type="primary", disabled=bool(problems), use_container_width=True):
    try:
        with st.spinner("Asking the model..."):
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)  # cold start can take ~10s
        r.raise_for_status()
        res = r.json()
    except requests.HTTPError:
        st.error(f"The API rejected the input ({r.status_code}): {r.text}")
        st.stop()
    except requests.RequestException as e:
        st.error(f"Couldn't reach the API at {API_URL}. ({e.__class__.__name__})")
        st.stop()

    p = res["probability"]
    if res["ultra_processed"] == 1:
        st.error(f"### 🏭 Likely ultra-processed\nModel confidence it is NOVA 4: **{p:.0%}**")
    else:
        st.success(f"### 🌾 Probably not ultra-processed\nModel confidence it is NOVA 4: **{p:.0%}**")
    st.progress(p, text=f"P(ultra-processed) = {p:.2f}")

    with st.expander("What does this mean?"):
        st.markdown(
            "- **NOVA** groups foods by how processed they are: 1 = unprocessed, 2 = culinary ingredients, "
            "3 = processed, 4 = ultra-processed (industrial formulations, often with additives).\n"
            "- The model only sees nine numbers from the label. Real NOVA classification depends on the "
            "**ingredient list**, so the model can be wrong, e.g. a product with no additives but "
            "industrial ingredients.\n"
            "- The additive count is usually the strongest signal."
        )

# ---------- footer ----------
st.divider()
st.caption(
    "⚠️ Educational demo, not nutrition or medical advice. Labels come from crowdsourced "
    "[Open Food Facts](https://world.openfoodfacts.org) data (ODbL), which has errors and over-represents "
    "European products. NOVA itself is a debated classification.  \n"
    f"API: `{API_URL}` · [API docs]({API_URL}/docs)"
)
