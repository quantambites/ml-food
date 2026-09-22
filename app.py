"""Streamlit UI: score a food (rule-based + XGBoost prediction), batch CSV, real products."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st
import joblib
from nutrition_quality import FLAG_COLUMNS, INGREDIENT_COLUMNS, NUTRIENT_COLUMNS, feature_columns, load_scorer, prepare_dataframe

BASE = Path(__file__).parent
SCORER = BASE / "data" / "output" / "nutrition_scorer.json"
MODEL_DIR = BASE / "models"
LABELS = {"artificial_sweetener": "Artificial sweetener", "artificial_colours": "Artificial colours", "preservatives": "Preservatives"}
NUTRIENT_LABELS = {"calories_per_100g": "Calories / 100g", "carbs_per_100g": "Carbs / 100g", "protein_per_100g": "Protein / 100g", "fat_per_100g": "Fat / 100g"}
UNIT_DEFAULTS = {"calories_per_100g": 120.0, "carbs_per_100g": 10.0, "protein_per_100g": 7.7, "fat_per_100g": 6.6}
EXAMPLES = (
    {"key": "avg", "name": "Average dish (at baseline)", "calories_per_100g": 126.0, "carbs_per_100g": 10.0, "protein_per_100g": 7.7, "fat_per_100g": 6.6, "ingredients": "mixed plate", "artificial_sweetener": 0, "artificial_colours": 0, "preservatives": 0, "note": "Sits on the trimmed baseline, so it should score near 100 (Excellent)."},
    {"key": "chicken", "name": "Chicken breast (cooked, skinless)", "calories_per_100g": 165.0, "carbs_per_100g": 0.0, "protein_per_100g": 31.0, "fat_per_100g": 3.6, "ingredients": "chicken breast", "artificial_sweetener": 0, "artificial_colours": 0, "preservatives": 0, "note": "High protein, low fat, no additives. Only mild calorie excess over baseline."},
    {"key": "salad", "name": "Garden salad (raw vegetables)", "calories_per_100g": 20.0, "carbs_per_100g": 4.0, "protein_per_100g": 1.5, "fat_per_100g": 0.5, "ingredients": "lettuce, tomato, cucumber, olive oil", "artificial_sweetener": 0, "artificial_colours": 0, "preservatives": 0, "note": "Very low calorie/fat; protein below baseline takes a small penalty."},
    {"key": "rice", "name": "White rice (cooked)", "calories_per_100g": 130.0, "carbs_per_100g": 28.0, "protein_per_100g": 2.7, "fat_per_100g": 0.3, "ingredients": "white rice, water", "artificial_sweetener": 0, "artificial_colours": 0, "preservatives": 0, "note": "Carb and protein deviate from baseline in both unhealthy directions."},
    {"key": "cake", "name": "Chocolate cake (sliced)", "calories_per_100g": 430.0, "carbs_per_100g": 58.0, "protein_per_100g": 6.0, "fat_per_100g": 22.0, "ingredients": "flour, sugar, cocoa, butter, eggs", "artificial_sweetener": 0, "artificial_colours": 0, "preservatives": 0, "note": "High calories, carbs and fat: large exponential deductions, no additives."},
    {"key": "cola", "name": "Coca-Cola Original", "calories_per_100g": 42.0, "carbs_per_100g": 10.6, "protein_per_100g": 0.0, "fat_per_100g": 0.0, "ingredients": "carbonated water, sugar, caramel color, phosphoric acid, caffeine", "artificial_sweetener": 0, "artificial_colours": 1, "preservatives": 0, "note": "Low calorie but zero protein (penalised) plus a colour flag."},
    {"key": "dietcola", "name": "Diet cola (artificial sweetener)", "calories_per_100g": 0.5, "carbs_per_100g": 0.0, "protein_per_100g": 0.0, "fat_per_100g": 0.0, "ingredients": "carbonated water, aspartame, acesulfame potassium, caffeine", "artificial_sweetener": 1, "artificial_colours": 0, "preservatives": 0, "note": "Near-zero calories; the artificial-sweetener flag adds a processing penalty."},
    {"key": "nutella", "name": "Nutella hazelnut spread", "calories_per_100g": 539.0, "carbs_per_100g": 57.5, "protein_per_100g": 6.3, "fat_per_100g": 30.9, "ingredients": "sugar, palm oil, hazelnuts, milk powder, cocoa, lecithin, vanillin", "artificial_sweetener": 0, "artificial_colours": 0, "preservatives": 0, "note": "Real commercial product, 4.3x the average calorie density: floors at 0 (Very Poor)."},
)


@st.cache_resource
def load_assets() -> tuple:
    scorer = load_scorer(SCORER)
    models = {name: joblib.load(MODEL_DIR / f"xgb_{name}.joblib") for name in ("nutrition", "ingredient", "combined")}
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    return scorer, models, metrics


def score_row(values: dict, model: str, models: dict, scorer) -> dict:
    frame = pd.DataFrame([{**values, **{column: (None if values[column] is None else float(values[column])) for column in NUTRIENT_COLUMNS}}])
    prepared = prepare_dataframe(frame)
    for column in NUTRIENT_COLUMNS:
        value = float(prepared[column].iloc[0])
        prepared[column] = UNIT_DEFAULTS[column] if pd.isna(value) else value
    prepared = prepared.fillna(0.0)
    scored = scorer.transform(prepared)
    row = scored.iloc[0]
    prediction = models[model].predict(scored[feature_columns(scored, model)].astype(float)).clip(0.0, 100.0)[0]
    return {**row.to_dict(), "predicted_healthiness_score": float(prediction)}


def result_table(result: dict) -> pd.DataFrame:
    rows = [("Rule-based score (target)", float(result["healthiness_score"]), str(result["score_category"]))]
    for column in NUTRIENT_COLUMNS:
        rows.append((NUTRIENT_LABELS[column] + " deviation penalty", float(result[f"{column}_penalty"]), f"relative +{float(result[f'{column}_relative_deviation'])*100:.0f}%"))
    for column in FLAG_COLUMNS:
        rows.append((LABELS[column] + " penalty", float(result[column]) * 5.0, "flagged" if float(result[column]) else "clean"))
    rows.append(("Processing penalty total", float(result["processing_penalty"]), ""))
    return pd.DataFrame(rows, columns=["Component", "Points", "Note"])


st.set_page_config(page_title="ML Food Nutrition Quality", layout="wide")
st.title("ML-Based Food Nutrition Quality Assessment")
scorer, models, metrics = load_assets()
st.caption("0-100 Nutrition Quality Score | categories: 80-100 Excellent, 65-79 Good, 50-64 Moderate, 30-49 Poor, 0-29 Very Poor | baselines (trimmed mean per 100g): " + ", ".join(f"{k.split('_')[0]} {v:.2f}" for k, v in scorer.baselines.items()))

tab_ex, tab_score, tab_batch, tab_real, tab_models = st.tabs(["Examples (run)", "Score a food", "Batch CSV", "Real products", "Models & SHAP"])


def render_result(display: dict, display_model: str, display_name: str) -> None:
    st.subheader(display_name or "Food")
    st.metric("Rule-based score", f"{float(display['healthiness_score']):.1f} / 100", display["score_category"])
    st.metric(f"XGBoost prediction ({display_model})", f"{float(display['predicted_healthiness_score']):.1f} / 100", delta=None)
    st.dataframe(result_table(display), hide_index=True)


with tab_ex:
    st.markdown("One-click examples. Click a row to run the full pipeline on that food: feature preparation, rule-based scoring, and the XGBoost prediction, shown with the per-nutrient penalty breakdown.")
    example = next((example for example in EXAMPLES if st.button(example["name"], key=f"example_{example['key']}")), None)
    if example:
        st.session_state["example_payload"] = (score_row({**example, "dish_id": "example"}, "combined", models, scorer), "combined", example["name"], example["note"])
    payload = st.session_state.get("example_payload")
    if payload:
        display, display_model, display_name, note = payload
        st.caption(note)
        render_result(display, display_model, display_name)
    summary = []
    for example in EXAMPLES:
        rule = scorer.transform(prepare_dataframe(pd.DataFrame([{**example, "dish_id": "summary"}]))).iloc[0]
        summary.append({"Example": example["name"], "Cal/100g": example["calories_per_100g"], "Carbs/100g": example["carbs_per_100g"], "Protein/100g": example["protein_per_100g"], "Fat/100g": example["fat_per_100g"], "Rule score": round(float(rule["healthiness_score"]), 1), "Category": rule["score_category"]})
    st.dataframe(pd.DataFrame(summary), hide_index=True)
    st.caption("Rule scores in this table are computed live with the same scorer used for the one-click runs above.")

with tab_score:
    left, right = st.columns([3, 2])
    with left:
        name = st.text_input("Food name", "My food")
        c1, c2, c3, c4 = st.columns(4)
        values = {column: (c1, c2, c3, c4)[i].number_input(NUTRIENT_LABELS[column], min_value=0.0, value=UNIT_DEFAULTS[column], format="%.2f") for i, column in enumerate(NUTRIENT_COLUMNS)}
        ingredients = st.text_area("Ingredients (comma separated)", "")
        st.subheader("Processing flags (manual)")
        flags = {column: st.checkbox(LABELS[column]) for column in FLAG_COLUMNS}
        model = st.selectbox("Prediction model", ["combined", "nutrition", "ingredient"], index=0)
        run = st.button("Score food", type="primary")
    with right:
        if run:
            st.session_state["score_payload"] = (score_row({**values, "dish_id": "ui", "ingredients": ingredients, **{column: int(flag) for column, flag in flags.items()}}, model, models, scorer), model, name)
        payload = st.session_state.get("score_payload")
        if payload:
            display, display_model, display_name = payload
            render_result(display, display_model, display_name)
        else:
            st.info("Fill the form and press Score food, or try the one-click examples on the first tab.")

with tab_batch:
    st.markdown("Upload a CSV with one food per row. Columns: `calories_per_100g, carbs_per_100g, protein_per_100g, fat_per_100g, ingredients` (optional `artificial_sweetener, artificial_colours, preservatives`, `name`). Missing flags default to 0.")
    upload = st.file_uploader("CSV", type=["csv"])
    if upload:
        frame = pd.read_csv(upload)
        frame["dish_id"] = [f"batch_{i}" for i in range(len(frame))]
        if "name" not in frame:
            frame["name"] = [f"batch_{i}" for i in range(len(frame))]
        for column in FLAG_COLUMNS:
            if column not in frame:
                frame[column] = 0
        prepared = prepare_dataframe(frame)
        scored = scorer.transform(prepared)
        for model_name in ("combined",):
            scored[f"predicted_{model_name}"] = models[model_name].predict(scored[feature_columns(scored, model_name)].astype(float)).clip(0.0, 100.0)
        display = scored[["name", "healthiness_score", "score_category", "predicted_combined", "artificial_sweetener", "artificial_colours", "preservatives"]].copy()
        display["healthiness_score"] = display["healthiness_score"].round(1)
        display["predicted_combined"] = display["predicted_combined"].round(1)
        st.dataframe(display.sort_values("healthiness_score"), hide_index=True)
        st.download_button("Download results", display.to_csv(index=False).encode(), "batch_scores.csv", "text/csv")

with tab_real:
    st.markdown("Real commercial products from public nutrition/ingredient sources (per 100 g), scored by this system on 2026-09-22.")
    if (BASE / "data" / "input" / "real_products.csv").exists():
        real = pd.read_csv(BASE / "data" / "input" / "real_products.csv")
        prepared = prepare_dataframe(real)
        scored = scorer.transform(prepared)
        scored["predicted_combined"] = models["combined"].predict(scored[feature_columns(scored, "combined")].astype(float)).clip(0.0, 100.0)
        scored["predicted_nutrition"] = models["nutrition"].predict(scored[feature_columns(scored, "nutrition")].astype(float)).clip(0.0, 100.0)
        display = scored[["name", "calories_per_100g", "carbs_per_100g", "protein_per_100g", "fat_per_100g", "healthiness_score", "score_category", "predicted_combined", "predicted_nutrition", "artificial_sweetener", "artificial_colours", "preservatives"]].copy()
        display["healthiness_score"] = display["healthiness_score"].round(1)
        display["predicted_combined"] = display["predicted_combined"].round(1)
        display["predicted_nutrition"] = display["predicted_nutrition"].round(1)
        st.dataframe(display.sort_values("healthiness_score"), hide_index=True)
        st.caption("Sources: nutella.com (per 100 g), coca-cola.com + Open Food Facts (per 100 ml = per 100 g), Kellogg's Froot Loops US (foodstruct/myfooddata, per 100 g), NutriDB Doritos Nacho Cheese (per 100 g), chicken breast USDA reference.")
    else:
        st.warning("data/input/real_products.csv not found.")

with tab_models:
    st.markdown("Dish-grouped 80/20 holdout on Nutrition5k (4,768 dishes, baselines refit on train dishes only to prevent leakage).")
    st.dataframe(pd.DataFrame(metrics).T, hide_index=True)
    for model_name in ("nutrition", "ingredient", "combined"):
        shap_csv = MODEL_DIR / f"shap_{model_name}.csv"
        if shap_csv.exists():
            col, side = st.columns([2, 3])
            col.markdown(f"**SHAP - {model_name}**")
            col.dataframe(pd.read_csv(shap_csv), hide_index=True)
            png = MODEL_DIR / f"shap_{model_name}.png"
            if png.exists():
                side.image(png)
