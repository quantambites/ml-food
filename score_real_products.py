from __future__ import annotations
from pathlib import Path
import pandas as pd
from nutrition_quality import feature_columns, load_scorer, prepare_dataframe
import joblib

BASE = Path(__file__).parent
INPUT = BASE / "data" / "input" / "real_products.csv"
OUTPUT = BASE / "data" / "output" / "real_products_scores.csv"


def main() -> None:
    scorer = load_scorer(BASE / "data" / "output" / "nutrition_scorer.json")
    models = {name: joblib.load(BASE / "models" / f"xgb_{name}.joblib") for name in ("nutrition", "ingredient", "combined")}
    frame = pd.read_csv(INPUT)
    prepared = prepare_dataframe(frame)
    scored = scorer.transform(prepared)
    for name in ("nutrition", "ingredient", "combined"):
        scored[f"predicted_{name}"] = models[name].predict(scored[feature_columns(scored, name)].astype(float)).clip(0.0, 100.0)
    columns = ["name", "calories_per_100g", "carbs_per_100g", "protein_per_100g", "fat_per_100g", "artificial_sweetener", "artificial_colours", "preservatives", "nutrient_penalty", "processing_penalty", "healthiness_score", "score_category", "predicted_nutrition", "predicted_ingredient", "predicted_combined"]
    display = scored[columns].copy()
    for column in ("nutrient_penalty", "processing_penalty", "healthiness_score", "predicted_nutrition", "predicted_ingredient", "predicted_combined"):
        display[column] = display[column].round(1)
    display = display.sort_values("healthiness_score")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    display.to_csv(OUTPUT, index=False)
    print(display.to_string(index=False))
    print(f"output={OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
