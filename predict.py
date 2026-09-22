from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from nutrition_quality import feature_columns, load_nutrition_dataset, load_scorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a Nutrition Quality Score for new food rows.")
    parser.add_argument("input_csv", nargs="+")
    parser.add_argument("--ingredient-path", default=None, help="Ingredient-level CSV for formulation features")
    parser.add_argument("--model", choices=("nutrition", "ingredient", "combined"), default="combined")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--scorer", default="nutrition_scorer.json")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    raw = load_nutrition_dataset(args.input_csv, args.ingredient_path)
    scorer = load_scorer(args.scorer)
    prepared = scorer.transform(raw)
    model = joblib.load(Path(args.model_dir) / f"xgb_{args.model}.joblib")
    columns = feature_columns(prepared, args.model)
    prepared["predicted_healthiness_score"] = model.predict(prepared[columns].astype(float)).clip(0.0, 100.0)
    prepared.to_csv(args.output, index=False)
    print(json.dumps({"rows": len(prepared), "output": str(Path(args.output).resolve()), "model": args.model}, indent=2))


if __name__ == "__main__":
    main()
