from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nutrition_quality import NutrientScorer, add_targets, augment_processing_variants, load_nutrition_dataset, save_scorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Nutrition5k healthiness targets from CSV metadata.")
    parser.add_argument("csv", nargs="+", help="One or more dish nutrition CSV files")
    parser.add_argument("--output", default="nutrition5k_scored.csv")
    parser.add_argument("--scorer-output", default="nutrition_scorer.json")
    parser.add_argument("--trim-quantile", type=float, default=0.01)
    parser.add_argument("--ingredient-path", default=None, help="Ingredient-level CSV with dish_id, ingr_name, grams, calories, fat, carb, protein")
    parser.add_argument("--augment-processing-flags", action="store_true", help="Add flag-on variants because Nutrition5k has no additive labels")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.scorer_output).parent.mkdir(parents=True, exist_ok=True)
    raw = load_nutrition_dataset(args.csv, args.ingredient_path)
    scorer = NutrientScorer(trim_quantile=args.trim_quantile).fit(raw)
    scored = add_targets(raw, scorer)
    if args.augment_processing_flags:
        scored = augment_processing_variants(scored, scorer)
    scored.to_csv(args.output, index=False)
    save_scorer(scorer, args.scorer_output)
    print(f"rows={len(scored)}")
    print(f"output={Path(args.output).resolve()}")
    print(f"scorer={Path(args.scorer_output).resolve()}")
    print("baselines=" + ", ".join(f"{key}={value:.4f}" for key, value in scorer.baselines.items()))
    print(f"score_mean={scored['healthiness_score'].mean():.4f}")


if __name__ == "__main__":
    main()
