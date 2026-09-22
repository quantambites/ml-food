from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

from nutrition_quality import INGREDIENT_COLUMNS, NUTRIENT_COLUMNS, NutrientScorer, feature_columns, load_scorer


def split_indices(frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    groups = frame["dish_id"] if "dish_id" in frame else pd.Series(np.arange(len(frame)), index=frame.index)
    return next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(frame, groups=groups))


def make_model(seed: int) -> XGBRegressor:
    return XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.85, colsample_bytree=0.9, objective="reg:squarederror", eval_metric="mae", random_state=seed, n_jobs=2)


def fit_model(frame: pd.DataFrame, model_name: str, target: pd.Series, indices: np.ndarray, seed: int) -> XGBRegressor:
    columns = feature_columns(frame, model_name)
    model = make_model(seed)
    model.fit(frame.iloc[indices][columns].astype(float), target.iloc[indices].astype(float), verbose=False)
    return model


def evaluate(model: XGBRegressor, frame: pd.DataFrame, model_name: str, target: pd.Series, test_idx: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    columns = feature_columns(frame, model_name)
    actual = target.iloc[test_idx].astype(float)
    predictions = np.clip(model.predict(frame.iloc[test_idx][columns].astype(float)), 0.0, 100.0)
    metrics = {"mae": float(mean_absolute_error(actual, predictions)), "rmse": float(np.sqrt(mean_squared_error(actual, predictions))), "r2": float(r2_score(actual, predictions))}
    ids = frame.iloc[test_idx]["dish_id"].to_numpy() if "dish_id" in frame else test_idx
    return metrics, pd.DataFrame({"dish_id": ids, "actual": actual.to_numpy(), "predicted": predictions})


def save_shap(model: XGBRegressor, frame: pd.DataFrame, model_name: str, output_dir: Path) -> None:
    columns = feature_columns(frame, model_name)
    sample = frame[columns].astype(float).sample(min(1000, len(frame)), random_state=0)
    values = shap.TreeExplainer(model).shap_values(sample)
    plt.figure(figsize=(9, 5))
    shap.summary_plot(values, sample, show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_{model_name}.png", dpi=180, bbox_inches="tight")
    plt.close()
    pd.DataFrame({"feature": columns, "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False).to_csv(output_dir / f"shap_{model_name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Nutrition, Ingredient, and Combined XGBoost regressors.")
    parser.add_argument("scored_csv")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--scorer", default=None, help="Optional scorer JSON used for the leakage-safe evaluation target")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.scored_csv)
    required = set(NUTRIENT_COLUMNS + INGREDIENT_COLUMNS + ("healthiness_score",))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}. Run build_targets.py first.")
    train_idx, test_idx = split_indices(frame, args.seed)
    full_scorer = load_scorer(args.scorer) if args.scorer else NutrientScorer()
    train_scorer = NutrientScorer(trim_quantile=full_scorer.trim_quantile, penalty_rate=full_scorer.penalty_rate, penalty_scale=full_scorer.penalty_scale, nutrient_weight=full_scorer.nutrient_weight, processing_weight=full_scorer.processing_weight, flag_penalties=tuple(full_scorer.flag_penalties)).fit(frame.iloc[train_idx])
    evaluation_target = pd.Series(train_scorer.score(frame), index=frame.index)
    production_target = frame["healthiness_score"].astype(float)
    summary = {}
    for model_name in ("nutrition", "ingredient", "combined"):
        evaluation_model = fit_model(frame, model_name, evaluation_target, train_idx, args.seed)
        metrics, predictions = evaluate(evaluation_model, frame, model_name, evaluation_target, test_idx)
        production_model = fit_model(frame, model_name, production_target, np.arange(len(frame)), args.seed)
        joblib.dump(production_model, output_dir / f"xgb_{model_name}.joblib")
        predictions.to_csv(output_dir / f"predictions_{model_name}.csv", index=False)
        save_shap(production_model, frame, model_name, output_dir)
        summary[model_name] = metrics
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
