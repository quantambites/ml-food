from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

NUTRIENT_COLUMNS = ("calories_per_100g", "carbs_per_100g", "protein_per_100g", "fat_per_100g")
INGREDIENT_NUTRIENT_COLUMNS = ("ingredient_calories_per_100g", "ingredient_carbs_per_100g", "ingredient_protein_per_100g", "ingredient_fat_per_100g")
INGREDIENT_FORMULATION_COLUMNS = ("ingredient_count", "top_ingredient_mass_fraction", "top3_ingredient_mass_fraction", "ingredient_entropy")
FLAG_COLUMNS = ("artificial_sweetener", "artificial_colours", "preservatives")
INGREDIENT_COLUMNS = INGREDIENT_NUTRIENT_COLUMNS + INGREDIENT_FORMULATION_COLUMNS + FLAG_COLUMNS
# Unhealthy direction per nutrient: calories/carbs/fat penalise values ABOVE baseline, protein penalises values BELOW baseline.
BAD_DIRECTION = {"calories_per_100g": "above", "carbs_per_100g": "above", "protein_per_100g": "below", "fat_per_100g": "above"}


def _canonical(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _first_column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    columns = {_canonical(column): column for column in frame.columns}
    for name in names:
        if _canonical(name) in columns:
            return columns[_canonical(name)]
    return None


def _numeric(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(lower=0.0)


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    aliases = {
        "dish_id": ("dish_id", "id", "food_id", "food"),
        "total_mass": ("total_mass", "mass", "weight", "total_weight"),
        "total_calories": ("total_calories", "calories", "calorie", "energy"),
        "total_carb": ("total_carb", "carb", "carbs", "carbohydrates", "total_carbohydrate"),
        "total_protein": ("total_protein", "protein"),
        "total_fat": ("total_fat", "fat"),
        "ingredients": ("ingredients", "ingredient", "ingredient_names", "ingredient_list"),
    }
    renamed = {}
    for target, candidates in aliases.items():
        source = _first_column(frame, candidates)
        if source is not None:
            renamed[source] = target
    frame = frame.rename(columns=renamed)
    if "dish_id" not in frame:
        frame.insert(0, "dish_id", [f"row_{i}" for i in range(len(frame))])
    if "total_mass" not in frame:
        frame["total_mass"] = 100.0
    for name in ("total_calories", "total_carb", "total_protein", "total_fat"):
        if name not in frame:
            frame[name] = 0.0
    frame["total_mass"] = _numeric(frame, "total_mass").replace(0.0, 100.0)
    for name in ("total_calories", "total_carb", "total_protein", "total_fat"):
        frame[name] = _numeric(frame, name)
    return frame


def _extract_flag(ingredients: object, patterns: tuple[str, ...]) -> int:
    text = str(ingredients).lower() if ingredients is not None else ""
    return int(any(re.search(pattern, text) for pattern in patterns))


def _ingredient_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "ingredients" not in frame:
        frame["ingredients"] = ""
    text = frame["ingredients"].fillna("").astype(str)
    if "ingredient_count" not in frame:
        frame["ingredient_count"] = text.map(lambda value: len([item for item in re.split(r"[,;|]", value) if item.strip()]))
    for column in INGREDIENT_NUTRIENT_COLUMNS + INGREDIENT_FORMULATION_COLUMNS:
        if column not in frame:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(lower=0.0)
    return frame


def aggregate_ingredient_metadata(dishes: pd.DataFrame, ingredients: pd.DataFrame) -> pd.DataFrame:
    dishes = _normalise_columns(dishes)
    ingredients = ingredients.copy()
    dish_id = _first_column(ingredients, ("dish_id",))
    name = _first_column(ingredients, ("ingr_name", "ingredient_name", "ingredient"))
    grams_column = _first_column(ingredients, ("grams", "ingr_grams", "mass"))
    if dish_id is None or grams_column is None:
        raise ValueError("Ingredient rows require dish_id and grams columns")
    rows = []
    for current_id, group in ingredients.groupby(dish_id, sort=False):
        grams = _numeric(group, grams_column)
        total_grams = max(float(grams.sum()), 1e-9)
        fractions = (grams / total_grams).sort_values(ascending=False)
        row = {"dish_id": current_id, "ingredient_count": int(len(group)), "top_ingredient_mass_fraction": float(fractions.iloc[0]) if len(fractions) else 0.0, "top3_ingredient_mass_fraction": float(fractions.iloc[:3].sum()), "ingredient_entropy": float(-(fractions * np.log2(fractions.clip(lower=1e-12))).sum()) if len(fractions) else 0.0}
        for output, candidates in (("ingredient_calories_per_100g", ("calories", "calorie")), ("ingredient_carbs_per_100g", ("carb", "carbs")), ("ingredient_protein_per_100g", ("protein",)), ("ingredient_fat_per_100g", ("fat",))):
            source = _first_column(group, candidates)
            row[output] = float(_numeric(group, source).sum() / total_grams * 100.0)
        if name is not None:
            row["ingredients"] = ", ".join(group[name].fillna("").astype(str))
        rows.append(row)
    return dishes.merge(pd.DataFrame(rows), on="dish_id", how="left")


def prepare_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _normalise_columns(frame)
    if all(column in frame for column in NUTRIENT_COLUMNS):
        for column in NUTRIENT_COLUMNS:
            frame[column] = _numeric(frame, column)
    else:
        mass = frame["total_mass"].replace(0.0, 100.0)
        frame["calories_per_100g"] = frame["total_calories"] / mass * 100.0
        frame["carbs_per_100g"] = frame["total_carb"] / mass * 100.0
        frame["protein_per_100g"] = frame["total_protein"] / mass * 100.0
        frame["fat_per_100g"] = frame["total_fat"] / mass * 100.0
    frame = _ingredient_features(frame)
    ingredients = frame["ingredients"].fillna("").astype(str)
    patterns = {
        "artificial_sweetener": (r"aspartame", r"sucralose", r"saccharin", r"acesulfame", r"stevia", r"sweetener"),
        "artificial_colours": (r"colour", r"color", r"dye", r"red\s*40", r"yellow\s*5", r"blue\s*1"),
        "preservatives": (r"preserv", r"benzoate", r"nitrite", r"nitrate", r"sorbate", r"sulfite"),
    }
    for column, column_patterns in patterns.items():
        if column not in frame:
            frame[column] = ingredients.map(lambda value, selected=column_patterns: _extract_flag(value, selected))
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    return frame


@dataclass
class NutrientScorer:
    trim_quantile: float = 0.01
    penalty_rate: float = 1.5
    penalty_scale: float = 120.0
    nutrient_weight: float = 1.0
    processing_weight: float = 1.0
    flag_penalties: tuple[float, float, float] = (5.0, 5.0, 5.0)
    baselines: dict[str, float] | None = None

    def fit(self, frame: pd.DataFrame) -> "NutrientScorer":
        if not 0.0 <= self.trim_quantile < 0.5:
            raise ValueError("trim_quantile must be in [0, 0.5)")
        clean = prepare_dataframe(frame)
        self.baselines = {}
        for column in NUTRIENT_COLUMNS:
            values = clean[column].replace([np.inf, -np.inf], np.nan).dropna()
            values = values[values >= 0]
            if not len(values):
                self.baselines[column] = 1.0
                continue
            lower, upper = values.quantile([self.trim_quantile, 1.0 - self.trim_quantile])
            trimmed = values[values.between(lower, upper)]
            baseline = float(trimmed.mean() if len(trimmed) else values.mean())
            self.baselines[column] = max(baseline, 1e-9)
        return self

    def _check_fit(self) -> None:
        if self.baselines is None:
            raise RuntimeError("NutrientScorer.fit must be called before scoring")

    def penalty(self, relative_deviation: np.ndarray) -> np.ndarray:
        """Capped exponential: grows exponentially with deviation, saturates at 120 points per nutrient so the 0-100 scale stays discriminative."""
        exponent = np.minimum(self.penalty_rate * np.clip(relative_deviation, 0.0, 32.0), 50.0)
        return self.penalty_scale * (1.0 - np.exp(-exponent))

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._check_fit()
        clean = prepare_dataframe(frame)
        penalties = []
        for column in NUTRIENT_COLUMNS:
            baseline = self.baselines[column]
            values = clean[column].to_numpy(dtype=float)
            deviation = np.maximum((values - baseline) / baseline, 0.0) if BAD_DIRECTION[column] == "above" else np.maximum((baseline - values) / baseline, 0.0)
            penalty = self.penalty(deviation)
            clean[f"{column}_relative_deviation"] = deviation
            clean[f"{column}_penalty"] = penalty
            penalties.append(penalty)
        clean["nutrient_penalty"] = np.sum(penalties, axis=0)
        clean["processing_penalty"] = sum(clean[column].to_numpy(dtype=float) * penalty for column, penalty in zip(FLAG_COLUMNS, self.flag_penalties))
        clean["healthiness_score"] = np.clip(100.0 - self.nutrient_weight * clean["nutrient_penalty"] - self.processing_weight * clean["processing_penalty"], 0.0, 100.0)
        clean["nutrition_quality_score"] = clean["healthiness_score"]
        clean["score_category"] = np.select([clean["healthiness_score"] >= 80, clean["healthiness_score"] >= 65, clean["healthiness_score"] >= 50, clean["healthiness_score"] >= 30], ["Excellent", "Good", "Moderate", "Poor"], default="Very Poor")
        return clean

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.transform(frame)["healthiness_score"].to_numpy()


def add_targets(frame: pd.DataFrame, scorer: NutrientScorer) -> pd.DataFrame:
    return scorer.transform(frame)


def _score_category(scores: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    return np.select([values >= 80, values >= 65, values >= 50, values >= 30], ["Excellent", "Good", "Moderate", "Poor"], default="Very Poor")


def augment_processing_variants(frame: pd.DataFrame, scorer: NutrientScorer) -> pd.DataFrame:
    scorer._check_fit()
    base = frame.copy()
    if "processing_penalty" not in base:
        base["processing_penalty"] = sum(base[column].to_numpy(dtype=float) * penalty for column, penalty in zip(FLAG_COLUMNS, scorer.flag_penalties))
    if "healthiness_score" not in base:
        base["healthiness_score"] = np.clip(100.0 - base["processing_penalty"].to_numpy(dtype=float), 0.0, 100.0)
    base["processing_variant"] = 0
    variants = [base]
    for column, penalty in zip(FLAG_COLUMNS, scorer.flag_penalties):
        variant = base.copy()
        added = 1.0 - variant[column].to_numpy(dtype=float)
        variant[column] = 1.0
        variant["processing_penalty"] = base["processing_penalty"].to_numpy(dtype=float) + added * penalty
        variant["healthiness_score"] = np.clip(base["healthiness_score"].to_numpy(dtype=float) - scorer.processing_weight * added * penalty, 0.0, 100.0)
        variant["nutrition_quality_score"] = variant["healthiness_score"]
        variant["score_category"] = _score_category(variant["healthiness_score"])
        variant["processing_variant"] = 1
        variants.append(variant)
    all_flags = base.copy()
    added = sum((1.0 - all_flags[column].to_numpy(dtype=float)) * penalty for column, penalty in zip(FLAG_COLUMNS, scorer.flag_penalties))
    all_flags.loc[:, list(FLAG_COLUMNS)] = 1.0
    all_flags["processing_penalty"] = base["processing_penalty"].to_numpy(dtype=float) + added
    all_flags["healthiness_score"] = np.clip(base["healthiness_score"].to_numpy(dtype=float) - scorer.processing_weight * added, 0.0, 100.0)
    all_flags["nutrition_quality_score"] = all_flags["healthiness_score"]
    all_flags["score_category"] = _score_category(all_flags["healthiness_score"])
    all_flags["processing_variant"] = 1
    variants.append(all_flags)
    return pd.concat(variants, ignore_index=True)


def feature_columns(frame: pd.DataFrame, model: str) -> list[str]:
    nutrition = list(NUTRIENT_COLUMNS)
    ingredient = list(INGREDIENT_COLUMNS)
    model = model.lower()
    if model == "nutrition":
        return nutrition
    if model == "ingredient":
        return ingredient
    if model == "combined":
        return nutrition + ingredient
    raise ValueError("model must be nutrition, ingredient, or combined")


def load_csvs(paths: list[str | Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError("No CSV files were supplied")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def load_nutrition_dataset(nutrition_paths: list[str | Path], ingredient_path: str | Path | None = None) -> pd.DataFrame:
    dishes = load_csvs(nutrition_paths)
    return aggregate_ingredient_metadata(dishes, pd.read_csv(ingredient_path)) if ingredient_path else dishes


def save_scorer(scorer: NutrientScorer, path: str | Path) -> None:
    payload = {"trim_quantile": scorer.trim_quantile, "penalty_rate": scorer.penalty_rate, "penalty_scale": scorer.penalty_scale, "nutrient_weight": scorer.nutrient_weight, "processing_weight": scorer.processing_weight, "flag_penalties": list(scorer.flag_penalties), "baselines": scorer.baselines}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_scorer(path: str | Path) -> NutrientScorer:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return NutrientScorer(**payload)
