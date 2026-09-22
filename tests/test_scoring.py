import numpy as np
import pandas as pd
from nutrition_quality import NutrientScorer, add_targets, prepare_dataframe


def test_baseline_ignores_major_outlier():
    scorer = NutrientScorer().fit(pd.DataFrame({"calories_per_100g": [100, 110, 90, 105, 1000], "carbs_per_100g": [10] * 5, "protein_per_100g": [10] * 5, "fat_per_100g": [10] * 5}))
    assert 95 < scorer.baselines["calories_per_100g"] < 120


def test_penalty_matches_requested_calibration_example():
    scorer = NutrientScorer()
    penalties = scorer.penalty(np.array([1 / 3, 2 / 3, 1.0]))
    assert 45 < penalties[0] < 50
    assert 73 < penalties[1] < 78
    assert 90 < penalties[2] < 96


def test_penalty_is_exponentially_growing_and_saturating():
    scorer = NutrientScorer()
    penalties = scorer.penalty(np.array([0.01, 0.02, 0.04, 0.08, 4.0, 8.0]))
    assert penalties[1] > 2 * penalties[0] * 0.95
    assert (penalties[1:] > penalties[:-1]).all()
    assert penalties[-1] < 120


def test_baseline_scores_100_and_extremes_are_clipped():
    frame = pd.DataFrame({"calories_per_100g": [100, 10000], "carbs_per_100g": [20, 20000], "protein_per_100g": [10, 10000], "fat_per_100g": [5, 5000]})
    scorer = NutrientScorer().fit(frame.iloc[[0]])
    scores = scorer.score(frame)
    assert scores[0] == 100
    assert 0 <= scores[1] <= 100


def test_ingredient_metadata_creates_formulation_features():
    from nutrition_quality import aggregate_ingredient_metadata
    dishes = pd.DataFrame({"dish_id": ["a"], "calories": [100], "mass": [100], "carb": [10], "protein": [10], "fat": [5]})
    ingredients = pd.DataFrame({"dish_id": ["a", "a"], "ingr_name": ["rice", "sauce"], "grams": [80, 20], "calories": [280, 20], "carb": [60, 2], "protein": [6, 1], "fat": [1, 1]})
    result = aggregate_ingredient_metadata(dishes, ingredients)
    assert result.loc[0, "ingredient_count"] == 2
    assert result.loc[0, "top_ingredient_mass_fraction"] == 0.8
    assert result.loc[0, "ingredient_calories_per_100g"] == 300


def test_processing_variants_create_nonzero_flag_targets():
    from nutrition_quality import augment_processing_variants
    frame = pd.DataFrame({"calories_per_100g": [100], "carbs_per_100g": [10], "protein_per_100g": [10], "fat_per_100g": [5], "healthiness_score": [100.0], "artificial_sweetener": [0.0], "artificial_colours": [0.0], "preservatives": [0.0]})
    scorer = NutrientScorer().fit(frame)
    variants = augment_processing_variants(frame, scorer)
    assert len(variants) == 5
    assert variants["healthiness_score"].min() < 100
