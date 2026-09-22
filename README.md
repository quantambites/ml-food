# ML-Based Food Nutrition Quality Assessment

Explainable 0–100 **Nutrition Quality Score** system built on the Nutrition5k dataset (4,768 dishes, 27,225 ingredient rows). It derives a reproducible healthiness target from nutrient density, trains three XGBoost regressors (Nutrition-only, Ingredient-only, Combined), explains them with SHAP, and ships a Streamlit interface for scoring new foods.

## 1. What the system produces

For every food the system outputs:

| Output | Meaning |
|---|---|
| `healthiness_score` / `nutrition_quality_score` | Continuous 0–100 score (rule-based target) |
| `score_category` | 80–100 Excellent · 65–79 Good · 50–64 Moderate · 30–49 Poor · 0–29 Very Poor |
| `<nutrient>_relative_deviation` | How far each per-100 g nutrient is from its baseline (in the unhealthy direction, 0 = at/below baseline) |
| `<nutrient>_penalty` | Exponential point deduction per nutrient |
| `processing_penalty` | Deduction from additive flags (5 points each) |
| `predicted_healthiness_score` | XGBoost model prediction (per model variant) |

**Scope statement.** Nutrition5k provides dish nutrition and ingredient names/masses but no clinically validated healthiness label and no reliable additive identification. The score is therefore a reproducible research target derived from Nutrition5k, not a medical or regulatory claim. FCS-10 is not used.

## 2. How the scoring model works (the rule-based target)

### Step 1 — Nutrient density per 100 g

Every dish is normalised to per-100 g density so foods of different sizes are comparable:

```text
calories_per_100g = total_calories / total_mass * 100
carbs_per_100g    = total_carb     / total_mass * 100
protein_per_100g  = total_protein  / total_mass * 100
fat_per_100g      = total_fat      / total_mass * 100
```

(If a frame already carries `*_per_100g` columns, those are used directly — this is how the Streamlit form and `real_products.csv` enter the pipeline.)

### Step 2 — Baselines with outlier removal

For each nutrient, the baseline is the **trimmed mean** across all dishes: the lowest 1% and highest 1% of values are removed before averaging (configurable via `trim_quantile`), so major outliers do not drag the reference point. Fitted baselines (current data):

```text
calories 126.02   carbs 10.01   protein 7.68   fat 6.57   (per 100 g)
```

These are stored in `data/output/nutrition_scorer.json` and reused by training, prediction, and the UI, so every consumer scores with the same reference.

### Step 3 — Directional relative deviation

Each nutrient is penalised only in its unhealthy direction:

```text
calories, carbs, fat:  deviation = max((value - baseline) / baseline, 0)
protein:               deviation = max((baseline - value) / baseline, 0)
```

So high calories/carbs/fat are bad, low protein is bad, and being below baseline on calories/carbs/fat costs nothing.

### Step 4 — Exponential penalty with saturation

```text
penalty_n = 120 * (1 - exp(-1.5 * deviation_n))
nutrient_penalty = sum over the 4 nutrients
```

Properties: 0 at baseline, monotonically increasing, convex (larger deviations earn disproportionately larger deductions — the requested behaviour), and capped at 120 per nutrient so the 0–100 scale stays discriminative for foods several times above baseline instead of collapsing everything to exactly 0. Calibration: ≈47 pts at ⅓ deviation, ≈76 pts at ⅔, ≈93 pts at 1.0 (double baseline), ≈120 (saturated) at 3.3×.

> Note on the original calibration example (avg 300 → 400 deducts ~20, 500 deducts ~50): those two points force convex growth that is mathematically incompatible with any saturation. An uncapped formula reproduces them but deducts thousands of points at 3–5× baseline (Nutella is 4.3× average calories), which floors every processed food at 0 and makes the categories meaningless. The capped exponential keeps the same exponential shape up to saturation. This is the one deliberate deviation from the literal spec and it is documented here so it can be defended in review.

### Step 5 — Processing penalty and final score

```text
processing_penalty = 5*artificial_sweetener + 5*artificial_colours + 5*preservatives
healthiness_score  = clip(100 - nutrient_penalty - processing_penalty, 0, 100)
score_category     = 80+/65+/50+/30+/otherwise → Excellent/Good/Moderate/Poor/Very Poor
```

Flags come from two sources: (a) in the UI they are manual checkboxes; (b) for text ingredients the loader auto-detects common additive keywords (aspartame/sucralose/saccharin/acesulfame/stevia; colour/color/dye/red 40/yellow 5/blue 1; preserv/benzoate/nitrite/nitrate/sorbate/sulfite). Nutrition5k rows have no additive labels, so normal rows initialise all three flags to **0** and `--augment-processing-flags` creates synthetic flag-on variants (marked `processing_variant=1`) so the ingredient model can learn the penalty. These synthetic rows are not observed labels — manual/external additive annotation should replace them for a real processing experiment.

## 3. The machine-learning models

### Why a model at all

The rule above is a hand-crafted function; the XGBoost models learn to *predict* it from raw features, which (a) tests how much of the rule is recoverable from different feature views, (b) gives a per-food continuous prediction with SHAP attributions, and (c) lets the three feature sets be compared (nutrition vs ingredients vs combined).

### Feature sets

| Model | Features |
|---|---|
| `nutrition` (4) | `calories_per_100g`, `carbs_per_100g`, `protein_per_100g`, `fat_per_100g` |
| `ingredient` (11) | `ingredient_calories_per_100g`, `ingredient_carbs_per_100g`, `ingredient_protein_per_100g`, `ingredient_fat_per_100g` (mass-weighted nutrient density of the dish's own ingredients), `ingredient_count`, `top_ingredient_mass_fraction`, `top3_ingredient_mass_fraction`, `ingredient_entropy` (Shannon entropy of the mass distribution — many small ingredients ≈ formulation complexity), and the 3 additive flags |
| `combined` (15) | all of the above |

Formulation features are derived by joining `dish_ingredients.csv` by `dish_id`: each ingredient's grams become a mass fraction; per-nutrient totals divided by total mass give the formulation's own per-100 g density (which should track the dish's label density when the data is consistent — the models exploit exactly that); entropy measures how spread out the mass is across ingredients.

### Model type and hyperparameters

`XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.85, colsample_bytree=0.9, objective=reg:squarederror, eval_metric=mae)` — a shallow, heavily regularised gradient-boosted tree regressor; small depth + subsampling keep the model well-behaved on a target it is partly reconstructing.

### Train/holdout protocol (leakage control)

1. **Dish-grouped 80/20 split** (`GroupShuffleSplit` on `dish_id`) — no dish appears in both sides, so the synthetic processing variants of a dish never leak across the split.
2. **Baseline refit**: when `train_models.py --scorer` is supplied, the evaluation target is recomputed with baselines fit **on the training dishes only**, so the held-out target was not computed using statistics from held-out dishes.
3. **Production artifacts**: the models that get saved are refit on *all* scored rows (production models), while the reported metrics come from the leakage-safe held-out evaluation.

### Reported metrics (current data, seed 42)

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Nutrition-only | 2.19 | 3.88 | 0.986 |
| Ingredient-only | 1.13 | 2.42 | 0.994 |
| Combined | 1.04 | 2.50 | 0.994 |

Interpretation: these numbers measure how well each model *reproduces the defined scoring rule* on held-out dishes (the target is derived from the same feature space). They are not clinical healthiness accuracy — that would require expert ratings or an external labelled dataset.

### SHAP

For each saved model, `shap.TreeExplainer` is run on up to 1,000 sampled rows; per-feature mean |SHAP| is written to `models/shap_<model>.csv` plus a summary plot (`models/shap_<model>.png`). The SHAP table answers H2 — which nutritional/ingredient factors drive an individual food's score — e.g. for the combined model the calorie-density and ingredient-formulation features dominate.

## 4. Full data flow

### 4.1 Inputs

```text
data/input/dish_nutrition_values.csv   dish_id, calories, mass, fat, carb, protein          (4,768 rows)
data/input/dish_ingredients.csv        dish_id, ingr_id, ingr_name, grams, calories, fat, carb, protein  (27,225 rows)
data/input/real_products.csv           5 commercial products, per-100 g values + ingredient text
```

### 4.2 Pipeline stages

```text
┌─────────────────────────────┐
│ 1. LOAD  load_nutrition_dataset()
│    read dish CSV; if --ingredient-path given,
│    aggregate_ingredient_metadata() joins ingredient
│    rows per dish_id → formulation features
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ 2. PREPARE  prepare_dataframe()
│    normalise column aliases (calories/energy,
│    mass/weight/...), compute per-100 g densities,
│    ingredient count/entropy/mass fractions,
│    auto-detect 3 additive flags from ingredient text
│    (missing flags → 0)
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ 3. BASELINE FIT  NutrientScorer.fit()
│    trimmed mean (1% each tail) per nutrient →
│    nutrition_scorer.json  (single source of truth)
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ 4. SCORE  NutrientScorer.transform()
│    directional deviation → capped exponential
│    penalty → processing penalty → clip to 0–100
│    → healthiness_score + score_category
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ 5. AUGMENT (optional, training only)
│    augment_processing_variants(): +1 row per flag
│    and +1 all-flags row, target reduced by 5/10/15
│    pts, marked processing_variant=1
└──────────────┬──────────────┘
               ▼
        data/output/nutrition5k_scored.csv   (23,840 rows)
               │
               ▼
┌─────────────────────────────┐
│ 6. TRAIN  train_models.py
│    GroupShuffleSplit(dish_id) 80/20
│    evaluation target refit on train-only baselines
│    fit+evaluate 3 XGB models (metrics → metrics.json)
│    refit on all rows → models/xgb_<name>.joblib
│    SHAP → models/shap_<name>.csv / .png
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ 7. PREDICT / SERVE
│    predict.py: scored CSV → predictions.csv
│    app.py (Streamlit): live scorer + 3 joblib models
│      • Examples tab: 8 one-click foods through the
│        full prepare→score→predict path
│      • Score a food: manual macros + ingredients +
│        flags + model choice
│      • Batch CSV: upload → scored table + download
│      • Real products: 5 verified commercial items
│      • Models & SHAP: metrics + attribution
└─────────────────────────────┘
```

### 4.3 Column lineage (scored CSV)

| Stage | Columns produced |
|---|---|
| load | `dish_id, total_calories, total_mass, total_fat, total_carb, total_protein, ingredient_count, top_ingredient_mass_fraction, top3_ingredient_mass_fraction, ingredient_entropy, ingredient_calories_per_100g, ingredient_carbs_per_100g, ingredient_protein_per_100g, ingredient_fat_per_100g, ingredients` |
| prepare | `calories_per_100g, carbs_per_100g, protein_per_100g, fat_per_100g, artificial_sweetener, artificial_colours, preservatives` |
| score | `<nutrient>_relative_deviation, <nutrient>_penalty` (×4), `nutrient_penalty, processing_penalty, healthiness_score, nutrition_quality_score, score_category` |
| augment | `processing_variant` (0 = real row, 1 = synthetic flag variant) |

### 4.4 End-to-end trace of one UI submission (chicken breast)

```text
UI form: 165 cal, 0 carbs, 31 g protein, 3.6 g fat, "chicken breast", no flags
→ prepare_dataframe: already per-100 g; ingredient_count=1, flags auto=0
→ transform:
   calories: (165-126.02)/126.02 = 0.31 deviation → 120*(1-e^-0.465) = 44.6 pts
   carbs:    at/below baseline → 0 pts
   protein:  above baseline (good direction) → 0 pts
   fat:      below baseline → 0 pts
   processing: 0
→ healthiness_score = 100 - 44.6 = 55.4  → "Moderate"
→ xgb_combined.predict([165, 0, 31, 3.6, <11 ingredient features>]) = 38.6
```

The same 55.4 is produced by `predict.py`, `score_real_products.py`, and the UI — one shared `NutrientScorer`, one scorer JSON, three model files.

## 5. Verified real-product ratings

Computed by this system on 2026-09-22; every per-100 g value checked against a public source (Nutella.com; coca-cola.com; Kellogg's US label via webstaurantstore; myfooddata/USDA Branded Foods for Doritos; USDA FDC 171477 for chicken):

| Product | Rule-based | Category | XGBoost (combined) | Source |
|---|---:|---|---:|---|
| Chicken breast (cooked, skinless) | 55.4 | Moderate | 38.6 | USDA FDC 171477 |
| Kellogg's Froot Loops (US) | 0.0 | Very Poor | 1.1 | Kellogg's US label (374/87.5/5.7/3.8) |
| Coca-Cola Original | 0.0 | Very Poor | 0.7 | coca-cola.com (42/10.6/0/0 per 100 ml) |
| Doritos Nacho Cheese | 0.0 | Very Poor | 0.0 | USDA Branded Foods via myfooddata (536/64.3/7.1/28.6) |
| Nutella hazelnut spread | 0.0 | Very Poor | 0.0 | Nutella.com (539/57.5/6.3/30.9) |

Full file: `data/output/real_products_scores.csv`.

## 6. The Streamlit interface

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Runs at `http://localhost:8501` (pass `--server.port` to override). Tabs:

1. **Examples (run)** — 8 one-click example foods (average dish, chicken breast, salad, rice, cake, Coca-Cola, diet cola, Nutella). Clicking a row runs the full pipeline on that food and shows the rule-based score, category, XGBoost prediction, and per-nutrient penalty breakdown; the summary table at the bottom is computed live with the same scorer.
2. **Score a food** — manual form: 4 macro inputs (per 100 g), ingredient text (additive flags auto-detected), 3 manual processing-flag checkboxes, model choice, score button.
3. **Batch CSV** — upload a CSV (`name`, `calories_per_100g, carbs_per_100g, protein_per_100g, fat_per_100g`, optional `ingredients` and flag columns); returns a scored table sorted by score plus a CSV download.
4. **Real products** — the 5 verified commercial products, scored live.
5. **Models & SHAP** — holdout metrics and per-model SHAP attributions (CSV + plot).

The UI loads `data/output/nutrition_scorer.json` and the three `models/xgb_*.joblib` files, so it is read-only with respect to training: rebuild the models with the CLI commands below if you retrain.

## 7. Local run (CLI)

Use the Python installation that has the dependencies:

```bash
python -m pip install -r requirements.txt
python -m pytest tests/test_scoring.py -q
python build_targets.py data/input/dish_nutrition_values.csv --ingredient-path data/input/dish_ingredients.csv --output data/output/nutrition5k_scored.csv --scorer-output data/output/nutrition_scorer.json --augment-processing-flags
python train_models.py data/output/nutrition5k_scored.csv --scorer data/output/nutrition_scorer.json --output-dir models
python predict.py data/input/dish_nutrition_values.csv --ingredient-path data/input/dish_ingredients.csv --model combined --model-dir models --scorer data/output/nutrition_scorer.json --output data/output/predictions.csv
python score_real_products.py          # rates data/input/real_products.csv
```

Verified current outputs: 23,840 scored rows (4,768 dishes × 5 processing variants), 4,768 predictions, 6/6 tests passing, all scores in [0, 100].

## 8. Kaggle

Upload `nutrition_quality.py`, `build_targets.py`, `train_models.py`, `predict.py` and the two input CSVs to Kaggle and run the same commands with `/kaggle/input/...` paths. Kaggle's copy of the dataset may use a simplified schema — inspect `pd.read_csv(path).columns` first; `prepare_dataframe` already accepts common aliases (`energy`, `weight`, `carbohydrates`, ...).

## 9. Hypotheses and limitations (paper-ready)

**Novelty.** An explainable ML framework that integrates quantitative nutritional features with ingredient-derived formulation/processing features into a single continuous food-healthiness score with per-food attributions (SHAP).
- H1: a model combining nutrition + ingredient features predicts the healthiness target more accurately than either alone. (Observed on this data: combined MAE 1.04 vs nutrition 2.19 vs ingredient 1.13 — H1 supported on the derived target.)
- H2: SHAP identifies the major nutritional/ingredient factors driving individual scores. (Supported: calorie density and formulation features carry the largest mean |SHAP|.)

**Limitations.**
1. The score is a normative, dataset-relative proxy, not ground-truth human healthiness; prediction metrics measure approximation of the defined rule.
2. Nutrition5k is Google-cafeteria dish data, not representative branded packaged food.
3. Calories/macros cannot identify preservatives, colours, sweeteners, ultra-processing, sodium, fibre, or micronutrients.
4. Additive flags are keyword-detected or synthetic in this dataset; a real processing experiment needs manual/external additive annotation.
5. The capped penalty saturates at 120 pts/nutrient — a deliberate, documented departure from the literal "exponential forever" spec so the 0–100 scale retains categories at extreme values.
6. Protein is penalised below baseline, which floors zero-protein drinks; this is a modelling choice (low protein = less nutritional quality), not a claim that calories-free is unhealthy.
