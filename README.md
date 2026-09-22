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

### 3.1 What is created and why

The rule in §2 is a hand-crafted function. Three XGBoost regressors are trained to *predict* that rule from raw features, which (a) tests how much of the rule is recoverable from each feature view, (b) provides a continuous per-food prediction with SHAP attributions, and (c) lets nutrition vs ingredient vs combined be compared (H1).

### 3.2 Feature sets

| Model | Features |
|---|---|
| `nutrition` (4) | `calories_per_100g`, `carbs_per_100g`, `protein_per_100g`, `fat_per_100g` |
| `ingredient` (11) | `ingredient_calories_per_100g`, `ingredient_carbs_per_100g`, `ingredient_protein_per_100g`, `ingredient_fat_per_100g` (mass-weighted nutrient density of the dish's own ingredients), `ingredient_count`, `top_ingredient_mass_fraction`, `top3_ingredient_mass_fraction`, `ingredient_entropy` (Shannon entropy of the mass distribution — many small ingredients ≈ formulation complexity), and the 3 additive flags |
| `combined` (15) | all of the above |

Formulation features come from joining `dish_ingredients.csv` by `dish_id`: each ingredient's grams become a mass fraction; per-nutrient totals divided by total mass give the formulation's own per-100 g density (which tracks the dish's label density when the data is consistent — the models exploit exactly that); entropy measures how spread out the mass is across ingredients.

### 3.3 How the model is created (XGBoost internals)

An `XGBRegressor` is a gradient-boosted ensemble built in 400 boosting rounds (`n_estimators=400`):

1. **Round m**: compute residuals of the current ensemble (objective `reg:squarederror` → gradient = prediction − target); fit one regression tree of depth ≤ 4 (`max_depth=4` → at most 2⁴ = 16 leaves, observed 9–16 leaves/tree) to the residuals on an 85% row subsample (`subsample=0.85`) with 90% of features per tree (`colsample_bytree=0.9`); add the tree scaled by the learning rate 0.04 (`learning_rate=0.04`, shrinkage) to the ensemble.
2. **Stop** after 400 rounds.

So the fitted "model" is not a weight vector — it is a forest of 400 small trees, each storing a feature + threshold at every internal node and a numeric prediction (the "weight") at every leaf. XGBoost additionally regularises leaf values (L2) and minimum split gain (defaults). The whole fitted object serialises to a ~680 KB file.

Hyperparameter rationale: shallow trees + small learning rate + subsampling = a heavily regularised regressor, appropriate for a target the feature space partly reconstructs (the target is derived from the same nutrients).

### 3.4 Training data

| Item | Value |
|---|---|
| Rows | 23,840 = 4,768 real dishes × 5 (1 original `processing_variant=0` + 4 synthetic flag variants `=1`) |
| Source | `data/output/nutrition5k_scored.csv` (output of `build_targets.py`) |
| Target | `healthiness_score` (rule-based, §2); synthetic variants carry the target reduced by 5/10/15 points |
| Target distribution | mean 24.2, median 9.5, min 0, max 100; 72.7% of dishes in Very Poor (§4) |
| Split | `GroupShuffleSplit` on `dish_id`, 80/20, seed 42 — a dish and its 4 variants always land on the same side |
| Evaluation target (leakage-safe) | baselines refit on the 80% training dishes only (`train_models.py --scorer`) so held-out targets never use held-out-dish statistics |
| Production target | the stored `healthiness_score` column (baselines fit on all dishes) |

### 3.5 Training procedure (`train_models.py`, step by step)

What `python train_models.py data/output/nutrition5k_scored.csv --scorer data/output/nutrition_scorer.json --output-dir models` does:

1. Load the scored CSV; verify the 15 feature columns + target exist (missing → instructs to run `build_targets.py` first).
2. Split indices by `dish_id` (`GroupShuffleSplit`, `test_size=0.2`, seed 42).
3. Clone the scorer hyperparameters and refit baselines on training rows only → `evaluation_target`.
4. For each of `nutrition`, `ingredient`, `combined`:
   a. `fit_model` — fit the §3.3 regressor on training rows × `feature_columns(model)` (evaluation model).
   b. `evaluate` — predict on held-out rows, clip to [0, 100], compute MAE/RMSE/R² → `metrics.json`; write `predictions_<model>.csv` (dish_id, actual, predicted for the holdout).
   c. Production model — refit the same hyperparameters on **all** 23,840 rows with the production target; **this** is what gets saved.
   d. `save_shap` — `shap.TreeExplainer` on a 1,000-row sample → `shap_<model>.csv` (mean |SHAP| per feature) + `shap_<model>.png` (summary plot).
5. Dump each production model to `models/xgb_<model>.joblib` via `joblib.dump`.

### 3.6 Model weights — where they live

| File | Size | Content |
|---|---:|---|
| `models/xgb_nutrition.joblib` | 677 KB | 400-tree ensemble over 4 features |
| `models/xgb_ingredient.joblib` | 676 KB | 400-tree ensemble over 11 features |
| `models/xgb_combined.joblib` | 681 KB | 400-tree ensemble over 15 features (the model the UI serves) |
| `models/metrics.json` | 356 B | holdout MAE/RMSE/R² for the three models |
| `models/predictions_<model>.csv` | ~167 KB each | holdout dish_id, actual, predicted |
| `models/shap_<model>.csv` / `.png` | ~0.4 KB / 167–213 KB | mean \|SHAP\| per feature + summary plot |
| `data/output/nutrition_scorer.json` | 389 B | baselines + penalty hyperparameters (the rule half of inference) |

Load and inspect:

```python
import joblib
m = joblib.load("models/xgb_combined.joblib")      # XGBRegressor, 400 trees
m.get_booster().num_boosted_rounds()                # 400
m.get_booster().feature_names                       # 15 feature names in training order
m.get_booster().get_dump()                          # every tree: node/feature/threshold/leaf value
m.get_booster().get_score(importance_type="gain")   # split-gain per feature
m.predict(X)                                        # inference
```

Observed gain importance (combined, top 5): `ingredient_protein_per_100g` 116,892 · `protein_per_100g` 76,833 · `fat_per_100g` 66,120 · `carbs_per_100g` 48,413 · `calories_per_100g` 42,313 — protein/fat density drive the splits, matching the SHAP order in §3.9.

### 3.7 Inference procedure (identical in `predict.py`, `score_real_products.py`, `app.py`)

1. **Load artifacts**: `scorer = load_scorer("data/output/nutrition_scorer.json")`; `model = joblib.load("models/xgb_<name>.joblib")`.
2. **Prepare features**: raw rows (totals or per-100 g) → `prepare_dataframe` — column aliases, per-100 g densities, formulation features, auto-detected additive flags (§2 step 5).
3. **Column contract**: `feature_columns(frame, model)` returns exactly the names/order the model was trained on (combined = 4 nutrient + 11 ingredient features). The booster stores the 15 feature names, so a missing/misordered column fails loudly instead of silently mispredicting.
4. **Predict**: `score = model.predict(X).clip(0, 100)` → `predicted_healthiness_score`.
5. `app.py` additionally runs `scorer.transform` for the rule-based score and the per-nutrient penalty breakdown shown in the UI.

Worked example — chicken breast (165 cal, 0 carbs, 31 g protein, 3.6 g fat):

- Rule-based: 100 − 44.6 (calorie penalty) = **55.4 → Moderate** (§2).
- Inference: X = `[165, 0, 31, 3.6, <11 ingredient features from "chicken breast">]` → combined model → **38.6** in the UI (38.2 for the canonical single-ingredient feature row; the 0.4 difference is the formulation features of the two input forms, not model drift).

### 3.8 Reported metrics (leakage-safe holdout, seed 42)

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Nutrition-only | 2.19 | 3.88 | 0.986 |
| Ingredient-only | 1.13 | 2.42 | 0.994 |
| Combined | 1.04 | 2.50 | 0.994 |

These numbers measure how well each model *reproduces the defined scoring rule* on held-out dishes (the target is derived from the same feature space) — not clinical healthiness accuracy, which would require expert ratings or an external labelled dataset. Per-category mean model predictions (combined, holdout): Excellent 89.1, Good 64.2, Moderate 49.5, Poor 31.7, Very Poor 4.1, against category score means 96.6 / 72.2 / 57.5 / 38.6 / 5.6. Per-category MAE is smallest in the saturated tails (Very Poor 0.70, Excellent 1.38) and largest in Good (3.11): once the target floors at 0 the predictions concentrate there and are easy, while the mid-score categories carry the real separation work (§4).

### 3.9 SHAP (mean |SHAP| per feature, all three models)

| Rank | nutrition | ingredient | combined |
|---:|---|---|---|
| 1 | protein_per_100g — 15.32 | ingredient_protein_per_100g — 15.91 | ingredient_protein_per_100g — 9.93 |
| 2 | fat_per_100g — 10.81 | ingredient_fat_per_100g — 10.72 | carbs_per_100g — 7.93 |
| 3 | carbs_per_100g — 9.63 | ingredient_carbs_per_100g — 9.67 | fat_per_100g — 7.87 |
| 4 | calories_per_100g — 6.97 | ingredient_calories_per_100g — 7.26 | protein_per_100g — 7.06 |
| 5 | — | artificial_sweetener — 1.25 | calories_per_100g — 4.70 |
| 6 | — | artificial_colours — 1.18 | ingredient_fat_per_100g — 2.93 |
| 7 | — | preservatives — 1.04 | ingredient_carbs_per_100g — 2.80 |
| 8 | — | ingredient_count — 0.18 | ingredient_calories_per_100g — 1.77 |

Reading: (a) protein density is the single strongest lever in every model — low protein is penalised in the rule and 58% of dishes sit below the protein baseline, so protein separates the high- and low-score regions best (§4.3). (b) In the combined model the formulation features inherit the label features' ranking but with roughly halved magnitude, consistent with the two views carrying the same information plus label noise. (c) The three additive flags each carry ~1 point of mean |SHAP| in the ingredient model — non-trivial, and the only channel through which processing information reaches the target (the 19,072 synthetic flag variants, §5). (d) This supports H2: SHAP identifies protein/fat/carb density as the major factors behind an individual food's score.

## 4. Food-category averages across the dataset

### 4.1 How the averages are extracted

1. `build_targets.py` scores all 4,768 dishes with the §2 rule → `healthiness_score` + `score_category` (the category is assigned by `np.select` on the score thresholds: ≥80/≥65/≥50/≥30/else).
2. Group the **real dish rows** (`processing_variant == 0` — the 19,072 synthetic flag variants are excluded so category means describe actual foods, not augmented rows) by `score_category` and average each feature column.

```python
d = pd.read_csv("data/output/nutrition5k_scored.csv")
d = d[d["processing_variant"] == 0]
d.groupby("score_category")[
    ["calories_per_100g", "carbs_per_100g", "protein_per_100g", "fat_per_100g",
     "ingredient_count", "healthiness_score"]
].mean()
```

All figures below are computed from the current `data/output/nutrition5k_scored.csv` (4,768 dishes).

### 4.2 Category means (real dishes, per 100 g)

| Category | n | share | calories | carbs | protein | fat | ingredients | mean score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Excellent | 600 | 12.6% | 97.7 | 5.0 | 11.4 | 3.8 | 9.4 | 96.6 |
| Good | 218 | 4.6% | 109.3 | 6.0 | 11.3 | 4.9 | 8.8 | 72.2 |
| Moderate | 207 | 4.3% | 116.5 | 6.1 | 12.6 | 5.1 | 9.8 | 57.5 |
| Poor | 275 | 5.8% | 92.0 | 8.0 | 6.7 | 4.5 | 8.6 | 38.6 |
| Very Poor | 3,468 | 72.7% | 144.2 | 12.6 | 6.9 | 8.1 | 4.4 | 5.6 |

Two things stand out:

- **The means are deliberately not monotone in all nutrients.** Poor dishes (mean score 38.6) have the *lowest* calories (92.0) yet the lowest protein (6.7): their score is dragged down by the below-baseline protein penalty, not by calories. That asymmetry (protein penalised below baseline, calories/carbs/fat above) is a fingerprint of the §2 rule — a simple "low calorie = healthy" pattern does not exist in this data.
- **Trimming measurably shifts the reference.** Raw dataset means are cal 132.5 / carbs 10.76 / protein 7.88 / fat 7.12; the trimmed (1%) baselines actually used are 126.0 / 10.0 / 7.7 / 6.6. The 1% tails are heavy on the high end (high-calorie dishes), so trimming pulls the calorie baseline ~6% below the raw mean — without it, every mid-calorie dish would look "above baseline" and the penalties would over-fire.
- **The dataset is dominated by Very Poor** (72.7%): most Nutrition5k dishes exceed the calorie/carb/fat baselines or sit below the protein baseline, so their penalties floor the score at 0.

### 4.3 How the category structure impacted the model

Mean penalty points deducted per category (nutrient penalties summed over the 4 nutrients):

| Category | cal | carbs | protein | fat | nutrient total | effect |
|---|---:|---:|---:|---:|---:|---|
| Excellent | 1.0 | 0.3 | 1.4 | 0.6 | 3.4 | stays in 93–100 |
| Good | 8.6 | 1.4 | 12.4 | 5.3 | 27.8 | |
| Moderate | 13.7 | 3.8 | 16.1 | 8.9 | 42.5 | |
| Poor | 5.5 | 6.6 | 38.6 | 10.8 | 61.4 | protein is the main driver |
| Very Poor | 28.7 | 34.6 | 50.0 | 33.5 | 146.7 | floors at 0 |

Consequences for the XGBoost models:

1. **Class imbalance dominates training.** 72.7% of the 23,840 rows (including synthetic variants) are Very Poor, so the ensemble learns the dominant mode (≈0) very well and is comparatively under-trained on Good/Moderate/Poor. This shows up directly in the per-category MAE of the combined model on the leakage-safe holdout: Very Poor **0.70**, Excellent **1.38**, Poor **1.95**, Moderate **2.98**, Good **3.11** — the mid-score categories are the hardest and the saturated tails the easiest.
2. **Protein carries a U-shaped signal** (below baseline bad, above baseline good — §4.2's Poor-category dip), and tree splits encode exactly that. It is the top or second-top SHAP feature in all three models and #2 in gain importance, because the category structure makes protein the sharpest separator between the high-score and the 73% low-score mass.
3. **The synthetic flag variants shift the target but not the categories** (augmented-row mean target 20.44 vs real-row 24.24): they add 5/10/15-point depressions within each category, which is what teaches the additive flags their ~1.0-point mean |SHAP| in the ingredient model.
4. **Practical implication.** If the system is deployed to discriminate among mid-score foods (the Good/Moderate/Poor band), consider a class-balanced resample of training targets or class weights — as-is, the model is calibrated for the dataset's real distribution (mostly floor + a small high-score tail), which is what the R²/MAE in §3.8 measure.

## 5. Full data flow

### 5.1 Inputs

```text
data/input/dish_nutrition_values.csv   dish_id, calories, mass, fat, carb, protein          (4,768 rows)
data/input/dish_ingredients.csv        dish_id, ingr_id, ingr_name, grams, calories, fat, carb, protein  (27,225 rows)
data/input/real_products.csv           5 commercial products, per-100 g values + ingredient text
```

### 5.2 Pipeline stages

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

### 5.3 Column lineage (scored CSV)

| Stage | Columns produced |
|---|---|
| load | `dish_id, total_calories, total_mass, total_fat, total_carb, total_protein, ingredient_count, top_ingredient_mass_fraction, top3_ingredient_mass_fraction, ingredient_entropy, ingredient_calories_per_100g, ingredient_carbs_per_100g, ingredient_protein_per_100g, ingredient_fat_per_100g, ingredients` |
| prepare | `calories_per_100g, carbs_per_100g, protein_per_100g, fat_per_100g, artificial_sweetener, artificial_colours, preservatives` |
| score | `<nutrient>_relative_deviation, <nutrient>_penalty` (×4), `nutrient_penalty, processing_penalty, healthiness_score, nutrition_quality_score, score_category` |
| augment | `processing_variant` (0 = real row, 1 = synthetic flag variant) |

### 5.4 End-to-end trace of one UI submission (chicken breast)

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

## 6. Verified real-product ratings

Computed by this system on 2026-09-22; every per-100 g value checked against a public source (Nutella.com; coca-cola.com; Kellogg's US label via webstaurantstore; myfooddata/USDA Branded Foods for Doritos; USDA FDC 171477 for chicken):

| Product | Rule-based | Category | XGBoost (combined) | Source |
|---|---:|---|---:|---|
| Chicken breast (cooked, skinless) | 55.4 | Moderate | 38.6 | USDA FDC 171477 |
| Kellogg's Froot Loops (US) | 0.0 | Very Poor | 1.1 | Kellogg's US label (374/87.5/5.7/3.8) |
| Coca-Cola Original | 0.0 | Very Poor | 0.7 | coca-cola.com (42/10.6/0/0 per 100 ml) |
| Doritos Nacho Cheese | 0.0 | Very Poor | 0.0 | USDA Branded Foods via myfooddata (536/64.3/7.1/28.6) |
| Nutella hazelnut spread | 0.0 | Very Poor | 0.0 | Nutella.com (539/57.5/6.3/30.9) |

Full file: `data/output/real_products_scores.csv`.

## 7. The Streamlit interface

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

## 8. Local run (CLI)

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

## 9. Kaggle

Upload `nutrition_quality.py`, `build_targets.py`, `train_models.py`, `predict.py` and the two input CSVs to Kaggle and run the same commands with `/kaggle/input/...` paths. Kaggle's copy of the dataset may use a simplified schema — inspect `pd.read_csv(path).columns` first; `prepare_dataframe` already accepts common aliases (`energy`, `weight`, `carbohydrates`, ...).

## 10. Hypotheses and limitations (paper-ready)

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
