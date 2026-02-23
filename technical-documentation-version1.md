# Technical Documentation: Global AI Job Salary Prediction

## 1. Problem Statement

Salary structures in AI/data-related fields lack transparency, creating difficulties for job seekers in career planning and offer benchmarking, and for policymakers addressing pay equity gaps. This project builds a **regression model** to predict salary (USD) based on job attributes, skills, and experience.

**ML value proposition:** Without ML, salary estimates rely on manual surveys and subjective guessing. ML enables data-driven, scalable predictions across roles, geographies, and skill sets.

## 2. ML Formulation

| Item | Detail |
|------|--------|
| **Problem type** | Supervised Regression |
| **Target variable (Y)** | `salary_usd` (continuous, range: \$32,519 -- \$399,095, mean: \$115,349) |
| **Feature variables (X)** | `job_title`, `experience_level`, `years_experience`, `required_skills`, `company_location`, `company_size`, `employment_type`, `remote_ratio`, `education_required`, `industry`, `benefits_score` |
| **Dropped columns** | `job_id`, `salary_currency`, `employee_residence`, `posting_date`, `application_deadline`, `job_description_length`, `company_name` (identifiers, leakage-prone, or low predictive value) |

## 3. Dataset Overview

- **Source:** AI & Data Job Salaries and Skills Dataset 2024--2025 (synthetic)
- **Location:** `./data/ai_job_dataset.csv`
- **Size:** 14,999 rows x 19 columns

### Data Limitations

This dataset is **synthetically generated**, which means feature distributions and inter-variable relationships may not fully reflect real-world salary dynamics. Consequently, models trained on this data should be treated as proof-of-concept demonstrations rather than production-grade predictors. If the project is extended to a real-world application, retraining on verified salary data (e.g., Glassdoor, Levels.fyi, or government labor statistics) is strongly recommended.

### Key Feature Distributions

| Feature | Type | Unique Values |
|---------|------|---------------|
| `job_title` | Categorical | 20 (e.g., Data Scientist, ML Engineer, NLP Engineer) |
| `experience_level` | Categorical | 4 -- EN (Entry), MI (Mid), SE (Senior), EX (Executive) |
| `employment_type` | Categorical | 4 -- FT (Full-time), PT (Part-time), CT (Contract), FL (Freelance) |
| `company_location` | Categorical | 20 countries |
| `company_size` | Categorical | 3 -- S (Small), M (Medium), L (Large) |
| `remote_ratio` | Categorical | 3 -- 0 (On-site), 50 (Hybrid), 100 (Remote) |
| `education_required` | Ordinal | 4 -- Associate, Bachelor, Master, PhD |
| `years_experience` | Numerical | 0--19 |
| `required_skills` | Text (multi-label) | Comma-separated list (e.g., "Python, PyTorch, AWS") |
| `industry` | Categorical | 15 |
| `benefits_score` | Numerical | Continuous |

### Potential Multicollinearity

`years_experience` and `experience_level` are likely highly correlated (e.g., entry-level roles typically correspond to low years of experience). During modeling, this will be assessed via a correlation matrix and Variance Inflation Factor (VIF) analysis. If significant multicollinearity is detected, strategies include dropping one of the two features, or relying on tree-based models which are inherently robust to collinearity.

## 4. Data Preprocessing

### 4.1 Missing Value Handling

Before any feature engineering, inspect every column for missing or null values. The handling strategy depends on the missing rate and feature type:

| Missing Rate | Strategy |
|--------------|----------|
| < 5% | Drop rows (minimal data loss) or impute with median (numerical) / mode (categorical) |
| 5% -- 30% | Impute: median for numerical features, mode for categorical features; optionally add a binary `_is_missing` indicator column |
| > 30% | Consider dropping the feature entirely, or use model-based imputation (e.g., KNN Imputer, IterativeImputer) if the feature is known to be important |

All imputation parameters (e.g., median values) must be fitted on the **training set only** and then applied to the test set to prevent data leakage.

### 4.2 Handle `required_skills`

Multi-hot encode the comma-separated skills into binary columns (one per unique skill).

**Dimensionality control:** To avoid an excessively sparse feature matrix, only retain skills that appear in at least **2% of all samples**. Rare skills below this threshold are grouped into an `other_skills_count` feature (integer count of rare skills per row). If the resulting dimensionality is still too high, consider applying PCA or TruncatedSVD to the skill matrix to compress it into a lower-dimensional representation while preserving most variance.

### 4.3 Encode Categoricals

- `education_required` -- ordinal encode (Associate=0, Bachelor=1, Master=2, PhD=3).
- `experience_level` -- ordinal encode (EN=0, MI=1, SE=2, EX=3).
- `job_title`, `company_location`, `industry`, `employment_type`, `company_size`, `remote_ratio` -- one-hot encode for linear/MLP models; native categorical handling or target encoding for tree-based models (XGBoost/LightGBM).

### 4.4 Scale Numericals

Standardize `years_experience` and `benefits_score` (zero mean, unit variance) for distance-based and gradient-sensitive models (Linear Regression, MLP). Tree-based models do not require scaling.

### 4.5 Train/Test Split

80/20 stratified split (stratify by `experience_level` to preserve distribution across experience tiers). Random seed fixed for reproducibility.

## 5. Evaluation Metrics

| Metric | Rationale |
|--------|-----------|
| **MAE (Mean Absolute Error)** | Interpretable in dollar terms; robust to outliers |
| **RMSE (Root Mean Squared Error)** | Penalizes large errors more heavily; standard regression metric |
| **R² (Coefficient of Determination)** | Shows proportion of variance explained; allows cross-model comparison |

Use **5-fold cross-validation** on the training set for model selection, then report final metrics on the held-out test set.

## 6. Recommended ML Models

Given a team of 4--5 members (requiring at least n-1 = 3--4 techniques):

| # | Model | Why |
|---|-------|-----|
| 1 | **Linear Regression (+ Ridge/Lasso)** | Baseline; interpretable coefficients; regularization handles high-dimensional skill features |
| 2 | **Random Forest** | Handles mixed feature types natively; captures non-linear interactions; feature importance ranking |
| 3 | **Gradient Boosting (XGBoost or LightGBM)** | State-of-the-art tabular performance; built-in handling of categorical features (LightGBM) |
| 4 | **Neural Network (MLP)** | Demonstrates deep learning from the course syllabus; can model complex feature interactions |

**Optional additions:** Support Vector Regression (SVR), K-Nearest Neighbors (KNN) for additional comparison points.

## 7. Hyperparameter Tuning

For each model, perform hyperparameter tuning using the training set with cross-validation:

| Model | Key Hyperparameters | Tuning Method |
|-------|---------------------|---------------|
| Ridge/Lasso | `alpha` | GridSearchCV (small search space) |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split` | RandomizedSearchCV |
| XGBoost/LightGBM | `learning_rate`, `n_estimators`, `max_depth`, `subsample`, `colsample_bytree` | Optuna (Bayesian Optimization) -- more efficient than grid/random for large search spaces |
| MLP | `hidden_layer_sizes`, `learning_rate_init`, `dropout`, `batch_size` | Optuna or manual staged search |

**Why Optuna for complex models?** GridSearch is computationally prohibitive when the hyperparameter space is large (e.g., XGBoost has 5+ key parameters). Bayesian optimization intelligently samples promising regions of the search space, typically finding better configurations in fewer iterations.

## 8. Model Interpretability & Explainability

Since the project aims to help job seekers and policymakers make informed decisions, model interpretability is critical -- not just raw prediction accuracy.

**Global explanations (what drives salaries overall):**
- **Feature importance** from tree-based models (Random Forest, XGBoost) to identify top salary predictors.
- **SHAP (SHapley Additive exPlanations) summary plots** to show the direction and magnitude of each feature's impact across all predictions.
- **Lasso coefficients** from the linear baseline to highlight which features have the strongest linear relationship with salary.

**Local explanations (why a specific prediction was made):**
- **SHAP waterfall/force plots** for individual predictions, showing how each feature pushed the predicted salary up or down from the baseline.

These analyses will be documented in the evaluation notebook (`04_evaluation.ipynb`) with visualizations.

## 9. Project Structure

```
salary-prediction/
├── data/
│   └── ai_job_dataset.csv
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_preprocessing.ipynb    # Feature engineering & encoding
│   ├── 03_modeling.ipynb         # Train & compare models
│   └── 04_evaluation.ipynb       # Final evaluation, SHAP analysis & visualization
├── src/                          # Reusable Python modules (optional)
├── app/                          # Demo application (optional, see Section 11)
├── requirement.md
├── technical-documentation.md
└── README.md
```

## 10. Workflow Summary

```
Raw CSV
  → EDA (distributions, correlations, missing values, multicollinearity check)
  → Feature Engineering (skill encoding with frequency filtering, ordinal mapping, one-hot)
  → Missing Value Imputation (fitted on training set only)
  → Train/Test Split (80/20, stratified)
  → Model Training (Linear, RF, XGBoost, MLP)
  → Cross-Validation (5-fold on train set)
  → Hyperparameter Tuning (GridSearch / Optuna)
  → Test Set Evaluation (MAE, RMSE, R²)
  → Model Interpretability (SHAP, feature importance)
  → Model Comparison & Final Recommendation
```

## 11. Deployment (Optional Extension)

For demonstration or practical use, the best-performing model can be served via a lightweight web application:

- **Streamlit** (recommended for rapid prototyping): Build an interactive UI where users input job attributes and receive a predicted salary range.
- **Flask/FastAPI** (recommended for API-first use): Expose a REST endpoint for programmatic salary prediction.

The trained model should be serialized using `joblib` or `pickle`, and the preprocessing pipeline (imputers, encoders, scalers) must be saved alongside the model to ensure consistent feature transformation at inference time.
