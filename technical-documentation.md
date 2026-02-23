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

## 4. Data Preprocessing

1. **Handle `required_skills`:** Multi-hot encode the comma-separated skills into binary columns (one per unique skill).
2. **Encode categoricals:**
   - `education_required` -- ordinal encode (Associate=0, Bachelor=1, Master=2, PhD=3).
   - `experience_level` -- ordinal encode (EN=0, MI=1, SE=2, EX=3).
   - `job_title`, `company_location`, `industry`, `employment_type`, `company_size`, `remote_ratio` -- one-hot encode or target encode depending on model.
3. **Scale numericals:** Standardize `years_experience` and `benefits_score` for distance-based models.
4. **Train/test split:** 80/20 stratified split (stratify by `experience_level` or `job_title` to preserve distribution).

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

## 7. Project Structure

```
salary-prediction/
├── data/
│   └── ai_job_dataset.csv
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_preprocessing.ipynb    # Feature engineering & encoding
│   ├── 03_modeling.ipynb         # Train & compare models
│   └── 04_evaluation.ipynb       # Final evaluation & visualization
├── src/                          # Reusable Python modules (optional)
├── requirement.md
├── technical-documentation.md
└── README.md
```

## 8. Workflow Summary

```
Raw CSV
  -> EDA (distributions, correlations, missing values)
  -> Feature Engineering (skill encoding, ordinal mapping, one-hot)
  -> Train/Test Split (80/20)
  -> Model Training (Linear, RF, XGBoost, MLP)
  -> Cross-Validation (5-fold on train set)
  -> Hyperparameter Tuning (GridSearch / RandomSearch)
  -> Test Set Evaluation (MAE, RMSE, R²)
  -> Model Comparison & Final Recommendation
```
