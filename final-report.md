# AI Job Salary Prediction: Final Technical Report

**Course:** Applied Machine Learning
**Date:** February 2026
**Dataset:** AI Job Salary Dataset (15,000 records)

---

## 1. Executive Summary

This report presents a machine learning approach to predicting salaries for AI and data science positions. Using a dataset of 15,000 job postings with 19 features, we developed and compared six regression models: Linear Regression, Ridge Regression, Lasso Regression, Decision Tree, Random Forest, and Gradient Boosting.

After systematic preprocessing — including IQR-based outlier winsorization, log transformation of the target variable, multi-hot skill encoding, and ordinal/one-hot categorical encoding — we trained all models on 94 engineered features and evaluated them on a held-out test set.

**Gradient Boosting emerged as the best-performing model**, achieving a test set MAE of $13,183, RMSE of $17,452, and R² of 0.9053. This represents a 1.37% improvement over the Linear Regression baseline on the validation set. All models except Random Forest passed business logic validation checks for monotonic salary-experience relationships. We recommend Gradient Boosting for deployment, with the caveat that the dataset is synthetically generated and the model should be validated on real-world salary data before production use.

---

## 2. Introduction & Problem Statement

### Business Context

Salary prediction is a critical challenge for HR departments, job seekers, and recruitment platforms. Accurate salary estimates help organizations remain competitive in talent acquisition, assist candidates in negotiating fair compensation, and enable job platforms to provide transparent salary ranges. In the rapidly growing AI and data science sector, where roles span from entry-level data analysts to executive-level AI architects, salary ranges vary widely based on experience, skills, location, and company characteristics.

### Machine Learning Value Proposition

Traditional salary benchmarking relies on static survey data and manual analysis. A machine learning approach offers several advantages: (1) the ability to learn complex nonlinear relationships between features and salary, (2) automatic incorporation of multi-dimensional interactions (e.g., how the combination of specific skills, experience level, and location jointly affect compensation), and (3) scalability to continuously update predictions as new data becomes available.

### Target Users

This model serves three primary audiences: **HR professionals** seeking data-driven salary benchmarks for job postings, **job seekers** estimating expected compensation for AI roles, and **recruitment platforms** providing salary range estimates for listed positions.

---

## 3. Dataset Description

### Source and Overview

The dataset contains 15,000 AI job postings with 19 features. It was loaded with zero missing values across all columns, eliminating the need for imputation. The dataset covers 20 job titles, 20 countries, and 15 industries, with roughly balanced distributions across experience levels, education requirements, company sizes, and remote work ratios.

### Feature Summary

| Feature | Type | Description | Unique Values |
|---------|------|-------------|---------------|
| `job_id` | Identifier | Unique job identifier | 15,000 |
| `job_title` | Categorical | Role title (e.g., Data Scientist, NLP Engineer) | 20 |
| `salary_usd` | Numerical | **Target variable** — annual salary in USD | Continuous |
| `salary_currency` | Categorical | Original currency of posting | Multiple |
| `experience_level` | Ordinal | EN (Entry), MI (Mid), SE (Senior), EX (Executive) | 4 |
| `employment_type` | Categorical | FT, PT, CT, FL (Full-time, Part-time, Contract, Freelance) | 4 |
| `company_location` | Categorical | Country of the company | 20 |
| `company_size` | Ordinal | S (Small), M (Medium), L (Large) | 3 |
| `employee_residence` | Categorical | Country of the employee | Multiple |
| `remote_ratio` | Categorical | 0 (On-site), 50 (Hybrid), 100 (Remote) | 3 |
| `required_skills` | Text (list) | Comma-separated skill requirements | 24 unique skills |
| `education_required` | Ordinal | Associate, Bachelor, Master, PhD | 4 |
| `years_experience` | Numerical | Years of professional experience (0–19) | 20 |
| `industry` | Categorical | Sector of the company | 15 |
| `posting_date` | Date | Date the job was posted | — |
| `application_deadline` | Date | Application closing date | — |
| `job_description_length` | Numerical | Character count of job description | Continuous |
| `benefits_score` | Numerical | Benefits rating (5.0–10.0) | Continuous |
| `company_name` | Categorical | Name of the employer | Multiple |

### Target Variable Statistics

| Statistic | Value |
|-----------|-------|
| Count | 15,000 |
| Mean | $115,349 |
| Std Dev | $60,261 |
| Min | $32,519 |
| 25th Percentile | $70,180 |
| Median | $99,705 |
| 75th Percentile | $146,409 |
| Max | $399,095 |

The salary distribution is right-skewed (skewness = 0.937), with the majority of salaries concentrated between $50,000 and $175,000 and a long tail extending toward $400,000.

### Key Distributions

- **Experience levels** are nearly uniformly distributed: EN (3,718), MI (3,781), SE (3,741), EX (3,760).
- **Education requirements** are similarly balanced: Bachelor (3,789), Associate (3,785), Master (3,748), PhD (3,678).
- **Company sizes** are evenly split: S (5,007), M (4,995), L (4,998).
- **Top skills by frequency**: Python (4,450), SQL (3,407), TensorFlow (3,022), Kubernetes (3,009), Scala (2,794).

---

## 4. Data Preprocessing & Feature Engineering

### 4.1 Dropped Columns

Seven columns were removed before modeling:

| Dropped Column | Rationale |
|----------------|-----------|
| `job_id` | Unique identifier with no predictive value |
| `salary_currency` | Salaries already normalized to USD |
| `employee_residence` | Highly correlated with `company_location`; redundant |
| `posting_date` | Temporal feature not relevant for cross-sectional salary prediction |
| `application_deadline` | Derived from posting date; no independent signal |
| `job_description_length` | Weak predictor; character count does not meaningfully relate to salary |
| `company_name` | High cardinality identifier; risk of overfitting |

This reduced the working dataset from 19 to 12 columns (11 features + 1 target).

### 4.2 Outlier Handling (IQR Winsorization)

We applied IQR-based winsorization to the target variable (`salary_usd`) to reduce the influence of extreme values without discarding data:

- **IQR bounds**: Q1 = $70,180, Q3 = $146,409, IQR = $76,229
- **Lower bound**: $-44,163 (no values below)
- **Upper bound**: $260,752
- **Outliers capped**: 483 salaries above $260,752 were clipped to the upper bound
- **Post-capping stats**: Mean = $114,125, Std = $56,554, Max = $260,752

### 4.3 Log Transformation of Target

A `log1p` transformation was applied to normalize the right-skewed salary distribution:

- **Before**: Skewness = 0.937, Mean = $114,125
- **After**: Skewness = 0.053, Mean = 11.5274 (log scale)

All models were trained on log-transformed salary. Predictions were inverse-transformed via `expm1` back to dollar scale for evaluation, ensuring metrics are interpretable in real-world terms.

### 4.4 Skills Multi-Hot Encoding

The `required_skills` column contained comma-separated skill lists. We applied multi-hot (binary) encoding with a 2% frequency threshold:

- **24 unique skills identified** — all 24 met the 2% threshold
- Each skill became a binary feature (e.g., `skill_python`, `skill_sql`, `skill_tensorflow`)
- An additional `other_skills_count` feature captured the count of rare skills per posting
- **Result**: 25 skill-derived features (24 binary + 1 count)

### 4.5 Categorical Encoding

**Ordinal encoding** was applied to features with a natural order:

| Feature | Mapping |
|---------|---------|
| `experience_level` | EN → 0, MI → 1, SE → 2, EX → 3 |
| `education_required` | Associate → 0, Bachelor → 1, Master → 2, PhD → 3 |

**One-hot encoding** was applied to nominal categorical features: `job_title` (20 categories), `company_location` (20), `industry` (15), `employment_type` (4), `company_size` (3), and `remote_ratio` (3). This produced 65 binary columns.

### 4.6 Final Feature Matrix

| Category | Count |
|----------|-------|
| Numerical (years_experience, benefits_score) | 2 |
| Ordinal encoded (experience_level, education) | 2 |
| One-hot encoded (job title, location, industry, etc.) | 65 |
| Skills encoded (24 binary + 1 count) | 25 |
| **Total** | **94** |

### 4.7 Feature Scaling

StandardScaler was applied to the three continuous features (`years_experience`, `benefits_score`, `other_skills_count`), fitted exclusively on the training set to prevent data leakage. Binary and ordinal features were left unscaled. Scaled data was used for linear models (LR, Ridge, Lasso); unscaled data was used for tree-based models (Decision Tree, Random Forest, Gradient Boosting).

### 4.8 Train / Validation / Test Split

A two-stage stratified split (stratified by `experience_level`) produced:

| Set | Samples | Proportion |
|-----|---------|------------|
| Training | 10,500 | 70% |
| Validation | 1,500 | 10% |
| Test | 3,000 | 20% |

The validation set was used for hyperparameter tuning and model selection. The test set was held out for final unbiased evaluation.

---

## 5. Model Training & Hyperparameter Tuning

Six models were trained, progressing from simple linear baselines to ensemble methods. Linear models used scaled features; tree-based models used unscaled features. All models were trained on log-transformed salary.

### 5.1 Linear Regression (Baseline)

Ordinary least squares with no regularization, serving as the baseline for comparison.

- **Hyperparameters**: None (closed-form solution)
- **Validation MAE**: $13,318 | **R²**: 0.8985

### 5.2 Ridge Regression

L2-regularized linear regression. Hyperparameters tuned via 5-fold GridSearchCV:

- **Search grid**: alpha ∈ {0.01, 0.1, 1.0, 10.0, 100.0}
- **Best alpha**: 0.1
- **Validation MAE**: $13,318 | **R²**: 0.8985

### 5.3 Lasso Regression

L1-regularized linear regression with built-in feature selection. Hyperparameters tuned via 5-fold GridSearchCV:

- **Search grid**: alpha ∈ {0.0001, 0.001, 0.01, 0.1, 1.0}
- **Best alpha**: 0.0001
- **Validation MAE**: $13,310 | **R²**: 0.8987
- **Non-zero coefficients**: 81 out of 94 (13 features zeroed out)

### 5.4 Decision Tree

Single CART regressor. Hyperparameters tuned via 5-fold GridSearchCV over 64 combinations:

- **Search grid**: max_depth ∈ {5, 10, 15, 20}, min_samples_split ∈ {2, 5, 10, 20}, min_samples_leaf ∈ {1, 2, 4, 8}
- **Best parameters**: max_depth=15, min_samples_split=20, min_samples_leaf=8
- **Validation MAE**: $14,138 | **R²**: 0.8830

### 5.5 Random Forest

Bagged ensemble of decision trees. Hyperparameters tuned via 5-fold RandomizedSearchCV (20 iterations):

- **Search grid**: n_estimators ∈ {100, 200}, max_depth ∈ {10, 20, 30}, min_samples_split ∈ {2, 5, 10}, min_samples_leaf ∈ {1, 2, 4}
- **Best parameters**: n_estimators=100, max_depth=20, min_samples_split=10, min_samples_leaf=4
- **Validation MAE**: $13,754 | **R²**: 0.8929

### 5.6 Gradient Boosting

Sequential boosting ensemble. Hyperparameters tuned via 5-fold RandomizedSearchCV (20 iterations):

- **Search grid**: n_estimators ∈ {100, 200}, max_depth ∈ {3, 5, 7}, learning_rate ∈ {0.01, 0.05, 0.1}, subsample ∈ {0.8, 1.0}, min_samples_split ∈ {2, 5, 10}, min_samples_leaf ∈ {1, 2, 4}
- **Best parameters**: n_estimators=200, max_depth=3, learning_rate=0.1, subsample=1.0, min_samples_split=2, min_samples_leaf=1
- **Validation MAE**: $13,136 | **R²**: 0.9030

### Validation Set Results Summary

| Rank | Model | MAE ($) | RMSE ($) | R² |
|------|-------|---------|----------|-----|
| 1 | **Gradient Boosting** | **13,136** | **17,319** | **0.9030** |
| 2 | Lasso Regression | 13,310 | 17,696 | 0.8987 |
| 3 | Ridge Regression | 13,318 | 17,717 | 0.8985 |
| 4 | Linear Regression | 13,318 | 17,717 | 0.8985 |
| 5 | Random Forest | 13,754 | 18,196 | 0.8929 |
| 6 | Decision Tree | 14,138 | 19,022 | 0.8830 |

---

## 6. Model Comparison & Baseline Analysis

### Improvement vs. Linear Regression Baseline

| Model | MAE Improvement ($) | MAE Improvement (%) | R² Improvement |
|-------|---------------------|---------------------|----------------|
| Gradient Boosting | +182 | +1.37% | +0.0045 |
| Lasso Regression | +8 | +0.06% | +0.0003 |
| Ridge Regression | +0.45 | +0.00% | +0.0001 |
| Random Forest | -436 | -3.27% | -0.0056 |
| Decision Tree | -820 | -6.16% | -0.0155 |

The linear models (Ridge, Lasso) performed nearly identically to the baseline, suggesting the feature space is well-conditioned and L2/L1 regularization offers minimal benefit. Decision Tree and Random Forest underperformed the baseline on the validation set, likely due to the uniformly distributed nature of the synthetic data which limits the advantage of tree-based partitioning. Gradient Boosting, with its sequential error-correction mechanism, was the only model to materially improve upon the baseline.

### Test Set Final Results

The top three models were evaluated on the held-out test set (3,000 samples) for unbiased performance assessment:

| Model | Test MAE ($) | Test RMSE ($) | Test R² |
|-------|-------------|--------------|---------|
| **Gradient Boosting** | **13,183** | **17,452** | **0.9053** |
| Ridge Regression | 13,273 | 17,796 | 0.9015 |
| Random Forest | 13,414 | 17,783 | 0.9017 |

Gradient Boosting maintained its lead on the test set with the lowest MAE ($13,183) and highest R² (0.9053), confirming its generalization capability. The test set metrics are consistent with validation set results, indicating no overfitting.

### Recommendation

**Gradient Boosting is the recommended model** for deployment. It achieves the best performance across all three metrics (MAE, RMSE, R²), passes all business logic validation checks, and its shallow tree depth (max_depth=3) provides implicit regularization against overfitting.

---

## 7. Feature Importance & Interpretability

### 7.1 Tree-Based Feature Importance

Feature importance scores from the three tree-based models consistently identify the same top predictors:

**Top 10 Features (Random Forest importance scores)**:

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `years_experience` | 0.350 |
| 2 | `experience_level_enc` | 0.342 |
| 3 | `company_location_Switzerland` | 0.039 |
| 4 | `company_size_L` | 0.032 |
| 5 | `company_location_Denmark` | 0.031 |
| 6 | `company_location_Norway` | 0.030 |
| 7 | `company_location_United States` | 0.021 |
| 8 | `company_location_United Kingdom` | 0.013 |
| 9 | `company_location_Singapore` | 0.013 |
| 10 | `company_location_Netherlands` | 0.013 |

Experience-related features (`years_experience` and `experience_level_enc`) dominate with a combined importance of 0.692, accounting for nearly 70% of the model's predictive power. Company location is the next most important factor, with high-cost-of-living countries (Switzerland, Denmark, Norway, United States) showing the strongest signals.

### 7.2 Lasso Coefficient Analysis

Lasso regression retained 81 of 94 features (zeroing out 13), providing a form of automatic feature selection. The top coefficients reveal which features have the strongest linear relationship with log-salary:

**Positive impact (higher salary)**:
- `company_location_Switzerland`: +0.361
- `experience_level_enc`: +0.326
- `company_location_Denmark`: +0.281
- `company_location_Norway`: +0.277
- `company_location_United States`: +0.199
- `company_size_L`: +0.133

**Negative impact (lower salary)**:
- `company_location_India`: -0.316
- `company_location_Austria`: -0.311
- `company_location_Japan`: -0.307
- `company_location_Finland`: -0.303
- `company_location_China`: -0.303
- `company_size_S`: -0.102

### 7.3 SHAP Analysis

SHAP (SHapley Additive exPlanations) values were computed for the Random Forest model on a 500-sample subset of the test set. The SHAP analysis confirms the feature importance rankings: `years_experience` and `experience_level_enc` have the highest mean absolute SHAP values, followed by country-level location features.

The SHAP summary plot reveals the directionality of feature effects:
- Higher values of `years_experience` and `experience_level_enc` consistently push predictions upward
- Location in Switzerland, Denmark, and Norway increases predicted salary
- Location in lower-cost countries (India, China, Austria) decreases predicted salary

### 7.4 Business Logic Validation

Two monotonicity tests were conducted to ensure models produce economically sensible predictions:

| Model | Test 1: Salary increases with years_experience | Test 2: Salary increases with experience_level |
|-------|-----------------------------------------------|------------------------------------------------|
| Linear Regression | PASS | PASS |
| Ridge Regression | PASS | PASS |
| Lasso Regression | PASS | PASS |
| Decision Tree | PASS | PASS |
| Random Forest | **FAIL** | PASS |
| Gradient Boosting | PASS | PASS |

- **Test 1** (years_experience: 0 → 19): 5 of 6 models passed. Random Forest showed a non-monotonic response due to the averaging behavior of its constituent trees.
- **Test 2** (experience_level: EN → MI → SE → EX): All 6 models passed.

Gradient Boosting passed both tests, further supporting its selection as the recommended model. For applications requiring strict monotonicity guarantees, monotonic constraints can be added during training.

---

## 8. Conclusions & Recommendations

### Best Model Summary

Gradient Boosting achieved the best overall performance with a test set MAE of $13,183, RMSE of $17,452, and R² of 0.9053. On average, predictions are within approximately $13,000 of the actual salary, and the model explains over 90% of salary variance. It passed all business logic validation checks, confirming that its predictions follow expected economic patterns (salary increases with experience).

### Deployment Recommendation

We recommend deploying the Gradient Boosting model with the following pipeline:
1. **Input preprocessing**: Apply the same feature engineering pipeline (skills multi-hot encoding, ordinal encoding, one-hot encoding) using the saved preprocessing artifacts
2. **Prediction**: Pass the 94-feature vector through the trained Gradient Boosting model
3. **Output**: Apply `expm1` inverse transformation to convert log-scale predictions back to dollar amounts
4. **Monitoring**: Track prediction distributions and retrain periodically with updated salary data

### Limitations

- **Synthetic data**: The dataset is synthetically generated with nearly uniform distributions across categorical features (experience levels, education, company size). Real-world salary data exhibits more skewed distributions and complex interactions.
- **Geographic scope**: The 20-country scope may not generalize to markets not represented in the training data.
- **Temporal validity**: Salary benchmarks shift over time; the model should be retrained periodically.
- **Feature constraints**: The model does not account for negotiation dynamics, equity compensation, or cost-of-living adjustments.

### Future Work

1. **Real-world validation**: Evaluate the model on verified salary data (e.g., from Glassdoor, Levels.fyi, or Bureau of Labor Statistics)
2. **Additional features**: Incorporate cost-of-living indices, company revenue/funding stage, and job-specific requirements
3. **Advanced models**: Explore XGBoost, LightGBM, or neural network approaches for potential performance gains
4. **Monotonic constraints**: Apply monotonic constraints to tree-based models to guarantee business logic compliance
5. **Confidence intervals**: Implement prediction intervals to communicate uncertainty alongside point estimates

---

*This report was generated from analysis conducted across four Jupyter notebooks: EDA (`01_eda`), Preprocessing (`02_preprocessing`), Modeling (`03_modeling`), and Evaluation (`04_evaluation`).*
