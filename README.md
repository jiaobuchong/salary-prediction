# Global AI Job Salary Prediction

A machine learning project to predict salaries for AI/Data Science jobs based on skills, experience, and job attributes.

## Project Overview

This project builds regression models to predict salary (USD) for AI-related job positions. It addresses the lack of salary transparency in the AI/data field, helping job seekers with career planning and offer benchmarking.

## Dataset

- **Source**: AI & Data Job Salaries and Skills Dataset 2024-2025 (synthetic)
- **Size**: 14,999 rows x 19 columns
- **Target Variable**: `salary_usd` (range: $32,519 - $399,095)

## Project Structure

```
salary-prediction/
├── data/
│   ├── ai_job_dataset.csv          # Raw dataset
│   └── processed/                   # Preprocessed train/test splits
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb      # Feature Engineering & Encoding
│   ├── 03_modeling.ipynb           # Model Training & Comparison
│   └── 04_evaluation.ipynb         # Evaluation & SHAP Analysis
├── models/                          # Saved model artifacts
├── outputs/                         # Feature importance & reports
├── requirements.txt
├── requirement.md                   # Project requirements
├── technical-documentation.md       # Technical documentation
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the notebooks in order:

1. **01_eda.ipynb** - Explore the dataset
2. **02_preprocessing.ipynb** - Preprocess and save train/test data
3. **03_modeling.ipynb** - Train and compare models
4. **04_evaluation.ipynb** - Evaluate and interpret results

## Models Implemented

| Model | Description |
|-------|-------------|
| Ridge Regression | Linear model with L2 regularization |
| Lasso Regression | Linear model with L1 regularization |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosting |
| MLP | Neural network |

## Evaluation Metrics

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

## Key Features Used

- Job title, experience level, years of experience
- Required skills (multi-hot encoded)
- Company location, size, industry
- Education level, remote ratio, benefits score
