# Global AI Job Market Analysis: Salary Prediction

---

## 1. Business Problem & Machine Learning Application

### The Business Problem

* **Variable Salaries:** Compensation for AI and data roles varies widely based on job type, experience level, and location.


* **Market Mismatch:** There is a significant mismatch between salary expectations and market rates, compounded by a rapidly changing market.


* **Expectation Gap:** According to the Robert Walters Salary Survey 2026, 83% of professionals expect an increment above 10% when changing jobs, while only 27% of employers are willing to provide it.


* **Difficult Decision-Making:** These factors make it difficult for both job seekers and employers to make informed decisions.



### Machine Learning Application

Supervised Machine Learning (ML) can learn complex relationships between various factors and salary. This provides the following benefits:

| Feature | Without ML | With ML |
|---|---|---|
| **Data Processing** | Manual research<br>and averaging | Learns automatically<br>from data |
| **Complexity** | Hard to handle<br>multiple factors | Handles complex<br>mixed inputs |
| **Recency** | Insights become<br>outdated quickly | Factors in<br>current indicators |
| **Insights** | Limited explanatory<br>insights | Provides interpretable<br>insights |


### Target Users & Insights

* **Primary Users (Job Seekers):** Can use the model for market salary benchmarking, career and skill planning, and salary negotiation.


* **Secondary Users (Employers/HR):** Can use the model to provide competitive salaries and assist with staffing and budget planning.



---

## 2. Dataset Overview

The project utilizes the "AI & Data Job Salaries and Skills Dataset 2024-2025," which is a synthetically prepared global overview of job listings in AI, data science, and machine learningG4_Initial_Proposal_Deck_V1.0.txt].

* **Timeframe:** October 2024 – July 2025G4_Initial_Proposal_Deck_V1.0.txt].
* **Size:** 15,000 rows and 19 columnsG4_Initial_Proposal_Deck_V1.0.txt].
* **Data Split:**
* Training set: 70%
* Validation set: 10%
* Test set: 20%



### Key Pre-Processing Steps

1. **Exploratory Data Analysis (EDA)**.
2. **Normalization of salary**.
3. **Handling missing or extreme values**.
4. **Cleaning experience data**.
5. **Encoding categorical variables and Feature Engineering**.

---

## 3. Proposed Machine Learning Models

The group intends to balance accuracy with business explainability by using the following models:

* **Linear Models:** Linear Regression, Ridge Regression, and Lasso Regression.
* **Tree-Based Models:** Decision Tree Regressor.
* **Ensemble Models:** Random Forest and Gradient Boosting.

---

## 4. Evaluation and Performance Metrics

To ensure the model is robust and logical, the following evaluations will be conducted:

### Performance Metrics

* Mean Absolute Error (MAE).
* Root Mean Squared Error (RMSE).
* Coefficient of Determination (R^2).

### Robustness & Logic Checks

* **Cross-Validation:** Use K-fold cross-validation on the training set and compare all models against a baseline linear regression.
* **Business Logic Check:** Predictions must satisfy basic market logic, such as higher experience or senior roles resulting in higher predicted salaries.

# Salary prediction
Input: country, years of experience, experience level, education, job_title, skills, company size, employment type
Output: predict annual job Salary range

# About Dataset
The location of data is ./data.
The AI & Data Job Salaries and Skills Dataset 2024-2025 is synthetically prepared and provides a detailed, global overview of
job listings in artificial intelligence, data science, machine learning, and related fields. This dataset aggregates job 
postings from October 2024 to July 2025, capturing a wide spectrum of roles, required skills, compensation, and organizational
attributes across multiple countries and industries.

With over 15,000 entries, each record details the job title, salary (in USD and local currency), experience level, employment type, 
company location and size, remote work ratio, required technical skills, education and experience requirements, industry, posting and
application dates, job description length, benefits score, and company name. This rich dataset is designed for data scientists,
career analysts, HR professionals, and researchers interested in AI workforce trends, salary benchmarking, and skills demand analysis.