# 🏦 Bank Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![SQLite](https://img.shields.io/badge/SQLite-Database-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

An End-to-End Data Science project that predicts customer attrition (churn) for a bank. This project goes beyond simple modeling by implementing a robust **ETL pipeline using SQLite**, performing rigorous **statistical validation (VIF)**, and deploying a **Balanced Logistic Regression** model to solve the class imbalance problem.

---

## 📋 Table of Contents
- [Business Problem](#-business-problem)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Methodology (ETL & Analysis)](#-methodology-etl--analysis)
- [Model Performance](#-model-performance)
- [Business Insights](#-business-insights)
- [How to Run](#-how-to-run)
- [Author](#-author)

---

## 💼 Business Problem
Customer retention is critical in the banking industry. Acquiring a new customer is significantly more expensive than retaining an existing one. 
The goal of this project is to identify customers with a high probability of leaving the bank ("Exited" = 1) so that the marketing team can proactively offer retention incentives.

**Challenge:** The dataset is highly imbalanced (most customers stay), which makes standard accuracy metrics misleading. This project focuses on optimizing **Recall** to capture as many potential churners as possible.

---

## 🚀 Key Features
* **ETL Pipeline Integration:** Automated ingestion of raw CSV data into a **SQLite database** to simulate a real-world production environment.
* **SQL-Based Data Cleaning:** Preprocessing, filtering, and initial feature engineering performed directly via SQL queries.
* **Statistical Validation:** Multicollinearity analysis using **Variance Inflation Factor (VIF)** to ensure model stability.
* **Imbalanced Data Handling:** Implementation of `class_weight='balanced'` to penalize false negatives.
* **Interpretable AI:** Extraction of coefficients to explain *why* a customer is leaving.

---

## 🛠 Tech Stack
* **Language:** Python
* **Database:** SQLite3
* **Data Manipulation:** Pandas
* **Machine Learning:** Scikit-Learn (Logistic Regression, StandardScaler)
* **Statistics:** Statsmodels (VIF Analysis)

---

## 🔬 Methodology (ETL & Analysis)

### 1. Extract, Load, Transform (ETL)
Instead of loading the CSV directly into Pandas, I implemented an ETL process:
1.  **Extract:** Read raw data from `Churn_Modelling.csv`.
2.  **Load:** Store data into a structured **SQLite** database (`banco.db`).
3.  **Transform:** SQL queries are used to fetch clean data, applying logic for feature selection and binary encoding.

### 2. Statistical Analysis
Before modeling, I verified the independence of variables using **VIF (Variance Inflation Factor)**.
* *Result:* All variables showed a VIF < 1.4, confirming no multicollinearity issues.

### 3. Model Training
* **Algorithm:** Multiple Logistic Regression.
* **Strategy:** 80/20 Train-Test split with Stratified Sampling.
* **Scaling:** `StandardScaler` applied to normalize features (Age, Balance, Salary).

---

## 📊 Model Performance

The model was optimized to detect churners (Class 1).

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **71%** | Overall correctness of the model. |
| **Recall (Class 1)** | **~70%** | **CRITICAL:** The model successfully detects 7 out of 10 customers who are about to leave. |
| **ROC-AUC** | **0.77** | Good discriminatory power between classes. |

*Note: Without balancing the classes, the Recall was only 19%. The strategic use of class weights improved detection power significantly.*

---

## 💡 Business Insights (The "Why")

Based on the model coefficients, these are the top factors driving customer churn:

1.  **Age (High Positive Impact):** Older customers are significantly more likely to leave. *Action:* Create targeted products for the senior demographic.
2.  **IsActiveMember (High Negative Impact):** Active customers are loyal. *Action:* Incentivize engagement (app usage, transaction frequency).
3.  **Geography (Germany):** Customers in Germany have a higher churn rate compared to France or Spain. *Action:* Investigate local competitors or service quality in the German region.
4.  **Gender:** Female customers showed a slightly higher tendency to churn compared to male customers.

---

## 💻 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Aldo105/Bank-Churn-Prediction.git](https://github.com/Aldo105/Bank-Churn-Prediction.git)
