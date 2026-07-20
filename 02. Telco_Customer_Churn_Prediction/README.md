# Telco Customer Churn Prediction

A supervised machine learning project that predicts which telecom customers are likely to churn, using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 columns).

## Problem statement

Telecom is a subscription business — every churned customer is lost recurring revenue, and replacing one costs more than retaining one. This project builds a model that outputs a per-customer probability of churning, so retention efforts (discounts, outreach, contract offers) can be targeted at the customers actually at risk, rather than applied blindly.

## Dataset

- **7,043 rows**, one per customer; **21 columns** covering demographics, subscribed services, account/contract details, and billing.
- **Target:** `Churn` (Yes/No) — binary classification.
- **Class balance:** ~73.5% stayed / ~26.5% churned (moderate imbalance, handled via class weighting — see below).

## What the notebook does, step by step

1. **Load data** — 7,043 customers, 21 columns.
2. **Data cleaning** — `TotalCharges` is stored as text with 11 hidden blank values (whitespace, not `NaN`, so `isnull()` misses them). These belong to brand-new customers with `tenure = 0`, so they're filled with `0` rather than imputed.
3. **Fix redundant categories** — Six add-on columns (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) use `"No internet service"` as a stand-in for `"No"`; collapsed to avoid an artificial extra category.
4. **Exploratory data analysis** — churn rate by contract/internet type/tenure, a numeric correlation heatmap, and tenure distribution by churn status.
5. **Feature engineering** — `IsNewCustomer` (`tenure ≤ 6`), the one cutoff tested against the actual month-by-month churn curve and kept because churn drops sharply between month 5 (48.1%) and month 6 (36.4%) — a real inflection point, not a round-number guess. (Other candidates, like a services-used count, were tested and dropped after they turned out to be collinear with existing features.)
6. **Encode categorical features** — one-hot encoding, `drop_first=True`.
7. **Train/test split** — stratified 80/20, done *before* any resampling so the test set stays untouched and realistic.
8. **Handle class imbalance** — class weighting (`class_weight='balanced'` / `scale_pos_weight`), not oversampling. SMOTE was tested separately and underperformed class weighting on every model (lower recall and ROC-AUC across the board), so it was dropped in favor of the simpler, better-performing approach.
9. **Scale numeric features** — `StandardScaler`, fit on training data only.
10. **Train models with hyperparameter tuning** — Logistic Regression, Random Forest, and XGBoost, each tuned via `RandomizedSearchCV` (5-fold stratified CV, scored on ROC-AUC) rather than fixed hyperparameters.
11. **Compare models** — side-by-side metrics table, ROC curves, Precision-Recall curves, confusion matrices.
12. **Save the best model** — selected by F1-score (the metric that best reflects churn-catching performance for this use case), along with the fitted scaler and column order.
13. **Predictive system** — a function that takes a raw new-customer record and returns a churn prediction and probability.

## Results

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** ⭐ | 0.757 | 0.528 | 0.805 | **0.638** | 0.848 |
| XGBoost | 0.750 | 0.519 | 0.805 | 0.631 | 0.847 |
| Logistic Regression | 0.741 | 0.508 | 0.794 | 0.619 | 0.845 |

**Random Forest was selected as the final model** (best F1-score), tuned to `n_estimators=400, max_depth=6, min_samples_split=2, min_samples_leaf=2, max_features='log2'`.

All three models catch **~80% of actual churners** (recall), at the cost of a real, expected trade-off: only about half of "flagged as at-risk" predictions are true churners (precision ~51–53%) — the deliberate result of prioritizing not missing churners over minimizing false alarms, appropriate for a business that would rather over-flag than under-catch.

**Overfitting check:** train/test and CV/test performance gaps are small (≤0.02 ROC-AUC) across all three tuned models — confirmed by directly comparing against an unconstrained Random Forest, which hit 99.99% training ROC-AUC but only 82.8% on test data, the textbook overfitting signature the tuned model avoids.

## Key churn drivers (from EDA)

- **Contract type** is the strongest single driver: month-to-month customers churn at 42.7% vs. 2.8% for two-year contracts.
- **Tenure**: 47.4% of churners leave within their first year; risk drops sharply after month 6.
- **Payment method**: electronic check payers churn at 45.3% vs. 15–17% for automatic payment methods.
- **Fiber optic internet** and **higher monthly bills** correlate with higher churn, alongside **missing tech support / online security** add-ons.

## Files produced when the notebook runs

| File | Contents |
|---|---|
| `churn_model.pkl` | The trained, tuned best-performing model |
| `scaler.pkl` | `StandardScaler` fit on training data (needed for Logistic Regression scoring) |
| `model_columns.pkl` | Exact column order expected by the model, for scoring new customers |
| `best_model_name.pkl` | Which model was selected |

## Requirements

```
pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, imbalanced-learn, joblib
```
Designed to run in Google Colab (`!pip install imbalanced-learn xgboost -q` in the first cell); upload `Telco-Customer-Churn.csv` to the Colab session before running.

## Possible next steps

- Threshold tuning (currently uses the default 0.5 cutoff)
- Probability calibration (`CalibratedClassifierCV`) for more reliable probability estimates
- Additional feature engineering informed by domain knowledge beyond what's in the raw columns
