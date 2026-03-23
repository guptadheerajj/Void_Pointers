# Credit Risk Model — Hackathon Submission

## Overview
This notebook fixes a deliberately broken credit risk ML model.
All 14 bugs identified and corrected. Full pipeline from EDA to final evaluation.

## Results
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.9172 |
| Recall | 0.5888 |
| Precision | 0.7975 |
| F1-Score | 0.6774 |
| Accuracy | 0.9774 |

## Files
| File | Description |
|------|-------------|
| `FIXED_Credit_Risk_Model.ipynb` | Main fixed notebook |
| `credit_risk_dataset.csv` | Dataset (13,266 rows × 20 cols) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Setup & Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place credit_risk_dataset.csv in the same folder

# 3. Open and run the notebook
jupyter notebook FIXED_Credit_Risk_Model.ipynb
```

## All 14 Bugs Fixed
| # | Bug | Fix |
|---|-----|-----|
| 1 | SimpleImputer wrong import | Moved to `sklearn.impute` |
| 2 | Wrong file path | Fixed to `credit_risk_dataset.csv` |
| 3 | Data leakage — post-outcome cols | Dropped before any processing |
| 4 | Feature selection on full data | Done after split on train only |
| 5 | Preprocessor fit on train+test | `fit` on train, `transform` on test |
| 6 | Hyperparameter tuning on test set | GridSearchCV with StratifiedKFold |
| 7 | Threshold tuning on test set | Tuned on validation fold |
| 8 | Normalization stats from full data | Computed via pipeline on train only |
| 9 | More leakage features (section 3) | All post-outcome features dropped |
| 10 | Wrong column name `annual_income` | Corrected to `annual_inc` |
| 11 | `auc` variable shadows `auc()` function | Renamed to `auc_score` |
| 12 | Missing metric imports | Added `precision_score`, `recall_score`, `brier_score_loss` |
| 13 | Feature importance mismatch after OHE | Used `get_feature_names_out()` |
| 14 | Undefined `tp_y`/`fn_y` variables | Replaced with `recall_score()` on subset |

## Why ROC-AUC is the Correct Metric
A model predicting "no default" for everyone achieves 96% accuracy for free.
Accuracy is completely misleading on a 96/4 imbalanced dataset.
ROC-AUC measures discrimination across all thresholds — robust to class imbalance.

## Pipeline Steps
1. Load Data
2. EDA (distributions, missing values, outliers, correlations)
3. Data Cleaning (drop leakage, fix case, cap outliers)
4. Feature Engineering (debt-to-income, credit-loan ratio)
5. Train/Test Split (stratified, 80/20)
6. Preprocessing (median impute + StandardScaler on train only)
7. SMOTE (on training data only)
8. Model Training (Logistic Regression, Random Forest, Gradient Boosting)
9. Threshold Tuning (on validation folds, not test)
10. Final Evaluation (test set used once)
