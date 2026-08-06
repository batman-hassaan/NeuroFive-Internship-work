# Telco Customer Churn — Model Comparison

Predicting customer churn using four models of increasing complexity: a single Decision Tree,
Logistic Regression, and two ensemble methods (Random Forest and XGBoost), all trained and
evaluated on the same 80/20 split of the [Telco Customer Churn dataset](WA_Fn-UseC_-Telco-Customer-Churn.csv)
(7,043 customers, 26.5% churn rate).

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree | 0.7622 | 0.5399 | 0.7059 | 0.6118 | 0.8174 |
| Logistic Regression | 0.7388 | 0.5051 | 0.7968 | 0.6183 | 0.8398 |
| Random Forest | 0.7679 | 0.5444 | 0.7701 | **0.6379** | **0.8417** |
| XGBoost | 0.7516 | 0.5214 | 0.7807 | 0.6253 | 0.8393 |

**Best overall: Random Forest** (highest ROC-AUC at 0.842 and highest F1 at 0.638), narrowly ahead of
XGBoost and Logistic Regression. All three top models comfortably beat the single Decision Tree.
Logistic Regression has the single highest recall (0.797) if catching the maximum number of churners
matters more than precision.

*Class imbalance (~2.8:1 No-Churn:Churn) was handled via `class_weight='balanced'` for the
Decision Tree / Logistic Regression / Random Forest, and `scale_pos_weight` for XGBoost.*

## Top Churn Drivers (agreement across all 4 models)

1. **Contract type** — month-to-month customers churn at ~42% vs. ~3% for two-year contracts
2. **Tenure** — churn risk is highest in the first 12 months
3. **Online Security / Tech Support / Monthly Charges** — customers without these add-ons, and those on
   higher bills, churn more

## Random Forest vs. XGBoost — How They Differ

Random Forest builds many decision trees independently and in parallel, each on a random bootstrap
sample of the data and a random subset of features (*bagging*), then averages their votes — this
randomness reduces overfitting since individual trees' errors tend to cancel out. XGBoost instead builds
trees sequentially via *gradient boosting*: each new tree is trained specifically to correct the residual
errors left by the trees before it, so the ensemble improves in a directed, additive way rather than by
simple averaging. This makes Random Forest generally more robust and harder to overfit out-of-the-box,
while XGBoost is often more accurate but more sensitive to hyperparameters and easier to overfit without
careful tuning. In our feature importance plots, this shows up clearly: Random Forest spreads importance
more evenly across the top features, while XGBoost concentrates more heavily on Contract type alone.

## Files

- `Telco_Churn_Analysis.ipynb` — full notebook (EDA → Decision Tree/Logistic Regression → Random Forest/XGBoost)
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` — dataset
- `model_comparison.csv` — metrics table (machine-readable)
