# Portuguese Bank Marketing — Term Deposit Subscription Prediction

**Certified Data Scientist Capstone (PRCP-1000)** | Bhingradiya Vasu

Predicting whether a bank client will subscribe to a term deposit, using the UCI Bank Marketing dataset (`bank-additional-full.csv`: 41,188 clients × 21 columns).

## Project structure

```
portuguese-bank-marketing/
├── README.md                  # this file
├── requirements.txt           # Python dependencies
├── data/
│   └── bank-additional-full.csv
├── notebooks/
│   └── Portuguese_Bank_Marketing_PRCP1000.ipynb   # final notebook, executed
├── models/
│   └── Portuguese_final_model.pkl                 # saved final model
└── reports/
    ├── Portuguese_Project_Analysis.txt            # ready-to-paste portal text
    └── Portuguese_model_comparison_final.csv      # full comparison table
```

## Dataset

- **Source:** UCI Bank Marketing (`bank-additional-full.csv`), 41,188 rows × 21 columns.
- **Target:** `y` — no 88.73% / yes 11.27% (only 4,640 subscribers, so the data is imbalanced).
- **Cleaning:** 12 duplicate rows removed; no blank cells — missingness hides inside `unknown` categories (e.g. `default` is 20.87% unknown), kept as their own category.
- **Leakage found:** `duration` (call length) looks like the strongest signal (subscribers averaged 553 sec vs 221 sec for non-subscribers), but it is known only after the call ends, so it was **dropped before modelling**.

## Approach

1. **Task 1 — Data analysis:** subscriber rate is highest after a previous campaign success (`poutcome=success`: 65.1%), and for students (31.4%) and retired clients (25.2%); lowest for blue-collar workers (6.9%).
2. **Task 2 — Classification:** Logistic Regression vs Random Forest vs Gradient Boosting on the same stratified 80/20 split (test = 8,236 clients, 928 subscribers), judged on recall / F1 / AUC — not accuracy, because predicting "no" for everyone already scores 88.73%.
3. **Task 3 — Imbalance handling:** `class_weight='balanced'` plus threshold tuning (0.3–0.7). Balancing alone lifted Logistic Regression recall from 21.01% to 64.55%.

## Results

| Model | Accuracy | Precision (yes) | Recall (yes) | F1 (yes) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression (unbalanced) | 89.83% | 65.00% | 21.01% | 31.76% | 0.8004 |
| Random Forest (unbalanced) | 89.25% | 54.31% | 29.20% | 37.98% | 0.7755 |
| LR balanced @0.5 | 83.03% | 35.91% | 64.55% | 46.15% | 0.8002 |
| RF balanced @0.5 | 87.36% | 44.18% | 46.23% | 45.18% | 0.7851 |
| **Gradient Boosting (balanced) @0.5 — FINAL** | **84.85%** | **39.61%** | **65.73%** | **49.43%** | **0.8152** |

**Final pick: Gradient Boosting (balanced) at threshold 0.5** — F1 49.43%, ROC-AUC 0.8152, recall 65.73% (610 of 928 subscribers found), confusion matrix [[TN 6378, FP 930], [FN 318, TP 610]]. A small hyperparameter check (max_iter=200, lr=0.05: F1 49.08%) gave no gain, so defaults were kept. Top drivers by permutation importance: `nr.employed`, `euribor3m`, `cons.conf.idx`.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/Portuguese_Bank_Marketing_PRCP1000.ipynb
```

Open the notebook from the `notebooks/` folder and Run All — it finds the dataset in `data/` by itself. No GPU needed (runs in about a minute on CPU).
