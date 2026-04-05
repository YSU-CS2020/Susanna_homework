# Assignment 3 — Decision Trees & Ensemble Methods

**YSU CS2020 · Machine Learning** **Student:** Susanna Mkrtchyan

---

## Learning objectives

| # | Concept | Implementation Location |
|---|---------|-------------------|
| 1 | Gini Impurity & Info Gain | `decision_tree.py` → `_gini()` & `_information_gain()`|
| 2 | Recursive Tree Splitting | `decision_tree.py` → `_best_split()` |
| 3 | Bootstrap Aggregation (Bagging) | `random_forest.py` → `_bootstrap_sample()` |
| 4 | Feature Randomness (Subspacing) | `random_forest.py` → `fit()` |
| 5 | Bias-Variance Trade-off Analysis | `experiments.py` → Exp 2.2 |
| 6 | Feature Importance Ranking | `random_forest.py` → `get_feature_importance()` |

---

## Project structure

```text
assignment3_susanna_mkrtchyan/
├── code/
│   ├── decision_tree.py           ← Manual ID3/CART implementation
│   ├── random_forest.py           ← Ensemble of Decision Trees (Bagging)
│   ├── experiments.py             ← Main entry point for all 3 experiments
│   ├── requirements.txt           ← Dependencies (numpy, pandas, etc.)
│   └── README.md                  ← Documentation (this file)
│   └── winequality-red.csv        ← Primary dataset
│
├── report/
│   └── report_assignment3.pdf      ← Technical analysis and findings
│       
├── figures/                       ← Generated plots (15 figures)
│   ├── test_accuracy_comparison.png
│   ├── dt_bias_variance_depth.png
│   └── feature_importance.png
│
└── dataset_info.txt               ← Technical attribute descriptions
```

---

## Mathematical Reference

### Entropy
$$H(S) = -\sum p_i \log_2(p_i)$$

### Gini Impurity
$$Gini(S) = 1 - \sum p_i^2$$

### Random Forest Prediction
$$\hat{y} = \text{mode}\{T_1(x), T_2(x), \dots, T_n(x)\}$$

---

## Quick Start

```bash
# Install dependencies
pip install -r code/requirements.txt

# Run experiments
cd code
python experiments.py

[Read the Full Report (PDF)](./report/report_assignment3_susanna_mkrtchyan.pdf)