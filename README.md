# Student Dropout Prediction — Data Mining Project

End-to-end machine learning project following the **CRISP-DM methodology** to predict student academic outcomes (Dropout / Enrolled / Graduate) using the UCI Student Academic Success dataset.

**Course:** Data Mining Techniques & Applications — Sarajevo School of Science and Technology
**Authors:** Selma Karačić, Zerina Kulić
**Period:** December 2025 – January 2026

## Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- **Size:** 4,424 students · 36 input features · 1 target variable
- **Target classes:** Dropout (32%) · Graduate (50%) · Enrolled (18%)
- **Missing values:** None

Features cover demographics, academic performance (1st and 2nd semester), family background, socio-economic indicators, and macro-economic data.

## Notebooks

| Notebook | Description |
|---|---|
| `01_Data_Understanding_EDA.ipynb` | Exploratory data analysis — 20+ visualizations including univariate, bivariate, multivariate, and target distribution analysis |
| `02_Data_Preparation.ipynb` | Data cleaning, feature engineering (12 new features), feature selection, outlier handling, SMOTE balancing, train/test split |
| `03_Modeling_Classification.ipynb` | Model training, hyperparameter tuning (GridSearchCV), evaluation, and comparison of three classifiers |

## Pipeline

1. **EDA** — distribution analysis, correlation heatmap, hypothesis formation
2. **Feature Engineering** — 12 engineered features (performance trends, interaction terms)
3. **Feature Selection** — removed 2 redundant features based on correlation and importance
4. **Outlier Handling** — removed 59 outliers via IQR method
5. **Class Imbalance** — SMOTE applied to training set to handle 32/50/18 class split
6. **Train/Test Split** — 80/20 stratified split, leakage prevention enforced
7. **Modeling** — Decision Tree, k-Nearest Neighbours, Naïve Bayes
8. **Evaluation** — Accuracy, Macro F1, Confusion Matrix, per-class metrics, fairness/bias checks

## Models Compared

| Model | Notes |
|---|---|
| Decision Tree (CART/ID3) | Interpretable, feature importance available, tuned via GridSearchCV |
| k-Nearest Neighbours | Distance-based, StandardScaler applied, k optimized |
| Naïve Bayes | Lightweight baseline, Bayesian approach |

## Tech Stack

- Python 3.13
- Jupyter Notebook (Anaconda)
- pandas · NumPy · scikit-learn · imbalanced-learn (SMOTE)
- Matplotlib · Seaborn
- SciPy

## Results

See `graphs/` for all generated visualizations including confusion matrices, feature importance plots, decision tree structure, and model comparison charts.
