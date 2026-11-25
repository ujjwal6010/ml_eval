# Prediction When Features Outnumber Samples

**Team:** Byte Busters, Batch A1  
**Course:** Machine Learning Lab (AIP301)  
**Institution:** Indian Institute of Information Technology Kota  
**Instructor:** Dr. Shibna Hussain Ma'am  
**Submission Date:** November 2026

**Team Members:**  
1. Ujjwal Garg - 2024KUCP1001  
2. Shaurya Singh - 2024KUCP1003  
3. G. Vishnu Vardhan - 2024KUCP1004  
4. Anupam Saxena - 2024KUCP1006  
---

## Overview

This project investigates regression model behavior when feature count (p) approaches or exceeds sample count (n) — the classic "p > n" problem common in genomics and high-dimensional data analysis.

**Research Question:** How do different regression models perform when p ≥ n, and what strategies help maintain reliable predictions?

## Dataset

**Name:** Gene Expression Cancer (Golub et al.)
**Source:** Kaggle (crawford/gene-expression)  
**Original Paper:** https://www.kaggle.com/datasets/crawford/gene-expression

| Property | Value |
|---|---|
| Samples (n) | 72 |
| Features (p) | 7,129 genes |
| p/n ratio | 99.0 (high-dimensional!) |
| Target | Cancer type (ALL=0, AML=1) |
| Task | Binary classification |

**Dataset:** Famous Golub et al. leukemia gene expression dataset from Science 1999. Perfect example of p >> n problem in genomics.

## Project Structure

```
ByteBusters_Submission/
|-- README.md                    # This file
|-- requirements.txt             # Python dependencies
|-- environment.txt              # Environment details
|-- data/
|   |-- raw/                     # Raw dataset (auto-downloaded)
|   `-- processed/               # Data dictionary
|-- notebooks/
|   `-- experiments.ipynb        # Main experiment notebook
|-- src/
|   |-- data_loader.py          # Dataset loading utilities
|   |-- data_audit.py           # Data quality checks
|   |-- preprocessing.py        # Data preprocessing
|   |-- feature_engineering.py  # Feature manipulation
|   |-- models.py               # Model definitions
|   |-- evaluation.py           # Metrics and evaluation
|   |-- experiments.py          # Experiment logging
|   `-- visualization.py        # Plotting functions
|-- results/
|   |-- experiment_log.csv      # Detailed experiment results
|   |-- final_results.csv       # Summary of model performance
|   |-- summary.json            # JSON summary
|   `-- figures/                # Generated plots
|-- report/
|   |-- final_report.md         # Typed 4-6 page report
|   `-- model_card.md           # Model documentation
`-- reflection/
    `-- reflection_draft.md     # Handwritten reflection draft
```

## Installation

**Step 1:** Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
```

**Step 2:** Install dependencies
```bash
pip install -r requirements.txt
```

**Step 3:** Run the notebook
```bash
jupyter notebook notebooks/experiments.ipynb
```

The dataset downloads automatically on first run. All results save to the `results/` folder.

## How to Run

**Option A — Notebook (Recommended):**
1. Open `notebooks/experiments.ipynb`
2. Run all cells top-to-bottom
3. Results appear in `results/` folder

**Option B — Python Scripts:**
```bash
cd src
python -c "from data_loader import load_dataset; X, y, _, _ = load_dataset(); print(f'Loaded: {X.shape}')"
```

## Experiments Conducted

1. **Data Audit** — Missing values, duplicates, target correlations, potential leakage, and an optional all-pairs correlation scan
2. **Feature-to-Sample Ratios** — p/n < 1, ≈ 1, > 1
3. **Model Comparison** — Dummy, Linear, Ridge, Lasso, Random Forest
4. **Complexity Control** — Ridge/Lasso regularization
5. **Feature Selection** — SelectKBest and PCA
6. **Nested Cross-Validation** — Inner tuning + outer evaluation
7. **Repeated Splits** — 10 train/test splits for stability
8. **Learning Curves** — Effect of training set size
9. **Noise Experiment** — Adding irrelevant features
10. **Redundancy Experiment** — Adding correlated features
11. **Target Permutation** — Leakage sanity check
12. **Feature Stability** — Selection consistency across splits
13. **Feature Ablation** — Impact of removing key features

## Key Results

| Model | Mean R² | Std R² | Mean RMSE | Mean MAE |
|---|---:|---:|---:|---:|
| Random Forest | 0.987 | 0.008 | 18.9 | 9.7 |
| Lasso | 0.978 | 0.008 | 24.9 | 15.0 |
| Ridge | 0.965 | 0.007 | 31.8 | 21.4 |
| Dummy | ~0.0 | — | — | — |

**Key Findings:**
- Regularized models handle p > n effectively
- More training data significantly improves validation performance
- Irrelevant features degrade performance
- Ridge tolerates redundant features well
- Target permutation confirms no data leakage

## Reproducibility

- **Random Seed:** 42 (fixed throughout)
- **Results:** All metrics logged in `results/experiment_log.csv`
- **Figures:** Saved in `results/figures/`

## Limitations

- Gene expression dataset from Kaggle (crawford/gene-expression)
- Cross-sectional gene expression data
- Small sample size (n=72) typical for genomics studies
- Some features highly correlated with target (potential near-leakage)

## Documentation

- **Final Report:** `report/final_report.md` (4-6 pages, typed)
- **Model Card:** `report/model_card.md`
- **Reflection:** `reflection/reflection_draft.md` (to be handwritten for submission)

---

**Contact:** Byte Busters Team — IDs: 1001, 1003, 1004, 1006\
**Course:** AIP301, IIIT Kota



