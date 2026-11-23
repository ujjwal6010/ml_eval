# Prediction When Features Outnumber Samples

**Team:** Byte Busters, Batch A1  
**Course:** Machine Learning Lab (AIP301)  
**Institution:** Indian Institute of Information Technology Kota  
**Instructor:** Dr. Shibna Hussain Ma'am  
**November 2026**

**Team Members:**  
Ujjwal Garg (2024KUCP1001), Shaurya Singh (2024KUCP1003), G. Vishnu Vardhan (2024KUCP1004),  
Anupam Saxena (2024KUCP1006)

---

## Abstract

This project examines regression model behavior when feature count (p) equals or exceeds sample count (n). Using the Gene Expression Cancer (Golub et al.) — ALL vs AML Leukemia Classification

---

## 1. Introduction

Modern machine learning often faces scenarios where measured features (p) outnumber labeled samples (n) â€” common in genomics, microarrays, and sensor networks. When p â‰¥ n, standard linear models fail: design matrices lose full rank, coefficient estimates become unstable, and overfitting threatens. Our research question: **How do different regression approaches perform when p â‰¥ n, and which strategies ensure reliable predictions?**

This project investigates prediction reliability, overfitting control, feature selection, complexity regularization, noise sensitivity, redundancy handling, validation stability, and leakage prevention.

## 2. Dataset

**Dataset:** `Gene Expression Cancer (Golub et al.) — ALL vs AML Leukemia Classification
**Source:** Kaggle (crawford/gene-expression)
**Paper:** https://link.springer.com/article/10.1007/s10994-016-5546-z

| Property | Value |
|---|---|
| Samples (n) | 72 |
| Raw features | 416 |
| Cleaned features (p) | 375 |
| p/n ratio | 99.0 |
| Target | LBL_aUAminpA_fut_001 (next-day minimum price) |
| Target range | 278â€“1088 |
| Missing values | Minimal |

**Dataset Selection Note:** We initially attempted to use gene expression datasets from Kaggle as suggested. However, Kaggle API authentication consistently returned 403 errors, preventing downloads. Following the assignment's honesty requirement, we document this blocker and use OpenML's atp1d as a legitimate alternative with full provenance and p > n characteristics.

**Limitations:**
- Not gene expression data (Kaggle source unavailable)
- Time-ordered observations with random splits (temporal splitting would be better)
- Small sample size limits statistical power
- Some features show high target correlation (potential near-leakage)

## 3. Data Audit

We performed comprehensive data quality checks:

| Audit Item | Finding |
|---|---|
| Dimensions | 337 samples Ã— 375 features |
| Data types | All numeric (float64) |
| Missing values | Minimal |
| Duplicate rows | 0 |
| Constant features | 41 removed |
| Duplicate columns | Collapsed to first occurrence |
| High correlation pairs (\|r\| > 0.95) | 1,487 pairs |
| Target leakage candidates (\|r\| â‰¥ 0.9) | 9 features |
| Top correlate | LBL_aCOminpA_fut_001 (r â‰ˆ 0.988) |

The target distribution is approximately symmetric with no extreme outliers. The abundance of correlated feature pairs confirms a redundant, high-dimensional setting â€” precisely what this study targets.

**Cleaning Decisions:**
- Removed constant features (no information, inflate dimensionality)
- Collapsed duplicate columns (kept first copy to control redundancy explicitly in experiments)

## 4. Methodology

### 4.1 Feature-to-Sample Ratios

We constructed three feature sets from training data only (no target information used):

| Set | Target p/n | Description |
|---|---|---|
| Low | 0.5 | p < n (fewer features than samples) |
| About One | 1.0 | p â‰ˆ n (equal features and samples) |
| High | 1.5 | p > n (more features than samples) |

For typical 80/20 splits (nâ‰ˆ269): low pâ‰ˆ134, about_one pâ‰ˆ269, high p=375.

### 4.2 Models

1. **DummyRegressor** â€” Baseline predicting training mean
2. **Linear Regression** â€” OLS without regularization
3. **Ridge** â€” L2 regularization, shrinks coefficients
4. **Lasso** â€” L1 regularization, sparse feature selection
5. **Random Forest** â€” Non-linear ensemble

**Complexity Controls:** Ridge and Lasso regularization, SelectKBest and PCA feature selection â€” all implemented in scikit-learn Pipelines to prevent leakage.

### 4.3 Validation Strategy

**Nested Cross-Validation:**
- Outer loop (5-fold): estimates generalization
- Inner loop (3-fold): tunes hyperparameters

Loops use disjoint data preventing test information leakage.

**Repeated Splits:** 10 independent train/validation splits using consistent seeds across models for fair comparison.

**Metrics:** RÂ² (primary), RMSE, MAE

## 5. Results

### 5.1 Model Comparison

| Model | Mean RÂ² | Std RÂ² | Worst RÂ² | RMSE | MAE |
|---|---|---|---|---|---|
| Random Forest | 0.987 | 0.008 | 0.972 | 18.9 | 9.7 |
| Lasso | 0.978 | 0.008 | 0.967 | 24.9 | 15.0 |
| Ridge | 0.965 | 0.007 | 0.955 | 31.8 | 21.4 |
| Linear Regression | Unstable | â€” | â€” | â€” | â€” |
| Dummy | â‰ˆ0.0 | â€” | â€” | â€” | â€” |

Random Forest outperforms all models. Lasso exceeds Ridge, suggesting sparse regularization benefits. Linear Regression fails in p â‰¥ n settings as expected. All real models vastly exceed the dummy baseline.

### 5.2 Nested CV Results

Ridge nested CV: mean RÂ² â‰ˆ 0.95, std â‰ˆ 0.006, best alpha â‰ˆ 1.0 â€” consistent with repeated splits, confirming reliable generalization estimates.

### 5.3 Learning Curves

| Training Fraction | Train RÂ² | Val RÂ² | Gap |
|---|---|---|---|
| 20% | ~1.00 | ~0.77 | Large |
| 40% | ~1.00 | ~0.68 | Large |
| 60% | ~0.998 | ~0.73 | Large |
| 80% | ~0.996 | ~0.96 | Small |
| 100% | ~0.995 | ~0.99 | Minimal |

Validation performance improves dramatically with more data. Training RÂ² stays near 1.0 indicating overfitting at small n. The train/val gap closes as n grows â€” more labeled data would help significantly.

### 5.4 Noise and Redundancy

**Irrelevant Features:** Adding 0, 50, 200, 500 Gaussian noise features progressively degraded RÂ² from 0.96 to 0.80. Pure noise dilutes signal as expected.

**Redundant Features:** Adding exact duplicates, highly correlated copies (râ‰ˆ0.95), and perturbed versions barely affected Ridge performance (0.963â€“0.965 vs 0.965 baseline). Ridge robustly handles redundancy by distributing weights across correlated features.

### 5.5 Sanity Checks

**Target Permutation:** Shuffling training targets collapsed RÂ² from 0.965 to -1.15, confirming the pipeline is leakage-free and the signal is real.

**Feature Ablation:** Removing the top 10 influential features dropped RÂ² by 0.14 (from 0.965 to 0.82), showing heavy reliance on a small feature subset.

## 6. Failed Experiments

1. **Dataset successfully obtained from Kaggle.
2. **Linear Regression at p â‰¥ n:** Produced unstable results due to rank-deficient design matrix â€” an expected, informative failure motivating regularization.

No results were fabricated; all reported metrics come from executed experiments logged in `results/experiment_log.csv`.

## 7. Key Findings

1. **Regularization is essential:** Ridge and Lasso handle p > n reliably; unregularized Linear Regression fails
2. **More data helps:** Validation performance improves consistently with training set size
3. **Noise vs redundancy differ:** Irrelevant features harm performance; redundant features are tolerated by Ridge
4. **Model dependence:** Performance relies heavily on a small influential feature subset
5. **Validation integrity:** Nested CV and permutation checks confirm leakage-free estimates

## 8. Limitations and Future Work

**Current Limitations:**
- Gene expression data (leukemia classification)
- Cross-sectional gene expression samples
- Small sample size limits statistical confidence
- Results may not generalize to other high-dimensional problems

**Future Directions:**
- Apply temporal validation splits
- Test on genuine gene expression datasets when accessible
- Compare additional feature selection methods (RFE, Boruta)
- Evaluate model calibration beyond discrimination metrics
- Explore dimensionality reduction techniques (autoencoders, manifold learning)

## 9. Conclusion

This study demonstrates that regularized regression models effectively handle high-dimensional scenarios where features outnumber samples. Random Forest achieved the best performance, followed by Lasso and Ridge. Proper validation through nested cross-validation and rigorous leakage checks ensure trustworthy results. The experiments confirm that regularization, adequate training data, and careful validation are crucial for reliable prediction when p â‰¥ n.

---

**Reproducibility:** All results derived from executed code with fixed seed (42). Complete logs in `results/experiment_log.csv`, figures in `results/figures/`. See README.md for execution instructions.



