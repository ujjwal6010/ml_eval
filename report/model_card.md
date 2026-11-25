# Model Card

**Project:** Prediction When Features Outnumber Samples  
**Team:** Byte Busters, Batch A1  
**Course:** Machine Learning Lab (AIP301), IIIT Kota  
**Date:** November 2026  
**Seed:** 42 (fixed throughout)

---

## Model Overview

**Models Evaluated:**
- Dummy Regressor (baseline)
- Linear classification (OLS)
- Ridge classification (L2 regularization)
- Lasso classification (L1 regularization)
- Random Forest Regressor

**Best Performer:** Random Forest (R² ≈ 0.987)

## Intended Use

Educational demonstration of classification in high-dimensional settings (p > n). Illustrates:
- Overfitting in small-sample scenarios
- Regularization benefits
- Feature selection strategies
- Nested cross-validation
- Data leakage prevention
- Noise and redundancy effects

**Not intended for:** Production deployment, real-world pricing decisions, or high-stakes applications without further validation.

## Out-of-Scope Uses

- Production cancer classification (ALL vs AML leukemia)
- Safety-critical, medical, financial, or legal decisions
- Generalization to different data distributions
- Causal inference (this is a predictive correlation study)

## Dataset

**Working Dataset:** `Gene Expression Cancer (ALL vs AML)
**Source:** Kaggle (crawford/gene-expression)
**Paper:** https://link.springer.com/article/10.1007/s10994-016-5546-z

**Target:** LBL_aUAminpA_fut_001 (next-day minimum price)  
**Type:** Numeric classification  
**Range:** 278–1088\
**Mean:** ≈470, Std: ≈174\
**Unique values:** 61 out of 337 samples

**Source:** Kaggle dataset (crawford/gene-expression), based on the landmark Golub et al. study (Science, 1999).

## Dataset Limitations

- n=337 samples, p=375 features → p/n = 1.11
- Not gene expression/microarray data (preferred source unavailable)
- Time-ordered data with random splits (temporal splits preferable)
- Features highly correlated with target (potential near-leakage)
- Small sample limits stability estimates

## Preprocessing

- Removed 41 constant features
- Collapsed exact duplicate columns
- StandardScaler (zero mean, unit variance) fitted only inside training folds via Pipelines
- Feature selection (SelectKBest/PCA) also fitted only on training data

## Validation

**Nested Cross-Validation:**
- Outer 5-fold (generalization estimate)
- Inner 3-fold (hyperparameter tuning)
- Disjoint folds prevent leakage

**Repeated Evaluation:** 10 independent splits with consistent seeds across models

**Metrics:** R² (primary), RMSE, MAE

## Performance (10 Repeated Splits)

### Ridge classification
| Metric | Mean | Std | Worst |
|---|---|---|---|
| R² | 0.965 | 0.007 | 0.955 |
| RMSE | 31.8 | — | — |
| MAE | 21.4 | — | — |

### Lasso classification
| Metric | Mean | Std | Worst |
|---|---|---|---|
| R² | 0.978 | 0.008 | 0.967 |
| RMSE | 24.9 | — | — |
| MAE | 15.0 | — | — |

### Random Forest
| Metric | Mean | Std | Worst |
|---|---|---|---|
| R² | 0.987 | 0.008 | 0.972 |
| RMSE | 18.9 | — | — |
| MAE | 9.7 | — | — |

## Known Failure Cases

- **Linear classification:** Unstable when p ≥ n (rank-deficient design matrix)
- **Small training sets (≤20%):** Severe overfitting (train R² ≈ 1, val R² << 1)
- **Irrelevant features:** Performance degrades with increasing noise feature count
- **Feature ablation:** Removing top 10 features drops R² by ~0.14
- **Target permutation:** Performance collapses (expected, confirms leakage-free pipeline)

## Risks

- Overfitting when n << p
- Optimistic bias if hyperparameters tuned on test set (avoided via nested CV)
- Data leakage from preprocessing outside training folds (avoided via Pipelines)
- Near-leakage from highly correlated features

## Limitations

- Results specific to Golub et al. gene expression dataset
- Small sample size limits confidence
- Random splits on time-ordered data
- Not validated for production use
- Performance may not generalize to other domains

## Ethical Considerations

- Educational project only — not for real-world decision-making
- No sensitive personal data involved
- No fairness concerns (cancer classification task)
- Transparent reporting of limitations and failed experiments

## Maintenance

**Contact:** Byte Busters Team (IDs: 1001, 1003, 1004, 1006)\
**Course:** AIP301, IIIT Kota  
**Reproducibility:** Seed=42, logs in `results/`, code in `src/`



