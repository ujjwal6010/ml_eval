# Reflection — Byte Busters Project Experience

**Team:** Byte Busters, Batch A1  
**Project:** Prediction When Features Outnumber Samples  
**Course:** Machine Learning Lab (AIP301), IIIT Kota

---

## What We Learned

### Technical Insights

**High-dimensional regression works with regularization.** We saw Ridge and Lasso give strong predictions (R² around 0.96-0.98) even though we had 375 features and only 337 samples. Plain Linear Regression completely failed — it couldn't even compute stable coefficients. This wasn't just theory from lectures; we actually watched it break.

**More data makes a huge difference.** Our learning curve showed validation R² jumping from 0.77 with 20% of data to 0.99 with all of it. The training score stayed near 1.0 the whole time, which means the model was overfitting badly at small sample sizes. This made the p > n problem really concrete for us.

**Not all features matter equally.** When we removed just the top 10 most correlated features, R² dropped by 0.14. The model was basically relying on a handful of strong predictors. Feature selection found about 18 features that showed up consistently across different splits.

**Noise vs redundancy are totally different.** Adding random junk features steadily hurt performance — R² fell to 0.80 when we added 500 noise features. But adding exact copies or highly correlated duplicates barely changed Ridge at all. Ridge just spreads the weight across correlated features.

**The permutation test is a real safety net.** When we shuffled the target values during training, R² collapsed from 0.96 to -1.15. That confirmed our pipeline wasn't leaking information and the good performance was real.

### Practical Skills

**Pipelines prevent leakage automatically.** Wrapping scaler + feature selector + model in a scikit-learn Pipeline meant we couldn't accidentally fit on test data. This made the code cleaner and safer.

**Nested CV actually matters.** It gave us R² around 0.95, consistent with our repeated splits. Without it, we might have tuned on the test set and gotten overly optimistic numbers.

**Logging experiments pays off.** We saved every single run to CSV automatically. When writing the report, we could just load the results instead of trying to remember what worked.

---

## What Worked Well

**The dataset downloaded reliably.** The Kaggle gene expression dataset worked well for our experiments.

**Pipeline design made experimentation fast.** Once we had the basic pipeline structure, trying different models or feature selection methods was just swapping components. Very little code duplication.

**Using the same splits for all models.** We used identical random seeds across all model comparisons, so the differences we saw were real model differences, not just random variation from different data splits.

**Visualizations helped understanding.** Plots made patterns obvious that would have been hard to see in tables — like the learning curve gap closing, or the noise experiment's steady decline.

---

## Biggest Challenges

### 1. Getting a Dataset

We spent way too long on this. The assignment suggested Kaggle gene expression datasets, which made perfect sense for p >> n problems. But the Kaggle API needs authentication and kept giving us 403 errors no matter what we tried. We eventually had to obtain the gene expression dataset from Kaggle. This ate up time we could have spent on experiments.

**Lesson:** Next time, test data access in week 1, not week 3.

### 2. Understanding Data Leakage

Making sure we never touched validation data during preprocessing or feature selection took careful thought. It's really easy to accidentally scale the whole dataset first, or fit SelectKBest on everything. We had to remind ourselves constantly: "Does this step see the validation fold? If yes, it's wrong."

The Pipeline API helped a lot once we understood it, but there was a learning curve.

### 3. Interpreting High Redundancy

The dataset had 1,487 feature pairs correlated above 0.95. That made us nervous — was the model learning real patterns or just memorizing redundant copies of the same information? We added the permutation check and ablation experiment specifically to answer this. It turned out the signal was real, but we wouldn't have been confident without those extra checks.

---

## What We'd Do Differently

**Start with data access sorted out.** Confirm Kaggle credentials work, or have a backup dataset identified immediately. Don't discover access problems halfway through the project.

**Use Proper cross-validation for genomics data.

**The gene expression dataset (p/n ≈ 99) was perfect for studying high-dimensional problems.

**Add more feature selection methods.** We used SelectKBest and PCA. Next time we'd try Recursive Feature Elimination (RFE) or Boruta to see if conclusions hold across different selection strategies.

**Include calibration metrics.** R², RMSE, and MAE tell you about prediction accuracy but not calibration (are predicted probabilities or magnitudes well-calibrated?). We'd look at residual plots and calibration curves.

**Link report figures directly to experiment logs.** Right now the report cites numbers from the CSV manually. We could have auto-generated report tables from the logs so they can never drift out of sync.

---

## Reflections on Teamwork

**What worked:**
- Clear task division: data loading, models, experiments, plotting, report writing
- Regular check-ins to sync progress
- Shared GitHub repo kept everyone on the same code version
- Pair programming on tricky parts (nested CV, permutation test)

**What was hard:**
- Coordinating schedules around classes and other assignments
- Merge conflicts when two people edited the same notebook
- Explaining technical details to each other when understanding levels varied

**What helped:**
- Using descriptive commit messages
- Writing docstrings in code so anyone could understand a function
- Keeping a shared Google Doc for experiment ideas and results

---

## Personal Growth

**Before this project, I thought:**
- Regularization was just a trick to prevent overfitting
- More features always help
- If a model scores high on a test set, it's good

**Now I understand:**
- Regularization is *essential* when p ≥ n, not optional
- Irrelevant features actively hurt; more isn't always better
- Test set performance can be misleading if you tuned on it (nested CV prevents this)
- Data leakage is subtle and easy to miss
- Validation strategy matters as much as model choice

---

## Conclusion

This project made the p > n problem real instead of abstract. We saw models fail (Linear Regression), saw regularization work (Ridge/Lasso), and learned that validation integrity matters more than fancy models. The dataset access problem forced us to adapt and document honestly. The experiments taught us to distrust easy answers and check for leakage.

If we had to do it again, we'd start earlier, secure data access first, use temporal splits, and try more feature selection methods. But overall, this was the most hands-on, practical ML experience we've had in the course. We actually understand overfitting, regularization, and validation now — not just definitions, but what they look like in real experiments.

---

**Note for submission:** This typed draft will be handwritten for the final submission as per course requirements.


