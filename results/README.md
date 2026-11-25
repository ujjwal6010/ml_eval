# Results Directory

This directory will be populated when you run the experiments.

## Generated Files

After running `notebooks/experiments.ipynb`, this folder will contain:

### Logs
- **experiment_log.csv** - Detailed log of all experiments with parameters and metrics
- **final_results.csv** - Summary table of final model comparison results
- **summary.json** - Machine-readable summary with metadata

### Figures (in `figures/` subdirectory)
- Performance vs p/n ratio plots
- Learning curves
- Feature stability analysis
- Error distribution plots
- Correlation heatmaps
- And more...

## How to Generate

```bash
cd ByteBusters_Submission
jupyter notebook notebooks/experiments.ipynb
```

Then: **Kernel → Restart & Run All**

All results will be automatically generated and saved here.
