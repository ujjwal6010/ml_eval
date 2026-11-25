"""Data quality checks and exploratory analysis."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIG_DIR = Path("results") / "figures"


def data_dictionary(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Create data dictionary with column stats."""
    rows = []
    for col in X.columns:
        s = X[col]
        rows.append({
            "column": col,
            "type": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "n_unique": int(s.nunique(dropna=True)),
            "min": float(s.min()) if pd.api.types.is_numeric_dtype(s) else None,
            "max": float(s.max()) if pd.api.types.is_numeric_dtype(s) else None,
            "mean": float(s.mean()) if pd.api.types.is_numeric_dtype(s) else None,
            "std": float(s.std()) if pd.api.types.is_numeric_dtype(s) else None,
            "role": "feature",
        })
    
    rows.append({
        "column": y.name,
        "type": str(y.dtype),
        "n_missing": int(y.isna().sum()),
        "n_unique": int(y.nunique(dropna=True)),
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "std": float(y.std()),
        "role": "target",
    })
    return pd.DataFrame(rows)


def basic_audit(
    X: pd.DataFrame,
    y: pd.Series,
    include_pairwise_corr: bool = False,
) -> dict:
    """Audit data quality, optionally counting highly correlated feature pairs."""
    high_corr_pairs = None
    if include_pairwise_corr:
        corr = X.corr(numeric_only=True).to_numpy()
        high_corr_pairs = sum(
            np.count_nonzero(np.abs(corr[i, i + 1:]) > 0.95)
            for i in range(corr.shape[0] - 1)
        )
    
    target_corr = X.corrwith(y).sort_values(key=np.abs, ascending=False)
    
    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "missing_X": int(X.isna().sum().sum()),
        "missing_y": int(y.isna().sum()),
        "duplicate_rows": int(X.duplicated().sum()),
        "constant_features": int((X.nunique(dropna=True) <= 1).sum()),
        "high_corr_pairs": high_corr_pairs,
        "top_target_correlations": target_corr.head(10).to_dict(),
    }


def find_leakage_candidates(X: pd.DataFrame, y: pd.Series, threshold=0.9):
    """Find features highly correlated with target."""
    corr = X.corrwith(y).abs().sort_values(ascending=False)
    return corr[corr >= threshold].to_dict()


def plot_target_distribution(y: pd.Series, fig_dir: Path = FIG_DIR) -> Path:
    """Plot target variable distribution."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(y, bins=30, color="#4c72b0", edgecolor="black", alpha=0.7)
    ax.set_xlabel(y.name)
    ax.set_ylabel("Frequency")
    ax.set_title("Target Distribution")
    path = fig_dir / "target_distribution.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_distributions(X: pd.DataFrame, fig_dir: Path = FIG_DIR, n_features=10):
    """Plot distributions of first n features."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    n = min(n_features, len(X.columns))
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i, col in enumerate(X.columns[:n]):
        axes[i].hist(X[col].dropna(), bins=20, color="#55a868", edgecolor="black", alpha=0.7)
        axes[i].set_title(col[:20], fontsize=8)
        axes[i].tick_params(labelsize=6)
    
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    
    fig.tight_layout()
    path = fig_dir / "feature_distributions.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
