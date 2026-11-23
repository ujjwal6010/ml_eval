"""Plotting utilities for experiment results."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = Path("results") / "figures"


def _save(fig, name):
    """Save figure to results/figures."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_performance_vs_pn(pn_summary, name="performance_vs_pn.png"):
    """Plot model performance against p/n ratio."""
    labels = list(pn_summary.keys())
    means = [pn_summary[k].get("mean_r2", np.nan) for k in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, means, color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_ylabel("Mean R² (validation)")
    ax.set_xlabel("Feature set (p/n ratio)")
    ax.set_title("Performance vs p/n Ratio")
    if means:
        ax.set_ylim(bottom=min(0, min(means) - 0.1))
    for i, v in enumerate(means):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return _save(fig, name)


def plot_learning_curve(lc_results, name="learning_curve.png"):
    """Plot learning curve showing train/val performance vs sample size."""
    fig, ax = plt.subplots(figsize=(6, 4))
    fracs = [r["fraction"] for r in lc_results]
    train_r2 = [r["train_r2"] for r in lc_results]
    val_r2 = [r["val_r2"] for r in lc_results]
    ax.plot(fracs, train_r2, "o-", label="Training R²", color="#4c72b0")
    ax.plot(fracs, val_r2, "o-", label="Validation R²", color="#c44e52")
    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("R²")
    ax.set_title("Learning Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, name)


def plot_feature_stability(freq_dict, name="feature_stability.png", top_n=20):
    """Plot feature selection frequency across splits."""
    items = sorted(freq_dict.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    if not items:
        return None
    names, freqs = zip(*items)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(names)), freqs, color="#55a868")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("Selection Frequency")
    ax.set_xlabel("Feature")
    ax.set_title("Feature Selection Stability")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return _save(fig, name)


def plot_repeated_split(model_results, name="repeated_split_performance.png"):
    """Plot repeated split performance for model comparison."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for model_name, rows in model_results.items():
        if hasattr(rows, "iterrows"):
            scores = [row["r2"] for _, row in rows.iterrows()]
        else:
            scores = [row["r2"] for row in rows]
        ax.plot(range(len(scores)), scores, "o-", label=model_name, alpha=0.7)
    
    ax.set_xlabel("Split")
    ax.set_ylabel("R²")
    ax.set_title("Model Performance Across Repeated Splits")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, name)


def plot_noise_experiment(noise_results, name="noise_experiment.png"):
    """Plot effect of irrelevant features."""
    n_noise = [r["n_noise"] for r in noise_results]
    r2_vals = [r["r2"] for r in noise_results]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n_noise, r2_vals, "o-", color="#c44e52")
    ax.set_xlabel("Number of Noise Features")
    ax.set_ylabel("R²")
    ax.set_title("Effect of Irrelevant Features")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, name)
