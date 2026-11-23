"""Feature manipulation for p/n ratio experiments."""

import numpy as np
import pandas as pd


def feature_sets_by_ratio(
    X_train: pd.DataFrame,
    ratios=("low", "about_one", "high"),
    low_p_over_n=0.5,
    about_one_p_over_n=1.0,
    high_multiplier=1.5,
    seed=42,
) -> dict:
    """Create feature subsets with different p/n ratios.
    
    p = number of features, n = number of training samples.
    Features selected by position, no target information used.
    
    Args:
        X_train: training features
        ratios: which sets to build ("low", "about_one", "high")
        low_p_over_n: target ratio for low set (p/n < 1)
        about_one_p_over_n: target ratio for about_one set (p/n ≈ 1)
        high_multiplier: multiplier for high set (p = multiplier × n)
        seed: random seed for reproducibility
    
    Returns:
        dict of feature sets with metadata
    """
    rng = np.random.default_rng(seed)
    n = len(X_train)
    all_features = list(X_train.columns)
    total_p = len(all_features)
    
    def pick(k):
        """Pick k features evenly spaced by index."""
        k = int(max(1, min(k, total_p)))
        if k >= total_p:
            return list(all_features)
        idx = np.linspace(0, total_p - 1, k).round().astype(int)
        idx = np.unique(idx)
        return [all_features[i] for i in idx]
    
    sets = {}
    specs = {
        "low": (low_p_over_n, "p/n < 1"),
        "about_one": (about_one_p_over_n, "p/n ≈ 1"),
        "high": (high_multiplier, "p/n > 1"),
    }
    
    for name in ratios:
        if name not in specs:
            continue
        target_ratio, desc = specs[name]
        if name == "high":
            p = int(high_multiplier * n)
        else:
            p = int(target_ratio * n)
        p = max(1, min(p, total_p))
        feats = pick(p)
        actual_ratio = p / n
        sets[name] = {
            "X": X_train[feats],
            "feature_names": feats,
            "p": p,
            "n": n,
            "p_over_n": float(actual_ratio),
            "description": desc,
        }
    return sets


def add_irrelevant_features(X, n_noise, seed=42, prefix="noise_"):
    """Add random Gaussian noise features."""
    rng = np.random.default_rng(seed)
    noise = pd.DataFrame(
        rng.standard_normal((X.shape[0], n_noise)),
        index=X.index,
        columns=[f"{prefix}{i}" for i in range(n_noise)],
    )
    return pd.concat([X.reset_index(drop=True), noise.reset_index(drop=True)], axis=1)


def add_redundant_features(
    X,
    n_dup=0,
    n_corr=0,
    n_perturb=0,
    seed=42,
    corr_strength=0.95,
    perturb_noise=0.05,
):
    """Add redundant versions of existing features.
    
    - n_dup: exact copies
    - n_corr: highly correlated versions
    - n_perturb: noisy copies
    """
    rng = np.random.default_rng(seed)
    X = X.copy()
    feature_names = list(X.columns)
    
    for i in range(n_dup):
        src = feature_names[i % len(feature_names)]
        X[f"dup_{i}"] = X[src].values
    
    for i in range(n_corr):
        src = feature_names[i % len(feature_names)]
        base = X[src].values.astype(float)
        noise = rng.standard_normal(len(base)) * (base.std() * 0.05 + 1e-8)
        X[f"corr_{i}"] = corr_strength * base + noise
    
    for i in range(n_perturb):
        src = feature_names[i % len(feature_names)]
        base = X[src].values.astype(float)
        noise = rng.standard_normal(len(base)) * (
            perturb_noise * (base.std() if base.std() > 0 else 1.0)
        )
        X[f"perturb_{i}"] = base + noise
    
    return X
