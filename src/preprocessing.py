"""Data preprocessing and splitting utilities."""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

RANDOM_SEED = 42


def make_preprocessing_pipeline() -> Pipeline:
    """Create pipeline with standard scaling.
    
    StandardScaler fits only on training data when used in pipeline.
    """
    return Pipeline([("scaler", StandardScaler())])


def train_val_split(X, y, test_size=0.2, seed=RANDOM_SEED):
    """Single train/validation split."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=True)


def repeated_splits(X, y, n_splits=10, test_size=0.2, seed=RANDOM_SEED):
    """Generate multiple train/val splits for stability testing.
    
    Uses different seeds for each split to ensure independence.
    """
    for i in range(n_splits):
        s = seed + i * 1000
        yield train_test_split(X, y, test_size=test_size, random_state=s, shuffle=True)


def kfold_splitter(n_splits=5, seed=RANDOM_SEED):
    """KFold splitter for cross-validation."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def fit_scaler_on_train(X_train):
    """Fit scaler on training data only."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_with_scaler(scaler, X):
    """Apply fitted scaler to data."""
    return scaler.transform(X)
