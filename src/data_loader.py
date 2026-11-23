"""Dataset loading and preprocessing for gene expression data."""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "kaggle"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

KAGGLE_DATASET = "crawford/gene-expression"


def load_dataset(
    force_download: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, list[str], dict]:
    """Load and prepare the gene expression dataset.
    
    Returns:
        X: input features (gene expression values)
        y: target variable (cancer type: ALL=0, AML=1)
        feature_names: list of gene names
        metadata: dict with dataset info
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    train_path = RAW_DIR / "data_set_ALL_AML_train.csv"
    test_path = RAW_DIR / "data_set_ALL_AML_independent.csv"
    labels_path = RAW_DIR / "actual.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_DIR}")
    
    print(f"Loading: {RAW_DIR}")
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    labels = pd.read_csv(labels_path)
    
    # Remove "call" columns
    def get_value_cols(df):
        return [col for col in df.columns 
                if col not in ['Gene Description', 'Gene Accession Number'] 
                and 'call' not in col.lower()]
    
    train_cols = get_value_cols(train_df)
    test_cols = get_value_cols(test_df)
    
    # Extract and transpose
    train_expr = train_df[train_cols].T
    test_expr = test_df[test_cols].T
    
    # Combine
    X = pd.concat([train_expr, test_expr], axis=0, ignore_index=True)
    X.columns = [f"gene_{i}" for i in range(X.shape[1])]
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Labels
    label_map = {'ALL': 0, 'AML': 1}
    y = pd.Series([label_map[label] for label in labels['cancer']], name='cancer_type')
    
    if X.shape[0] != len(y):
        raise ValueError(f"Mismatch: X={X.shape[0]} samples, y={len(y)} labels")
    
    # Remove constant and duplicate features
    nunique = X.nunique(dropna=True)
    constant_cols = [col for col in X.columns if nunique[col] <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
    
    X = X.T.drop_duplicates().T
    
    feature_names = list(X.columns)
    metadata = {
        "name": "Gene Expression Cancer (ALL vs AML)",
        "source": "Kaggle",
        "dataset": KAGGLE_DATASET,
        "url": f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}",
        "reference": "Golub et al., Science 1999"
    }
    
    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features (p/n = {X.shape[1]/X.shape[0]:.1f})")
    
    return X, y, feature_names, metadata


def audit_summary(X: pd.DataFrame, y: pd.Series) -> dict:
    """Dataset summary."""
    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "p_over_n": float(X.shape[1] / X.shape[0]),
        "target_name": y.name,
        "target_classes": y.nunique(),
        "class_distribution": y.value_counts().to_dict(),
        "missing_X": int(X.isna().sum().sum()),
        "missing_y": int(y.isna().sum()),
        "seed": RANDOM_SEED,
    }


if __name__ == "__main__":
    X, y, names, meta = load_dataset()
    print("\nDataset summary:")
    summary = audit_summary(X, y)
    for key, val in summary.items():
        print(f"  {key}: {val}")
