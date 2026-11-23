"""Experiment logging and execution utilities."""

import csv
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

from preprocessing import RANDOM_SEED, make_preprocessing_pipeline
from models import ALL_MODELS
from evaluation import regression_metrics

RESULTS_DIR = Path("results")
LOG_CSV = RESULTS_DIR / "experiment_log.csv"
SUMMARY_JSON = RESULTS_DIR / "summary.json"

INNER_GRID = {
    "Ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Lasso": {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]},
    "Random Forest": {"n_estimators": [100, 300], "max_depth": [None, 5, 10]},
    "Linear Regression": {},
    "Dummy (mean)": {},
}


class ExperimentLogger:
    """Log experiment results to CSV and JSON."""
    
    def __init__(self, log_path=LOG_CSV, summary_path=SUMMARY_JSON):
        self.log_path = Path(log_path)
        self.summary_path = Path(summary_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self._write_header_if_needed()
    
    def _write_header_if_needed(self):
        if not self.log_path.exists():
            fields = [
                "experiment_id", "timestamp", "dataset", "experiment_type",
                "model", "parameters", "feature_set_name",
                "original_feature_count", "added_feature_count",
                "retained_feature_count", "training_sample_count",
                "p_over_n_ratio", "split_seed", "r2", "rmse", "mae",
                "training_runtime_sec", "inference_runtime_sec", "notes",
            ]
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
    
    def log(
        self,
        experiment_id,
        experiment_type,
        model="",
        parameters="",
        feature_set_name="",
        original_feature_count=0,
        added_feature_count=0,
        retained_feature_count=0,
        training_sample_count=0,
        p_over_n_ratio=0.0,
        split_seed=RANDOM_SEED,
        metrics=None,
        training_runtime_sec=0.0,
        inference_runtime_sec=0.0,
        notes="",
        dataset="gene_expression",
    ):
        row = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": dataset,
            "experiment_type": experiment_type,
            "model": model,
            "parameters": parameters,
            "feature_set_name": feature_set_name,
            "original_feature_count": original_feature_count,
            "added_feature_count": added_feature_count,
            "retained_feature_count": retained_feature_count,
            "training_sample_count": training_sample_count,
            "p_over_n_ratio": float(p_over_n_ratio),
            "split_seed": split_seed,
            "r2": (metrics or {}).get("r2", np.nan),
            "rmse": (metrics or {}).get("rmse", np.nan),
            "mae": (metrics or {}).get("mae", np.nan),
            "training_runtime_sec": float(training_runtime_sec),
            "inference_runtime_sec": float(inference_runtime_sec),
            "notes": notes,
        }
        
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
        
        self.rows.append(row)
    
    def save_summary(self, summary_dict):
        """Save summary dict to JSON."""
        with open(self.summary_path, "w") as f:
            json.dump(summary_dict, f, indent=2)


def nested_cv(
    X, y, model_func, param_grid=None, inner_cv=3, outer_cv=5, seed=RANDOM_SEED
):
    """Nested cross-validation with inner tuning and outer evaluation."""
    if param_grid is None:
        param_grid = {}
    
    outer_kfold = KFold(n_splits=outer_cv, shuffle=True, random_state=seed)
    inner_kfold = KFold(n_splits=inner_cv, shuffle=True, random_state=seed + 1)
    
    outer_scores = []
    best_params_list = []
    
    for outer_idx, (train_idx, test_idx) in enumerate(outer_kfold.split(X)):
        X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_test = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
        y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        y_test = y.iloc[test_idx] if hasattr(y, "iloc") else y[test_idx]
        
        # inner CV: find best params
        best_score = -np.inf
        best_params = None
        
        if not param_grid:
            # no tuning needed
            pipeline = Pipeline([
                ("scaler", make_preprocessing_pipeline().steps[0][1]),
                ("model", model_func())
            ])
            best_params = {}
        else:
            # grid search
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            from itertools import product
            for param_combo in product(*param_values):
                params = dict(zip(param_names, param_combo))
                inner_scores = []
                
                for inner_train_idx, inner_val_idx in inner_kfold.split(X_train):
                    X_inner_train = X_train.iloc[inner_train_idx] if hasattr(X_train, "iloc") else X_train[inner_train_idx]
                    X_inner_val = X_train.iloc[inner_val_idx] if hasattr(X_train, "iloc") else X_train[inner_val_idx]
                    y_inner_train = y_train.iloc[inner_train_idx] if hasattr(y_train, "iloc") else y_train[inner_train_idx]
                    y_inner_val = y_train.iloc[inner_val_idx] if hasattr(y_train, "iloc") else y_train[inner_val_idx]
                    
                    pipeline = Pipeline([
                        ("scaler", make_preprocessing_pipeline().steps[0][1]),
                        ("model", model_func(**params))
                    ])
                    pipeline.fit(X_inner_train, y_inner_train)
                    pred = pipeline.predict(X_inner_val)
                    score = r2_score(y_inner_val, pred)
                    inner_scores.append(score)
                
                mean_score = np.mean(inner_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
        
        # train on full outer train with best params
        pipeline = Pipeline([
            ("scaler", make_preprocessing_pipeline().steps[0][1]),
            ("model", model_func(**best_params) if best_params else model_func())
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_score = r2_score(y_test, y_pred)
        
        outer_scores.append(test_score)
        best_params_list.append(best_params)
    
    return {
        "outer_scores": outer_scores,
        "mean_r2": float(np.mean(outer_scores)),
        "std_r2": float(np.std(outer_scores)),
        "best_params_per_fold": best_params_list,
    }

