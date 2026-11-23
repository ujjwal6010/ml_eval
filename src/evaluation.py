"""Evaluation metrics for regression tasks."""

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calculate regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"r2": float(r2), "rmse": rmse, "mae": mae}


def metrics_table(rows):
    """Convert list of dicts to DataFrame."""
    return pd.DataFrame(rows)
