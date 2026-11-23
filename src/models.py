"""Model definitions for regression experiments."""

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

RANDOM_SEED = 42


def dummy_regressor(strategy="mean"):
    """Baseline: predicts training target mean."""
    return DummyRegressor(strategy=strategy)


def linear_regression():
    """Ordinary least squares."""
    return LinearRegression()


def ridge_regression(alpha=1.0):
    """Ridge (L2) regression."""
    return Ridge(alpha=alpha, random_state=RANDOM_SEED)


def lasso_regression(alpha=0.001, max_iter=10000):
    """Lasso (L1) regression."""
    return Lasso(alpha=alpha, max_iter=max_iter, random_state=RANDOM_SEED)


def random_forest_regressor(n_estimators=200, max_depth=None, max_features=1.0):
    """Random forest regressor."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )


ALL_MODELS = {
    "Dummy (mean)": dummy_regressor,
    "Linear Regression": linear_regression,
    "Ridge": ridge_regression,
    "Lasso": lasso_regression,
    "Random Forest": random_forest_regressor,
}
