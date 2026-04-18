import time
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from itertools import product as iter_product

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None

from config import RANDOM_SEED, MLFLOW_EXPERIMENTS, get_logger
from evaluation.metrics import regression_metrics
from tracking.mlflow_tracking import log_sklearn_model

logger = get_logger(__name__)


def _best_params(model_cls, param_grid, X_train, y_train, X_val, y_val, fixed_params=None):
    """Simple grid search on validation set (no cross-validation — time-series)."""
    best_score = -np.inf
    best_model = None
    best_p = None

    keys = list(param_grid.keys())
    for values in iter_product(*param_grid.values()):
        params = dict(zip(keys, values))
        if fixed_params:
            params.update(fixed_params)
        model = model_cls(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        m = regression_metrics(y_val, preds)
        if m["r2"] > best_score:
            best_score = m["r2"]
            best_model = model
            best_p = params

    return best_model, best_p


def train_all_regressors(X_train, y_train, X_val, y_val):
    """Train all 7 regression models, evaluate, and log to MLflow.

    Returns:
        dict of {model_name: (model, val_metrics, params)}
    """
    experiment = MLFLOW_EXPERIMENTS["regression"]
    results = {}

    # 1. Linear Regression (baseline)
    logger.info("Training LinearRegression")
    t0 = time.time()
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    preds = lr.predict(X_val)
    m = regression_metrics(y_val, preds)
    m["training_time"] = time.time() - t0
    params = {"model": "LinearRegression"}
    log_sklearn_model(lr, "linear_regression", params, m, experiment)
    results["linear_regression"] = (lr, m, params)

    # 2. Ridge
    logger.info("Training Ridge")
    t0 = time.time()
    ridge, ridge_p = _best_params(
        Ridge, {"alpha": [0.01, 0.1, 1.0, 10.0]},
        X_train, y_train, X_val, y_val, {"random_state": RANDOM_SEED}
    )
    preds = ridge.predict(X_val)
    m = regression_metrics(y_val, preds)
    m["training_time"] = time.time() - t0
    ridge_p["model"] = "Ridge"
    log_sklearn_model(ridge, "ridge_regression", ridge_p, m, experiment)
    results["ridge"] = (ridge, m, ridge_p)

    # 3. Lasso
    logger.info("Training Lasso")
    t0 = time.time()
    lasso, lasso_p = _best_params(
        Lasso, {"alpha": [0.001, 0.01, 0.1, 1.0]},
        X_train, y_train, X_val, y_val, {"random_state": RANDOM_SEED}
    )
    preds = lasso.predict(X_val)
    m = regression_metrics(y_val, preds)
    m["training_time"] = time.time() - t0
    lasso_p["model"] = "Lasso"
    log_sklearn_model(lasso, "lasso_regression", lasso_p, m, experiment)
    results["lasso"] = (lasso, m, lasso_p)

    # 4. Random Forest
    logger.info("Training RandomForestRegressor")
    t0 = time.time()
    rf, rf_p = _best_params(
        RandomForestRegressor,
        {"n_estimators": [100, 200], "max_depth": [10, 20]},
        X_train, y_train, X_val, y_val, {"random_state": RANDOM_SEED, "n_jobs": -1}
    )
    preds = rf.predict(X_val)
    m = regression_metrics(y_val, preds)
    m["training_time"] = time.time() - t0
    rf_p["model"] = "RandomForest"
    log_sklearn_model(rf, "random_forest_regressor", rf_p, m, experiment)
    results["random_forest"] = (rf, m, rf_p)

    # 5. XGBoost
    if XGBRegressor is not None:
        logger.info("Training XGBRegressor")
        t0 = time.time()
        xgb, xgb_p = _best_params(
            XGBRegressor,
            {"learning_rate": [0.01, 0.1], "max_depth": [3, 6], "n_estimators": [100, 200]},
            X_train, y_train, X_val, y_val,
            {"random_state": RANDOM_SEED, "verbosity": 0, "subsample": 0.8}
        )
        preds = xgb.predict(X_val)
        m = regression_metrics(y_val, preds)
        m["training_time"] = time.time() - t0
        xgb_p["model"] = "XGBoost"
        log_sklearn_model(xgb, "xgboost_regressor", xgb_p, m, experiment)
        results["xgboost"] = (xgb, m, xgb_p)
    else:
        logger.warning("Skipping XGBoost regressor because xgboost is not installed.")

    # 6. LightGBM
    if LGBMRegressor is not None:
        logger.info("Training LGBMRegressor")
        t0 = time.time()
        lgbm, lgbm_p = _best_params(
            LGBMRegressor,
            {"learning_rate": [0.01, 0.1], "num_leaves": [31, 63], "n_estimators": [100, 200]},
            X_train, y_train, X_val, y_val, {"random_state": RANDOM_SEED, "verbosity": -1}
        )
        preds = lgbm.predict(X_val)
        m = regression_metrics(y_val, preds)
        m["training_time"] = time.time() - t0
        lgbm_p["model"] = "LightGBM"
        log_sklearn_model(lgbm, "lightgbm_regressor", lgbm_p, m, experiment)
        results["lightgbm"] = (lgbm, m, lgbm_p)
    else:
        logger.warning("Skipping LightGBM regressor because lightgbm is not installed.")

    # 7. SVR
    logger.info("Training SVR")
    t0 = time.time()
    svr, svr_p = _best_params(
        SVR,
        {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]},
        X_train, y_train, X_val, y_val, {"kernel": "rbf"}
    )
    preds = svr.predict(X_val)
    m = regression_metrics(y_val, preds)
    m["training_time"] = time.time() - t0
    svr_p["model"] = "SVR"
    log_sklearn_model(svr, "svr_regressor", svr_p, m, experiment)
    results["svr"] = (svr, m, svr_p)

    # Summary
    logger.info("=== Regression Results ===")
    for name, (_, m, _) in results.items():
        logger.info(f"  {name}: R²={m['r2']:.4f}, MAE={m['mae']:.6f}, Dir.Acc={m['directional_accuracy']:.4f}")

    return results


def get_best_regressor(results: dict):
    """Return the model name and object with the highest validation R²."""
    best_name = max(results, key=lambda k: results[k][1]["r2"])
    best_model, best_metrics, _ = results[best_name]
    logger.info(f"Best regressor: {best_name} (R²={best_metrics['r2']:.4f})")
    return best_name, best_model
