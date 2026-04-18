import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from itertools import product as iter_product

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None

from config import RANDOM_SEED, MLFLOW_EXPERIMENTS, get_logger
from evaluation.metrics import classification_metrics
from tracking.mlflow_tracking import log_sklearn_model

logger = get_logger(__name__)


def _best_params(model_cls, param_grid, X_train, y_train, X_val, y_val, fixed_params=None):
    """Simple grid search on validation set."""
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
        m = classification_metrics(y_val, preds)
        if m["f1_macro"] > best_score:
            best_score = m["f1_macro"]
            best_model = model
            best_p = params

    return best_model, best_p


def train_all_classifiers(X_train, y_train, X_val, y_val):
    """Train all 5 classification models, evaluate, and log to MLflow.

    Returns:
        dict of {model_name: (model, val_metrics, params)}
    """
    experiment = MLFLOW_EXPERIMENTS["classification"]
    results = {}

    # 1. Logistic Regression
    logger.info("Training LogisticRegression")
    t0 = time.time()
    lr, lr_p = _best_params(
        LogisticRegression,
        {"C": [0.01, 0.1, 1.0, 10.0]},
        X_train, y_train, X_val, y_val,
        {"solver": "lbfgs", "max_iter": 1000, "random_state": RANDOM_SEED}
    )
    preds = lr.predict(X_val)
    probs = lr.predict_proba(X_val)
    m = classification_metrics(y_val, preds, probs)
    m_log = {k: v for k, v in m.items() if k != "confusion_matrix"}
    m_log["training_time"] = time.time() - t0
    lr_p["model"] = "LogisticRegression"
    log_sklearn_model(lr, "logistic_regression", lr_p, m_log, experiment)
    results["logistic_regression"] = (lr, m, lr_p)

    # 2. Random Forest
    logger.info("Training RandomForestClassifier")
    t0 = time.time()
    rf, rf_p = _best_params(
        RandomForestClassifier,
        {"n_estimators": [100, 200], "max_depth": [10, 20]},
        X_train, y_train, X_val, y_val,
        {"random_state": RANDOM_SEED, "n_jobs": -1}
    )
    preds = rf.predict(X_val)
    probs = rf.predict_proba(X_val)
    m = classification_metrics(y_val, preds, probs)
    m_log = {k: v for k, v in m.items() if k != "confusion_matrix"}
    m_log["training_time"] = time.time() - t0
    rf_p["model"] = "RandomForest"
    log_sklearn_model(rf, "random_forest_classifier", rf_p, m_log, experiment)
    results["random_forest"] = (rf, m, rf_p)

    # 3. XGBoost
    if XGBClassifier is not None:
        logger.info("Training XGBClassifier")
        t0 = time.time()
        xgb, xgb_p = _best_params(
            XGBClassifier,
            {"learning_rate": [0.01, 0.1], "max_depth": [3, 6], "n_estimators": [100, 200]},
            X_train, y_train, X_val, y_val,
            {"random_state": RANDOM_SEED, "verbosity": 0, "use_label_encoder": False,
             "eval_metric": "mlogloss", "subsample": 0.8}
        )
        preds = xgb.predict(X_val)
        probs = xgb.predict_proba(X_val)
        m = classification_metrics(y_val, preds, probs)
        m_log = {k: v for k, v in m.items() if k != "confusion_matrix"}
        m_log["training_time"] = time.time() - t0
        xgb_p["model"] = "XGBoost"
        log_sklearn_model(xgb, "xgboost_classifier", xgb_p, m_log, experiment)
        results["xgboost"] = (xgb, m, xgb_p)
    else:
        logger.warning("Skipping XGBoost classifier because xgboost is not installed.")

    # 4. LightGBM
    if LGBMClassifier is not None:
        logger.info("Training LGBMClassifier")
        t0 = time.time()
        lgbm, lgbm_p = _best_params(
            LGBMClassifier,
            {"learning_rate": [0.01, 0.1], "num_leaves": [31, 63], "n_estimators": [100, 200]},
            X_train, y_train, X_val, y_val,
            {"random_state": RANDOM_SEED, "verbosity": -1}
        )
        preds = lgbm.predict(X_val)
        probs = lgbm.predict_proba(X_val)
        m = classification_metrics(y_val, preds, probs)
        m_log = {k: v for k, v in m.items() if k != "confusion_matrix"}
        m_log["training_time"] = time.time() - t0
        lgbm_p["model"] = "LightGBM"
        log_sklearn_model(lgbm, "lightgbm_classifier", lgbm_p, m_log, experiment)
        results["lightgbm"] = (lgbm, m, lgbm_p)
    else:
        logger.warning("Skipping LightGBM classifier because lightgbm is not installed.")

    # 5. SVM
    logger.info("Training SVC")
    t0 = time.time()
    svm, svm_p = _best_params(
        SVC,
        {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]},
        X_train, y_train, X_val, y_val,
        {"kernel": "rbf", "probability": True, "random_state": RANDOM_SEED}
    )
    preds = svm.predict(X_val)
    probs = svm.predict_proba(X_val)
    m = classification_metrics(y_val, preds, probs)
    m_log = {k: v for k, v in m.items() if k != "confusion_matrix"}
    m_log["training_time"] = time.time() - t0
    svm_p["model"] = "SVM"
    log_sklearn_model(svm, "svm_classifier", svm_p, m_log, experiment)
    results["svm"] = (svm, m, svm_p)

    # Summary
    logger.info("=== Classification Results ===")
    for name, (_, m, _) in results.items():
        logger.info(f"  {name}: F1={m['f1_macro']:.4f}, Acc={m['accuracy']:.4f}")

    return results


def get_best_classifier(results: dict):
    """Return the model name and object with the highest validation F1 (macro)."""
    best_name = max(results, key=lambda k: results[k][1]["f1_macro"])
    best_model, best_metrics, _ = results[best_name]
    logger.info(f"Best classifier: {best_name} (F1={best_metrics['f1_macro']:.4f})")
    return best_name, best_model
