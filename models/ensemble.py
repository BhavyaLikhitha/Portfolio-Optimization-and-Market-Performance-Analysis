import numpy as np
from sklearn.linear_model import Ridge
from config import RANDOM_SEED, MLFLOW_EXPERIMENTS, get_logger
from evaluation.metrics import regression_metrics, classification_metrics
from tracking.mlflow_tracking import log_sklearn_model

logger = get_logger(__name__)


def build_stacking_regressor(base_models, X_train, y_train, X_val, y_val):
    """Build a stacking ensemble from top regression models.

    Uses predictions from base models as meta-features for a Ridge meta-learner.

    Args:
        base_models: list of (name, fitted_model) tuples (top 3)
        X_train, y_train, X_val, y_val: numpy arrays

    Returns:
        dict with 'base_models', 'meta_learner', 'val_metrics'
    """
    logger.info(f"Building stacking regressor with {len(base_models)} base models")

    # Generate meta-features from base model predictions
    train_meta = np.column_stack([m.predict(X_train) for _, m in base_models])
    val_meta = np.column_stack([m.predict(X_val) for _, m in base_models])

    # Train meta-learner
    meta = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    meta.fit(train_meta, y_train)

    # Evaluate
    preds = meta.predict(val_meta)
    m = regression_metrics(y_val, preds)

    # Log
    params = {
        "model": "StackingRegressor",
        "base_models": str([n for n, _ in base_models]),
        "meta_learner": "Ridge",
    }
    log_sklearn_model(meta, "stacking_regressor", params, m, MLFLOW_EXPERIMENTS["regression"])
    logger.info(f"Stacking regressor: R²={m['r2']:.4f}, MAE={m['mae']:.6f}")

    return {
        "base_models": base_models,
        "meta_learner": meta,
        "val_metrics": m,
    }


def predict_stacking(ensemble, X):
    """Generate predictions from stacking ensemble."""
    meta_features = np.column_stack([m.predict(X) for _, m in ensemble["base_models"]])
    return ensemble["meta_learner"].predict(meta_features)


def build_voting_classifier(base_models, X_val, y_val):
    """Build a majority-vote classifier from top classification models.

    Args:
        base_models: list of (name, fitted_model) tuples (top 3)
        X_val, y_val: numpy arrays for evaluation

    Returns:
        dict with 'base_models', 'val_metrics'
    """
    logger.info(f"Building voting classifier with {len(base_models)} base models")

    # Majority vote
    all_preds = np.column_stack([m.predict(X_val) for _, m in base_models])
    # Mode across columns (majority vote)
    from scipy.stats import mode
    voted_preds = mode(all_preds, axis=1, keepdims=False).mode

    # Average probabilities for ROC-AUC
    all_probs = [m.predict_proba(X_val) for _, m in base_models]
    avg_probs = np.mean(all_probs, axis=0)

    m = classification_metrics(y_val, voted_preds, avg_probs)
    m_log = {k: v for k, v in m.items() if k != "confusion_matrix"}

    params = {
        "model": "VotingClassifier",
        "base_models": str([n for n, _ in base_models]),
    }
    log_sklearn_model(None, "voting_classifier", params, m_log, MLFLOW_EXPERIMENTS["classification"])
    logger.info(f"Voting classifier: F1={m['f1_macro']:.4f}, Acc={m['accuracy']:.4f}")

    return {
        "base_models": base_models,
        "val_metrics": m,
    }


def predict_voting(ensemble, X):
    """Generate predictions from voting classifier."""
    from scipy.stats import mode
    all_preds = np.column_stack([m.predict(X) for _, m in ensemble["base_models"]])
    return mode(all_preds, axis=1, keepdims=False).mode


def build_weighted_blend(models_with_weights, X):
    """Weighted average of regression predictions.

    Args:
        models_with_weights: list of (name, model, weight) tuples
        X: features

    Returns:
        blended predictions
    """
    total_weight = sum(w for _, _, w in models_with_weights)
    blended = sum(w * m.predict(X) for _, m, w in models_with_weights) / total_weight
    return blended
