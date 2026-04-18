import os
import pickle
from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENTS, MODELS_DIR, get_logger

try:
    import mlflow
except Exception:  # pragma: no cover - optional dependency
    mlflow = None

try:
    import torch
except Exception:  # pragma: no cover - environment dependent
    torch = None

logger = get_logger(__name__)


def setup_experiment(experiment_name: str) -> str:
    """Set MLflow tracking URI and create/get experiment."""
    if mlflow is None:
        logger.warning("MLflow is not installed. Skipping experiment setup for '%s'.", experiment_name)
        return ""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    logger.info(f"MLflow experiment '{experiment_name}' ready (id={experiment.experiment_id})")
    return experiment.experiment_id


def log_sklearn_model(model, model_name: str, params: dict, metrics: dict, experiment_name: str) -> str:
    """Log a scikit-learn model run to MLflow.

    Returns:
        run_id
    """
    if mlflow is None:
        logger.warning("MLflow is not installed. Skipping sklearn logging for '%s'.", model_name)
        return ""

    setup_experiment(experiment_name)
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Save model artifact
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(model_path)

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Logged sklearn model '{model_name}' — run_id={run_id}")
        return run_id


def log_pytorch_model(model, model_name: str, params: dict, metrics: dict,
                      experiment_name: str, artifacts: dict = None) -> str:
    """Log a PyTorch model run to MLflow.

    Args:
        artifacts: dict of {artifact_name: file_path} to log (e.g. loss curves, plots)

    Returns:
        run_id
    """
    if torch is None:
        raise ImportError("PyTorch is required to log PyTorch models.")
    if mlflow is None:
        logger.warning("MLflow is not installed. Skipping PyTorch logging for '%s'.", model_name)
        return ""

    setup_experiment(experiment_name)
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Save model state dict
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pt")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # Log additional artifacts (plots, etc.)
        if artifacts:
            for name, path in artifacts.items():
                if os.path.exists(path):
                    mlflow.log_artifact(path)

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Logged PyTorch model '{model_name}' — run_id={run_id}")
        return run_id


def log_backtest_results(metrics: dict, experiment_name: str, artifacts: dict = None) -> str:
    """Log portfolio backtest metrics to MLflow."""
    if mlflow is None:
        logger.warning("MLflow is not installed. Skipping backtest logging.")
        return ""

    setup_experiment(experiment_name)
    with mlflow.start_run(run_name="portfolio_backtest"):
        mlflow.log_metrics(metrics)
        if artifacts:
            for name, path in artifacts.items():
                if os.path.exists(path):
                    mlflow.log_artifact(path)
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Logged backtest results — run_id={run_id}")
        return run_id


def get_best_run(experiment_name: str, metric: str, maximize: bool = True):
    """Query MLflow for the best run by a given metric.

    Returns:
        (run_id, params, metrics) or None if no runs exist
    """
    if mlflow is None:
        logger.warning("MLflow is not installed. No run search performed.")
        return None

    setup_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    order = "DESC" if maximize else "ASC"
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )
    if runs.empty:
        return None
    best = runs.iloc[0]
    run_id = best["run_id"]
    params = {k.replace("params.", ""): v for k, v in best.items() if k.startswith("params.")}
    metrics_out = {k.replace("metrics.", ""): v for k, v in best.items() if k.startswith("metrics.")}
    logger.info(f"Best run for {metric}: {run_id}")
    return run_id, params, metrics_out
