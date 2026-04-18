import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)
from config import get_logger

logger = get_logger(__name__)


def regression_metrics(y_true, y_pred) -> dict:
    """Compute regression evaluation metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Directional accuracy: % where sign of prediction matches sign of actual
    signs_match = np.sign(y_true) == np.sign(y_pred)
    directional_acc = signs_match.mean()

    metrics = {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": float(directional_acc),
    }
    return metrics


def classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Compute classification evaluation metrics.

    Args:
        y_true: true labels (0=sell, 1=hold, 2=buy)
        y_pred: predicted labels
        y_prob: predicted probabilities (n_samples, n_classes) for ROC-AUC

    Returns:
        dict of metrics + 'confusion_matrix' key with the matrix
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except ValueError:
            metrics["roc_auc"] = 0.0

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm

    return metrics


def portfolio_metrics(portfolio_returns, benchmark_returns) -> dict:
    """Compute portfolio performance metrics.

    Args:
        portfolio_returns: pd.Series of daily portfolio returns
        benchmark_returns: pd.Series of daily benchmark returns

    Returns:
        dict of performance metrics
    """
    port = np.asarray(portfolio_returns)
    bench = np.asarray(benchmark_returns)

    def _annualized_return(returns):
        cumulative = (1 + returns).prod()
        n_years = len(returns) / 252
        return float(cumulative ** (1 / n_years) - 1) if n_years > 0 else 0.0

    def _annualized_volatility(returns):
        return float(np.std(returns) * np.sqrt(252))

    def _sharpe_ratio(returns):
        ann_ret = _annualized_return(returns)
        ann_vol = _annualized_volatility(returns)
        return float(ann_ret / ann_vol) if ann_vol > 0 else 0.0

    def _sortino_ratio(returns):
        ann_ret = _annualized_return(returns)
        downside = returns[returns < 0]
        downside_vol = float(np.std(downside) * np.sqrt(252)) if len(downside) > 0 else 1e-10
        return float(ann_ret / downside_vol)

    def _max_drawdown(returns):
        cumulative = (1 + returns).cumprod()
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return float(drawdown.min())

    def _win_rate(returns):
        # Monthly win rate
        n_months = len(returns) // 21  # approximate trading days per month
        if n_months == 0:
            return 0.0
        monthly_returns = []
        for i in range(n_months):
            chunk = returns[i * 21 : (i + 1) * 21]
            monthly_returns.append(chunk.sum())
        monthly_returns = np.array(monthly_returns)
        return float((monthly_returns > 0).mean())

    port_ann_ret = _annualized_return(port)
    port_max_dd = _max_drawdown(port)

    metrics = {
        "total_return": float((1 + port).prod() - 1),
        "annualized_return": port_ann_ret,
        "annualized_volatility": _annualized_volatility(port),
        "sharpe_ratio": _sharpe_ratio(port),
        "sortino_ratio": _sortino_ratio(port),
        "max_drawdown": port_max_dd,
        "win_rate_monthly": _win_rate(port),
        "calmar_ratio": float(port_ann_ret / abs(port_max_dd)) if port_max_dd != 0 else 0.0,
        "benchmark_total_return": float((1 + bench).prod() - 1),
        "benchmark_annualized_return": _annualized_return(bench),
        "benchmark_sharpe": _sharpe_ratio(bench),
    }
    return metrics
