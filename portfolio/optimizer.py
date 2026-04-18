import numpy as np
from scipy.optimize import minimize
from config import MAX_WEIGHT_PER_STOCK, get_logger

logger = get_logger(__name__)


def _neg_sharpe(weights, expected_returns, cov_matrix):
    """Negative Sharpe ratio (to minimize)."""
    port_return = np.dot(weights, expected_returns) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return -port_return / (port_vol + 1e-10)


def optimize_portfolio(expected_returns, cov_matrix, max_weight=MAX_WEIGHT_PER_STOCK):
    """Mean-variance optimization to maximize Sharpe ratio.

    Args:
        expected_returns: array of expected daily returns per stock
        cov_matrix: covariance matrix of daily returns
        max_weight: maximum weight per stock

    Returns:
        optimal weights array, or equal weights if optimization fails
    """
    n = len(expected_returns)
    initial_weights = np.ones(n) / n
    bounds = [(0, max_weight) for _ in range(n)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    result = minimize(
        _neg_sharpe, initial_weights,
        args=(expected_returns, cov_matrix),
        method="SLSQP", bounds=bounds, constraints=constraints,
    )

    if result.success:
        logger.info(f"Portfolio optimized: {(result.x > 0.001).sum()} non-zero positions")
        return result.x
    else:
        logger.warning("Optimization failed, using equal weights")
        return initial_weights


def equal_weight_portfolio(n_stocks):
    """Return equal-weight allocation."""
    return np.ones(n_stocks) / n_stocks
