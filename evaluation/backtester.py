import pandas as pd
import numpy as np
from portfolio.optimizer import optimize_portfolio
from portfolio.signals import generate_signals
from evaluation.metrics import portfolio_metrics
from config import get_logger

logger = get_logger(__name__)


def walk_forward_backtest(
    test_df, close_prices, regression_model, classification_model,
    feature_cols, scaler, rebalance_freq="M",
):
    """Walk-forward backtest with periodic rebalancing.

    At each rebalance date:
    1. Get latest features for all stocks
    2. Predict returns and signals
    3. Select stocks and optimize weights
    4. Hold until next rebalance, tracking daily returns

    Args:
        test_df: merged DataFrame (features + labels) for the test period
        close_prices: full close price DataFrame (all dates, all tickers)
        regression_model: fitted model with .predict()
        classification_model: fitted model with .predict()
        feature_cols: list of feature column names
        scaler: fitted StandardScaler for features
        rebalance_freq: pandas frequency string ('M' = monthly)

    Returns:
        portfolio_returns: Series of daily portfolio returns
        rebalance_log: list of dicts with rebalance details
    """
    test_dates = pd.Index(test_df.index.get_level_values("date").unique()).sort_values()
    rebalance_dates = pd.Index(
        test_dates.to_series().groupby(test_dates.to_period(rebalance_freq)).first().tolist()
    )

    portfolio_returns = []
    rebalance_log = []
    current_weights = None
    current_tickers = None

    for i, reb_date in enumerate(rebalance_dates):
        # Get features at rebalance date
        try:
            date_features = test_df.xs(reb_date, level="date")
        except KeyError:
            continue

        tickers = date_features.index.tolist()
        X_raw = date_features[feature_cols].values
        X = scaler.transform(X_raw) if scaler is not None else X_raw

        # Predict
        reg_preds = regression_model.predict(X)
        clf_preds = classification_model.predict(X)

        # Generate signals and select stocks
        signals = generate_signals(reg_preds, clf_preds, tickers)
        selected_tickers = signals["ticker"].tolist()
        selected_expected_returns = signals["predicted_return"].to_numpy(dtype=float)

        if len(selected_tickers) < 2:
            continue

        # Get historical returns for covariance estimate
        hist_end = reb_date
        hist_returns = close_prices[selected_tickers].pct_change().dropna()
        hist_returns = hist_returns.loc[:hist_end].tail(252)  # last year
        hist_returns = hist_returns.dropna(axis=1, how="any")

        aligned_tickers = hist_returns.columns.tolist()
        if len(aligned_tickers) < 2:
            continue

        expected_ret_map = dict(zip(selected_tickers, selected_expected_returns))
        expected_rets = np.array([expected_ret_map[ticker] for ticker in aligned_tickers], dtype=float)
        cov_mat = hist_returns.cov().values

        # Optimize weights
        weights = optimize_portfolio(expected_rets, cov_mat)

        current_weights = weights
        current_tickers = aligned_tickers

        rebalance_log.append({
            "date": reb_date,
            "n_stocks": len(aligned_tickers),
            "tickers": aligned_tickers[:10],  # log top 10
            "all_tickers": aligned_tickers,
            "weights": weights.tolist(),
        })

        # Calculate daily returns until next rebalance
        if i + 1 < len(rebalance_dates):
            next_reb = rebalance_dates[i + 1]
        else:
            next_reb = test_dates[-1]

        period_mask = (test_dates >= reb_date) & (test_dates < next_reb)
        period_dates = test_dates[period_mask]

        period_returns = close_prices[current_tickers].pct_change().reindex(period_dates)
        for date, daily_rets in period_returns.iterrows():
            if daily_rets.isna().all():
                continue
            daily_rets = daily_rets.fillna(0).values
            port_ret = np.dot(current_weights, daily_rets)
            portfolio_returns.append({"date": date, "return": port_ret})

    portfolio_returns = pd.DataFrame(portfolio_returns).set_index("date")["return"]
    logger.info(f"Backtest complete: {len(portfolio_returns)} trading days, {len(rebalance_log)} rebalances")
    return portfolio_returns, rebalance_log


def compute_benchmark_returns(sp500_prices, test_start, test_end):
    """Compute daily returns for S&P 500 over test period."""
    benchmark = sp500_prices.loc[test_start:test_end]
    returns = benchmark.pct_change().dropna()
    return returns


def backtest_report(portfolio_returns, benchmark_returns):
    """Compute and log backtest metrics."""
    # Align dates
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    port = portfolio_returns.loc[common]
    bench = benchmark_returns.loc[common]

    metrics = portfolio_metrics(port.values, bench.values)

    logger.info("=== Backtest Report ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    return metrics
