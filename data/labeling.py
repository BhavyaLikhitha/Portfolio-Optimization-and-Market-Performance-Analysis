import pandas as pd
import numpy as np
from config import FORWARD_DAYS, PRIMARY_FORWARD_DAYS, BUY_THRESHOLD, SELL_THRESHOLD, get_logger

logger = get_logger(__name__)


def create_regression_labels(close_prices: pd.DataFrame, forward_days: list = None) -> pd.DataFrame:
    """Create forward return labels for regression.

    Args:
        close_prices: DataFrame with columns=tickers, index=dates
        forward_days: list of forward-looking horizons (default from config)

    Returns:
        DataFrame with MultiIndex (date, ticker) and columns fwd_return_{n}d
    """
    if forward_days is None:
        forward_days = FORWARD_DAYS

    all_labels = []
    for ticker in close_prices.columns:
        price = close_prices[ticker]
        ticker_labels = {}
        for n in forward_days:
            ticker_labels[f"fwd_return_{n}d"] = price.shift(-n) / price - 1
        df = pd.DataFrame(ticker_labels, index=price.index)
        df["ticker"] = ticker
        all_labels.append(df)

    labels = pd.concat(all_labels, axis=0)
    labels = labels.reset_index().rename(columns={"index": "date", "Date": "date"})
    if "date" not in labels.columns:
        labels = labels.reset_index().rename(columns={"index": "date"})
    labels = labels.set_index(["date", "ticker"])

    logger.info(f"Regression labels shape: {labels.shape}")
    return labels


def create_classification_labels(
    close_prices: pd.DataFrame,
    forward_days: int = None,
    buy_thresh: float = None,
    sell_thresh: float = None,
) -> pd.DataFrame:
    """Create buy/hold/sell classification labels.

    Labels:
        0 = sell  (forward return < sell_thresh)
        1 = hold  (between sell_thresh and buy_thresh)
        2 = buy   (forward return > buy_thresh)

    Returns:
        DataFrame with MultiIndex (date, ticker) and column 'signal'
    """
    if forward_days is None:
        forward_days = PRIMARY_FORWARD_DAYS
    if buy_thresh is None:
        buy_thresh = BUY_THRESHOLD
    if sell_thresh is None:
        sell_thresh = SELL_THRESHOLD

    all_labels = []
    for ticker in close_prices.columns:
        price = close_prices[ticker]
        fwd_ret = price.shift(-forward_days) / price - 1
        signal = pd.Series(1, index=price.index, name="signal")  # default hold
        signal[fwd_ret > buy_thresh] = 2   # buy
        signal[fwd_ret < sell_thresh] = 0  # sell
        df = pd.DataFrame({"signal": signal})
        df["ticker"] = ticker
        all_labels.append(df)

    labels = pd.concat(all_labels, axis=0)
    labels = labels.reset_index().rename(columns={"index": "date", "Date": "date"})
    if "date" not in labels.columns:
        labels = labels.reset_index().rename(columns={"index": "date"})
    labels = labels.set_index(["date", "ticker"])

    logger.info(f"Classification labels — buy: {(labels['signal']==2).sum()}, hold: {(labels['signal']==1).sum()}, sell: {(labels['signal']==0).sum()}")
    return labels


def merge_features_and_labels(features: pd.DataFrame, reg_labels: pd.DataFrame, clf_labels: pd.DataFrame) -> pd.DataFrame:
    """Inner-join features with regression and classification labels.

    Drops rows where any label is NaN (near the end of dataset where forward
    returns cannot be computed).
    """
    merged = features.join(reg_labels, how="inner").join(clf_labels, how="inner")
    merged.dropna(inplace=True)
    logger.info(f"Merged dataset shape: {merged.shape}")
    return merged
