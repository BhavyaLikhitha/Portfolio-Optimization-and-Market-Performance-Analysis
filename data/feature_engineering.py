import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import (
    ROLLING_WINDOWS, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BOLLINGER_PERIOD, BOLLINGER_STD, ATR_PERIOD, STOCHASTIC_PERIOD,
    SMA_PERIODS, EMA_PERIODS, get_logger,
)

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Price-based features
# ------------------------------------------------------------------

def compute_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Compute return-based features for all stocks.

    Returns one row per (date, ticker) with columns:
        daily_return, log_return,
        rolling_mean_return_{w}, rolling_volatility_{w} for each window,
        momentum_{w} for windows [10, 20, 60],
        lagged_return_{l} for lags [1, 2, 5]
    """
    daily_ret = close.pct_change()
    log_ret = np.log(close / close.shift(1))

    frames = {
        "daily_return": daily_ret,
        "log_return": log_ret,
    }

    for w in ROLLING_WINDOWS:
        frames[f"rolling_mean_return_{w}"] = daily_ret.rolling(w).mean()
        frames[f"rolling_volatility_{w}"] = daily_ret.rolling(w).std()

    for w in [10, 20, 60]:
        frames[f"momentum_{w}"] = close.pct_change(w)

    for lag in [1, 2, 5]:
        frames[f"lagged_return_{lag}"] = daily_ret.shift(lag)

    return pd.concat(frames, axis=1)


# ------------------------------------------------------------------
# Technical indicators
# ------------------------------------------------------------------

def _rsi(close_series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close_series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _macd(close_series: pd.Series):
    ema_fast = close_series.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close_series.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(close_series: pd.Series):
    sma = close_series.rolling(BOLLINGER_PERIOD).mean()
    std = close_series.rolling(BOLLINGER_PERIOD).std()
    upper = sma + BOLLINGER_STD * std
    lower = sma - BOLLINGER_STD * std
    bandwidth = (upper - lower) / (sma + 1e-10)
    pct_b = (close_series - lower) / (upper - lower + 1e-10)
    return upper, lower, bandwidth, pct_b


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _obv(close_series: pd.Series, volume_series: pd.Series) -> pd.Series:
    direction = np.sign(close_series.diff())
    return (direction * volume_series).cumsum()


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = STOCHASTIC_PERIOD):
    lowest_low = low.rolling(period).min()
    highest_high = high.rolling(period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d = k.rolling(3).mean()
    return k, d


def compute_technical_indicators(ohlcv: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Compute technical indicators for a single stock."""
    close = ohlcv[(ticker, "Close")]
    high = ohlcv[(ticker, "High")]
    low = ohlcv[(ticker, "Low")]
    volume = ohlcv[(ticker, "Volume")]

    indicators = {}

    # RSI
    indicators["rsi"] = _rsi(close)

    # MACD
    macd_line, signal_line, histogram = _macd(close)
    indicators["macd"] = macd_line
    indicators["macd_signal"] = signal_line
    indicators["macd_histogram"] = histogram

    # Bollinger Bands
    upper, lower, bandwidth, pct_b = _bollinger(close)
    indicators["bb_upper"] = upper
    indicators["bb_lower"] = lower
    indicators["bb_bandwidth"] = bandwidth
    indicators["bb_pct_b"] = pct_b

    # Moving Averages
    for p in SMA_PERIODS:
        indicators[f"sma_{p}"] = close.rolling(p).mean()
    for p in EMA_PERIODS:
        indicators[f"ema_{p}"] = close.ewm(span=p, adjust=False).mean()

    # ATR
    indicators["atr"] = _atr(high, low, close)

    # OBV
    indicators["obv"] = _obv(close, volume)

    # Stochastic
    k, d = _stochastic(high, low, close)
    indicators["stochastic_k"] = k
    indicators["stochastic_d"] = d

    return pd.DataFrame(indicators, index=close.index)


# ------------------------------------------------------------------
# Risk metrics
# ------------------------------------------------------------------

def compute_risk_metrics(returns: pd.Series, market_returns: pd.Series, window: int = 60) -> pd.DataFrame:
    """Compute rolling risk metrics for a single stock."""
    metrics = {}

    # Beta
    rolling_cov = returns.rolling(window).cov(market_returns)
    rolling_var = market_returns.rolling(window).var()
    metrics["beta"] = rolling_cov / (rolling_var + 1e-10)

    # Rolling Sharpe (annualized, assuming 252 trading days)
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_std = returns.rolling(window).std() * np.sqrt(252)
    metrics["rolling_sharpe"] = rolling_mean / (rolling_std + 1e-10)

    # Rolling Sortino
    downside = returns.copy()
    downside[downside > 0] = 0
    downside_std = downside.rolling(window).std() * np.sqrt(252)
    metrics["rolling_sortino"] = rolling_mean / (downside_std + 1e-10)

    # Rolling Max Drawdown
    rolling_max = returns.rolling(window).apply(
        lambda x: (pd.Series(x).cumsum().cummax() - pd.Series(x).cumsum()).max(), raw=False
    )
    metrics["rolling_max_drawdown"] = rolling_max

    # Rolling VaR 95%
    metrics["var_95"] = returns.rolling(window).quantile(0.05)

    return pd.DataFrame(metrics, index=returns.index)


# ------------------------------------------------------------------
# Volume features
# ------------------------------------------------------------------

def compute_volume_features(volume: pd.Series) -> pd.DataFrame:
    """Volume MA ratio and rate of change."""
    vol_ma_20 = volume.rolling(20).mean()
    return pd.DataFrame({
        "volume_ma_ratio": volume / (vol_ma_20 + 1e-10),
        "volume_roc": volume.pct_change(),
    }, index=volume.index)


# ------------------------------------------------------------------
# Master builder
# ------------------------------------------------------------------

def build_feature_matrix(
    ohlcv_data: pd.DataFrame,
    fit_scaler: bool = True,
    scaler: StandardScaler = None,
    scale: bool = True,
):
    """Build the full feature matrix for all stocks.

    Args:
        ohlcv_data: MultiIndex DataFrame from ingestion (ticker, OHLCV)
        fit_scaler: if True, fit a new scaler; if False, use the provided one
        scaler: pre-fitted scaler (used for val/test sets)
        scale: if False, return raw features without applying a scaler

    Returns:
        features_df: DataFrame with MultiIndex (date, ticker) and 30+ feature columns
        scaler: fitted StandardScaler
    """
    tickers = ohlcv_data.columns.get_level_values(0).unique()
    logger.info(f"Building features for {len(tickers)} tickers")

    # Extract close prices for returns
    close_prices = pd.DataFrame({t: ohlcv_data[(t, "Close")] for t in tickers})
    daily_returns = close_prices.pct_change()
    market_returns = daily_returns.mean(axis=1)

    all_features = []

    for ticker in tickers:
        try:
            close_col = ohlcv_data[(ticker, "Close")]
            volume_col = ohlcv_data[(ticker, "Volume")]
            ret = daily_returns[ticker]

            # Return-based features
            ret_features = compute_returns(close_prices[[ticker]])
            # Flatten multi-level columns from single-ticker DataFrame
            ret_features.columns = [col[0] if isinstance(col, tuple) else col for col in ret_features.columns]

            # Technical indicators
            tech = compute_technical_indicators(ohlcv_data, ticker)

            # Risk metrics
            risk = compute_risk_metrics(ret, market_returns)

            # Volume features
            vol_feat = compute_volume_features(volume_col)

            # Combine all
            combined = pd.concat([ret_features, tech, risk, vol_feat], axis=1)
            combined["ticker"] = ticker
            all_features.append(combined)

        except Exception as e:
            logger.warning(f"Skipping {ticker}: {e}")
            continue

    features_df = pd.concat(all_features, axis=0)
    features_df = features_df.reset_index().rename(columns={"index": "date", "Date": "date"})

    # Set multi-index
    if "date" not in features_df.columns:
        features_df = features_df.reset_index().rename(columns={"index": "date"})
    features_df = features_df.set_index(["date", "ticker"])

    # Clean infinities and NaNs
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df.dropna(inplace=True)

    logger.info(f"Feature matrix shape: {features_df.shape}")

    # Scale only when explicitly requested. This lets the time split own the
    # scaler fit to avoid leakage from validation/test periods.
    feature_cols = features_df.columns.tolist()
    if scale:
        if fit_scaler:
            scaler = StandardScaler()
            features_df[feature_cols] = scaler.fit_transform(features_df[feature_cols])
        else:
            if scaler is None:
                raise ValueError("A fitted scaler must be provided when fit_scaler is False and scale is True.")
            features_df[feature_cols] = scaler.transform(features_df[feature_cols])

    return features_df, scaler
