import pandas as pd
import numpy as np
from config import TOP_N_STOCKS, get_logger

logger = get_logger(__name__)


def generate_signals(regression_preds, classification_preds, tickers):
    """Combine regression and classification predictions into stock selection.

    Args:
        regression_preds: array of predicted forward returns (per stock)
        classification_preds: array of predicted classes (0=sell, 1=hold, 2=buy)
        tickers: list of ticker symbols (same order as preds)

    Returns:
        DataFrame with columns: ticker, predicted_return, signal
        Filtered to buy signals only, ranked by predicted return, top N
    """
    df = pd.DataFrame({
        "ticker": tickers,
        "predicted_return": regression_preds,
        "signal": classification_preds,
    })

    # Keep only buy signals
    buy_signals = df[df["signal"] == 2].copy()

    if len(buy_signals) == 0:
        logger.warning("No buy signals generated. Falling back to top stocks by predicted return.")
        buy_signals = df.nlargest(TOP_N_STOCKS, "predicted_return")
    else:
        # Rank by predicted return, take top N
        buy_signals = buy_signals.nlargest(TOP_N_STOCKS, "predicted_return")

    logger.info(f"Selected {len(buy_signals)} stocks from {len(df)} candidates")
    return buy_signals.reset_index(drop=True)
