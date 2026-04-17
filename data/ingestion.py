import pandas as pd
import yfinance as yf
from config import TICKERS_URL, SP500_INDEX_TICKER, get_logger

logger = get_logger(__name__)


def fetch_sp500_tickers():
    """Fetch S&P 500 tickers and sectors from Wikipedia."""
    logger.info("Fetching S&P 500 ticker list from Wikipedia")
    sp500_table = pd.read_html(TICKERS_URL)[0]
    tickers_df = sp500_table[["Symbol", "GICS Sector"]].copy()
    # Fix dot notation for Yahoo Finance (BRK.B -> BRK-B)
    tickers_df["Symbol"] = tickers_df["Symbol"].str.replace(".", "-", regex=False)
    logger.info(f"Found {len(tickers_df)} tickers")
    return tickers_df


def fetch_stock_data(tickers, start_date, end_date, min_history_pct=0.8):
    """Download OHLCV data for given tickers.

    Args:
        tickers: list of ticker symbols
        start_date: start date string
        end_date: end date string
        min_history_pct: drop tickers with less than this fraction of trading days

    Returns:
        DataFrame with MultiIndex columns (ticker, OHLCV fields)
    """
    logger.info(f"Downloading OHLCV data for {len(tickers)} tickers")
    data = yf.download(tickers, start=start_date, end=end_date, group_by="ticker", threads=True)

    if data.empty:
        raise ValueError("yfinance returned no data. Check date range or tickers.")

    # Determine max possible trading days from the index
    max_days = len(data)
    min_days = int(max_days * min_history_pct)

    # Filter out tickers with too much missing data
    valid_tickers = []
    for ticker in tickers:
        try:
            ticker_close = data[ticker]["Close"] if ticker in data.columns.get_level_values(0) else None
            if ticker_close is not None and ticker_close.notna().sum() >= min_days:
                valid_tickers.append(ticker)
        except (KeyError, TypeError):
            continue

    if not valid_tickers:
        raise ValueError("No tickers passed the minimum history filter.")

    data = data[valid_tickers]

    # Forward-fill remaining gaps, then drop any leading NaNs
    data = data.ffill().dropna()

    logger.info(f"{len(valid_tickers)} tickers passed validation ({max_days} trading days, min {min_days} required)")
    return data


def get_sp500_index(start_date, end_date):
    """Download the actual S&P 500 index (^GSPC) for benchmarking."""
    logger.info("Downloading S&P 500 index data")
    sp500 = yf.download(SP500_INDEX_TICKER, start=start_date, end=end_date)
    if sp500.empty:
        raise ValueError("Failed to download S&P 500 index data.")
    return sp500["Close"].squeeze()


def get_close_prices(ohlcv_data):
    """Extract close prices from multi-level OHLCV DataFrame.

    Returns:
        DataFrame with columns = tickers, index = dates, values = close prices
    """
    tickers = ohlcv_data.columns.get_level_values(0).unique()
    close_dict = {}
    for ticker in tickers:
        try:
            close_dict[ticker] = ohlcv_data[ticker]["Close"]
        except KeyError:
            continue
    return pd.DataFrame(close_dict)
