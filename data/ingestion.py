from io import StringIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

from config import (
    DATA_PROVIDER,
    LOCAL_COMPANIES_FILE_CANDIDATES,
    LOCAL_DATA_DIR,
    LOCAL_SP500_FILE,
    LOCAL_STOCKS_FILE_CANDIDATES,
    SP500_INDEX_TICKER,
    TICKERS_URL,
    get_logger,
)

logger = get_logger(__name__)


def _local_path(name: str) -> Path:
    return Path(LOCAL_DATA_DIR) / name


def _find_first_existing(candidates):
    for candidate in candidates:
        path = _local_path(candidate)
        if path.exists():
            return path
    return None


def _read_local_stocks_frame() -> pd.DataFrame:
    stocks_path = _find_first_existing(LOCAL_STOCKS_FILE_CANDIDATES)
    if stocks_path is None:
        raise FileNotFoundError(
            f"No stock CSV found under {LOCAL_DATA_DIR}. Expected one of: {LOCAL_STOCKS_FILE_CANDIDATES}"
        )

    logger.info("Loading local stock history from %s", stocks_path)
    frame = pd.read_csv(stocks_path)
    rename_map = {
        "date": "Date",
        "Date": "Date",
        "open": "Open",
        "Open": "Open",
        "high": "High",
        "High": "High",
        "low": "Low",
        "Low": "Low",
        "close": "Close",
        "Close": "Close",
        "adj_close": "Adj Close",
        "Adj Close": "Adj Close",
        "volume": "Volume",
        "Volume": "Volume",
        "symbol": "Symbol",
        "ticker": "Symbol",
        "Ticker": "Symbol",
    }
    frame = frame.rename(columns=rename_map)
    required = {"Date", "Open", "High", "Low", "Close", "Volume", "Symbol"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Local stock CSV is missing required columns: {sorted(missing)}")

    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Symbol"] = frame["Symbol"].astype(str).str.replace(".", "-", regex=False)
    return frame


def _read_local_companies_frame(stocks_frame: pd.DataFrame) -> pd.DataFrame:
    companies_path = _find_first_existing(LOCAL_COMPANIES_FILE_CANDIDATES)
    if companies_path is not None:
        logger.info("Loading local company metadata from %s", companies_path)
        try:
            frame = pd.read_csv(companies_path, skipinitialspace=True, engine="python")
            frame.columns = [str(column).strip() for column in frame.columns]
            rename_map = {
                "symbol": "Symbol",
                "Symbol": "Symbol",
                "sector": "GICS Sector",
                "GICS Sector": "GICS Sector",
            }
            frame = frame.rename(columns=rename_map)
            if "Symbol" in frame.columns and "GICS Sector" in frame.columns:
                frame["Symbol"] = frame["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
                frame["GICS Sector"] = frame["GICS Sector"].astype(str).str.strip()
                return frame[["Symbol", "GICS Sector"]].drop_duplicates().reset_index(drop=True)
        except Exception as exc:
            logger.warning("Failed to parse local company metadata CSV (%s). Falling back to inferred tickers.", exc)

    logger.warning("Local company metadata CSV not found. Inferring ticker list from stock history.")
    inferred = pd.DataFrame(
        {
            "Symbol": sorted(stocks_frame["Symbol"].unique()),
            "GICS Sector": "Unknown",
        }
    )
    return inferred


def _read_local_sp500_frame() -> pd.Series:
    path = _local_path(LOCAL_SP500_FILE)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    logger.info("Loading benchmark S&P 500 series from %s", path)
    frame = pd.read_csv(path)
    rename_map = {
        "observation_date": "Date",
        "DATE": "Date",
        "date": "Date",
        "SP500": "Close",
        "close": "Close",
        "Close": "Close",
    }
    frame = frame.rename(columns=rename_map)
    required = {"Date", "Close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Benchmark CSV is missing required columns: {sorted(missing)}")

    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame = frame.dropna(subset=["Close"]).sort_values("Date").set_index("Date")
    return frame["Close"]


def _read_sp500_table():
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    request = Request(TICKERS_URL, headers=headers)
    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="ignore")
    return pd.read_html(StringIO(html))[0]


def fetch_sp500_tickers():
    logger.info("Fetching S&P 500 ticker list using provider '%s'", DATA_PROVIDER)

    if DATA_PROVIDER == "local_csv":
        stocks_frame = _read_local_stocks_frame()
        tickers_df = _read_local_companies_frame(stocks_frame)
    else:
        # FMP path intentionally commented out for easy switching back later.
        # payload = _fmp_request("sp500-constituent")
        # sp500_table = pd.DataFrame(payload)
        # sp500_table = sp500_table.rename(columns={"symbol": "Symbol", "sector": "GICS Sector"})
        # tickers_df = sp500_table[["Symbol", "GICS Sector"]].copy()
        # tickers_df["Symbol"] = tickers_df["Symbol"].astype(str).str.replace(".", "-", regex=False)

        # Yahoo/Wikipedia path intentionally commented out for easy switching back later.
        # sp500_table = _read_sp500_table()
        # tickers_df = sp500_table[["Symbol", "GICS Sector"]].copy()
        # tickers_df["Symbol"] = tickers_df["Symbol"].str.replace(".", "-", regex=False)
        raise ValueError("DATA_PROVIDER is not configured for the commented external providers.")

    logger.info("Found %s tickers", len(tickers_df))
    return tickers_df


def fetch_stock_data(tickers, start_date, end_date, min_history_pct=0.8):
    logger.info("Loading OHLCV data for %s tickers using provider '%s'", len(tickers), DATA_PROVIDER)

    if DATA_PROVIDER == "local_csv":
        frame = _read_local_stocks_frame()
        frame = frame[frame["Symbol"].isin(tickers)].copy()
        frame = frame[(frame["Date"] >= pd.to_datetime(start_date)) & (frame["Date"] <= pd.to_datetime(end_date))]
        if frame.empty:
            raise ValueError("Local stock CSV returned no rows for the requested tickers/date range.")

        ticker_frames = {}
        for ticker, ticker_frame in frame.groupby("Symbol"):
            ordered = ticker_frame.sort_values("Date").set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
            ticker_frames[ticker] = ordered

        data = pd.concat(ticker_frames, axis=1)
    else:
        # Yahoo path intentionally commented out for easy switching back later.
        # data = yf.download(
        #     tickers,
        #     start=start_date,
        #     end=end_date,
        #     group_by="ticker",
        #     threads=False,
        #     progress=False,
        #     auto_adjust=False,
        # )
        raise ValueError("DATA_PROVIDER is not configured for the commented external providers.")

    max_days = len(data)
    min_days = int(max_days * min_history_pct)

    valid_tickers = []
    for ticker in tickers:
        try:
            ticker_close = data[ticker]["Close"] if ticker in data.columns.get_level_values(0) else None
            if ticker_close is not None and ticker_close.notna().sum() >= min_days:
                valid_tickers.append(ticker)
        except (KeyError, TypeError):
            continue

    if not valid_tickers:
        raise ValueError("No tickers passed the minimum history filter from local data.")

    data = data[valid_tickers]
    data = data.ffill().dropna()

    logger.info("%s tickers passed validation (%s trading days, min %s required)", len(valid_tickers), max_days, min_days)
    return data


def get_sp500_index(start_date, end_date):
    logger.info("Loading S&P 500 benchmark using provider '%s'", DATA_PROVIDER)

    if DATA_PROVIDER == "local_csv":
        sp500 = _read_local_sp500_frame()
        sp500 = sp500.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
        if sp500.empty:
            raise ValueError("Local benchmark CSV returned no rows for the requested date range.")
        return sp500

    # Yahoo path intentionally commented out for easy switching back later.
    # sp500 = yf.download(SP500_INDEX_TICKER, start=start_date, end=end_date)
    # if sp500.empty:
    #     raise ValueError("Failed to download S&P 500 index data.")
    # return sp500["Close"].squeeze()
    raise ValueError("DATA_PROVIDER is not configured for the commented external providers.")


def get_close_prices(ohlcv_data):
    tickers = ohlcv_data.columns.get_level_values(0).unique()
    close_dict = {}
    for ticker in tickers:
        try:
            close_dict[ticker] = ohlcv_data[ticker]["Close"]
        except KeyError:
            continue
    return pd.DataFrame(close_dict)
