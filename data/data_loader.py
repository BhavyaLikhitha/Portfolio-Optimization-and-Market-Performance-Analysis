import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import TRAIN_END, VAL_END, SEQUENCE_LENGTH, BATCH_SIZE, RANDOM_SEED, get_logger

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except Exception:  # pragma: no cover - environment dependent
    torch = None
    Dataset = object
    DataLoader = None

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Time-based splitting
# ------------------------------------------------------------------

def split_by_time(df: pd.DataFrame, train_end: str = None, val_end: str = None):
    """Split a MultiIndex (date, ticker) DataFrame by time.

    Returns:
        train_df, val_df, test_df
    """
    if train_end is None:
        train_end = TRAIN_END
    if val_end is None:
        val_end = VAL_END

    dates = df.index.get_level_values("date")
    train_df = df[dates <= train_end]
    val_df = df[(dates > train_end) & (dates <= val_end)]
    test_df = df[dates > val_end]

    logger.info(f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return train_df, val_df, test_df


# ------------------------------------------------------------------
# Tabular data preparation (for classical ML)
# ------------------------------------------------------------------

def prepare_tabular_data(train_df, val_df, test_df, target_col, feature_cols=None):
    """Separate features/targets and scale using training statistics only.

    Args:
        train_df, val_df, test_df: DataFrames with features + target
        target_col: name of the target column (e.g. 'fwd_return_10d' or 'signal')
        feature_cols: list of feature column names; if None, all columns except
                      those starting with 'fwd_return_' and 'signal' are used

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test (numpy arrays), scaler
    """
    label_cols = [c for c in train_df.columns if c.startswith("fwd_return_") or c == "signal"]
    if feature_cols is None:
        feature_cols = [c for c in train_df.columns if c not in label_cols]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info(f"Tabular data — features: {len(feature_cols)}, target: {target_col}")
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_cols


# ------------------------------------------------------------------
# PyTorch Dataset for sequence models
# ------------------------------------------------------------------

class StockSequenceDataset(Dataset):
    """Sliding-window dataset for sequential DL models (LSTM, CNN, CNN-LSTM).

    Each sample is a window of `seq_len` consecutive days for a single stock.
    """

    def __init__(self, df: pd.DataFrame, target_col: str, feature_cols: list, seq_len: int = SEQUENCE_LENGTH):
        if torch is None:
            raise ImportError("PyTorch is required to create sequence datasets.")
        self.seq_len = seq_len
        self.samples = []
        self.targets = []

        label_cols = [c for c in df.columns if c.startswith("fwd_return_") or c == "signal"]
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in label_cols]

        # Group by ticker and create windows
        tickers = df.index.get_level_values("ticker").unique()
        for ticker in tickers:
            ticker_data = df.xs(ticker, level="ticker")
            features = ticker_data[feature_cols].values
            targets = ticker_data[target_col].values

            for i in range(len(features) - seq_len):
                self.samples.append(features[i : i + seq_len])
                self.targets.append(targets[i + seq_len - 1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y


def create_dataloaders(
    train_df, val_df, test_df, target_col, feature_cols,
    seq_len=SEQUENCE_LENGTH, batch_size=BATCH_SIZE,
):
    """Create PyTorch DataLoaders for train/val/test.

    The feature values should already be scaled before calling this.
    """
    if DataLoader is None:
        raise ImportError("PyTorch is required to create dataloaders.")

    train_ds = StockSequenceDataset(train_df, target_col, feature_cols, seq_len)
    val_ds = StockSequenceDataset(val_df, target_col, feature_cols, seq_len)
    test_ds = StockSequenceDataset(test_df, target_col, feature_cols, seq_len)

    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    logger.info(f"DataLoaders — train: {len(train_ds)} samples, val: {len(val_ds)}, test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
