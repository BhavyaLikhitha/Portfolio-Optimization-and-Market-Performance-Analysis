import logging
import torch

# ============================================================
# Random Seed
# ============================================================
RANDOM_SEED = 42

# ============================================================
# Data
# ============================================================
TICKERS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_INDEX_TICKER = "^GSPC"
START_DATE = "2015-01-01"
TRAIN_END = "2021-12-31"
VAL_END = "2023-06-30"
# Test: everything after VAL_END

# ============================================================
# Feature Engineering
# ============================================================
ROLLING_WINDOWS = [5, 10, 20, 60]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
ATR_PERIOD = 14
STOCHASTIC_PERIOD = 14
SMA_PERIODS = [20, 50, 200]
EMA_PERIODS = [12, 26]

# ============================================================
# Labeling
# ============================================================
FORWARD_DAYS = [5, 10, 30]
PRIMARY_FORWARD_DAYS = 10
BUY_THRESHOLD = 0.02
SELL_THRESHOLD = -0.02

# ============================================================
# Deep Learning
# ============================================================
SEQUENCE_LENGTH = 60
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
DROPOUT = 0.3
EARLY_STOPPING_PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Portfolio
# ============================================================
TOP_N_STOCKS = 50
MAX_WEIGHT_PER_STOCK = 0.05

# ============================================================
# MLflow
# ============================================================
MLFLOW_TRACKING_URI = "mlruns"
MLFLOW_EXPERIMENTS = {
    "regression": "return-regression",
    "classification": "signal-classification",
    "clustering": "clustering",
    "backtest": "portfolio-backtest",
}

# ============================================================
# Paths
# ============================================================
OUTPUTS_DIR = "outputs"
PLOTS_DIR = "outputs/plots"
MODELS_DIR = "outputs/models"

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def get_logger(name):
    return logging.getLogger(name)
