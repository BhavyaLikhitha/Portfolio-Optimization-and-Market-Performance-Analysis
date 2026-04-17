# 📈 Portfolio Optimization & Market Performance Analysis using ML/DL 📊

### An End-to-End ML/DL Pipeline for Stock Return Prediction, Trading Signal Classification & Portfolio Construction

> *12+ ML/DL models (classical + PyTorch/CUDA) trained on S&P 500 data, with walk-forward backtesting, SHAP explainability, MLflow experiment tracking, and Streamlit dashboard -- benchmarked against the S&P 500 index.*

---

## 🔍 Project Overview

This project is a **full end-to-end ML/DL pipeline** that approaches portfolio construction as a machine learning problem -- predicting future stock returns (regression), generating buy/hold/sell trading signals (classification), and selecting optimal portfolio weights using model predictions instead of static rules.

The pipeline processes **S&P 500 stock data (2015-present)** with **30+ engineered features** (technical indicators, risk metrics, volume signals), trains **12+ models** across classical ML and deep learning, evaluates them head-to-head with proper time-series validation, and constructs a portfolio that is walk-forward backtested against the S&P 500.

Every component follows production ML practices: time-based splits (no data leakage), MLflow experiment tracking, SHAP explainability, and proper evaluation metrics beyond just accuracy.

---

## ❓ What Problem Does This Solve

Traditional portfolio allocation relies on static rules -- market-cap weighting, sector diversification, or mean-variance optimization on historical averages. These approaches treat stock selection as a finance problem. **This project treats it as an ML problem.**

| Problem | Traditional Approach | What This Pipeline Does |
|---------|---------------------|------------------------|
| Stock selection based on historical averages | Pick stocks with highest past Sharpe ratio | Train regression models to **predict forward returns** -- select stocks based on learned patterns, not backward-looking metrics |
| No actionable trading signals | Buy-and-hold or manual chart reading | Classification models generate **buy/hold/sell signals** per stock with probability scores |
| Single model, no comparison | Run one LSTM, call it done | **12+ models compared head-to-head** -- Linear, Ridge, Lasso, RF, XGBoost, LightGBM, SVR, MLP, LSTM, 1D-CNN, CNN-LSTM, Autoencoder |
| Features limited to price | Use only close price as input | **30+ engineered features**: RSI, MACD, Bollinger Bands, ATR, OBV, Stochastic, rolling Sharpe/Sortino, beta, VaR, volume ratios |
| Random train/test split on time-series | Shuffle data, leak future information | **Strict time-based splitting** -- train on 2015-2021, validate on 2022-2023H1, test on 2023H2+ |
| No model interpretability | Black-box predictions | **SHAP values** show which features drive each prediction |
| No experiment tracking | Lose track of what was tried | **MLflow** logs every run -- params, metrics, artifacts, model registry |
| No realistic evaluation | Report R² on training set | **Walk-forward backtesting** with monthly rebalancing, compared to S&P 500 benchmark |

---

## 🌍 Why This Project Is Different

| Aspect | This Project |
|--------|--------------|
| Models | **12+ models**: 7 classical + 5 deep learning, all compared on same data |
| Task | **Both** -- regression (predict returns) AND classification (buy/hold/sell) on same dataset |
| Features | **30+ engineered features**: technical indicators, risk metrics, volume signals |
| Evaluation | MSE, MAE, R², **directional accuracy**, F1, ROC-AUC, confusion matrices, SHAP |
| Splitting | **Time-based split**: train < 2022, val = 2022-2023H1, test = 2023H2+ |
| Deep Learning | **MLP vs LSTM vs 1D-CNN vs CNN-LSTM** -- each for both regression and classification |
| Backtesting | **Walk-forward backtest** with monthly rebalancing, Sharpe/Sortino/drawdown vs S&P 500 |
| Tracking | **MLflow** tracks every experiment, every hyperparameter, every metric |
| Explainability | **SHAP values** for tree-based models, feature importance analysis |
| Portfolio | Ensemble predictions → stock selection → **mean-variance optimization** → backtest |

---

## 📐 Scope

| Layer | What's Built |
|---|---|
| **Data Engineering** | S&P 500 OHLCV data via yfinance, ticker cleaning/validation, time-based train/val/test splitting, PyTorch DataLoaders with sliding windows |
| **Feature Engineering** | 30+ features: technical indicators (RSI, MACD, Bollinger, ATR, OBV, Stochastic), rolling risk metrics (Sharpe, Sortino, beta, VaR, drawdown), volume signals, lagged returns, momentum |
| **Classical ML** | 7 regressors (Linear, Ridge, Lasso, RF, XGBoost, LightGBM, SVR) + 5 classifiers (Logistic, RF, XGBoost, LightGBM, SVM) with hyperparameter tuning |
| **Deep Learning** | 5 PyTorch models (MLP, LSTM, 1D-CNN, CNN-LSTM, Autoencoder) with CUDA support, early stopping, LR scheduling |
| **Clustering** | K-Means, Hierarchical, DBSCAN + Autoencoder-based deep clustering on learned representations |
| **Ensemble** | Stacking regressor (top 3 models as meta-features), voting classifier, weighted blending |
| **Portfolio** | Model predictions → buy signals → stock ranking → mean-variance optimization (max Sharpe, max 5% per stock) |
| **Backtesting** | Walk-forward validation with monthly rebalancing on held-out test period, compared to S&P 500 |
| **Explainability** | SHAP summary plots, feature importance rankings for tree-based models |
| **Experiment Tracking** | MLflow logging for all runs -- parameters, metrics, trained model artifacts, plots |
| **Dashboard** | Streamlit app with model comparison, training curves, SHAP plots, portfolio backtest results |

---

## 🛠️ Tech Stack

### Data & Features
| Technology | Purpose |
|---|---|
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | Data manipulation, feature engineering, rolling computations |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Numerical operations, array processing |
| ![yfinance](https://img.shields.io/badge/yfinance-000000?style=flat&logoColor=white) | S&P 500 OHLCV data + benchmark index download |

### Machine Learning
| Technology | Purpose |
|---|---|
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | 7 regressors, 5 classifiers, preprocessing, clustering, evaluation metrics |
| ![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat&logoColor=white) | Gradient boosted regression + classification |
| ![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat&logoColor=white) | Fast gradient boosted regression + classification |

### Deep Learning
| Technology | Purpose |
|---|---|
| ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) | 5 models: MLP, LSTM, 1D-CNN, CNN-LSTM, Autoencoder |
| ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) | GPU acceleration for all PyTorch models (auto-detect with CPU fallback) |

### Evaluation & Tracking
| Technology | Purpose |
|---|---|
| ![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white) | Experiment tracking -- params, metrics, model artifacts for all 12+ runs |
| ![SHAP](https://img.shields.io/badge/SHAP-000000?style=flat&logoColor=white) | Feature importance and model explainability for tree-based models |

### Visualization & Serving
| Technology | Purpose |
|---|---|
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logoColor=white) | Training curves, confusion matrices, equity curves, SHAP plots |
| ![Seaborn](https://img.shields.io/badge/Seaborn-444876?style=flat&logoColor=white) | Heatmaps, distribution plots, model comparison charts |
| ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) | Interactive charts in dashboard |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) | Interactive dashboard -- model comparison, backtest results, SHAP viewer |

---

## 🏗️ Architecture

```
S&P 500 Data (yfinance: OHLCV + benchmark)
    │
    ▼
Feature Engineering (30+ features)
    │── Technical: RSI, MACD, Bollinger, ATR, OBV, Stochastic, MAs
    │── Risk: beta, Sharpe, Sortino, drawdown, VaR
    │── Price: returns, momentum, lagged returns, volatility
    │── Volume: MA ratio, rate of change
    │
    ▼
Label Creation
    │── Regression targets: forward 5/10/30-day returns
    │── Classification targets: buy (>+2%) / hold / sell (<-2%)
    │
    ▼
Time-Based Split (NO random shuffling)
    │── Train: 2015-01-01 → 2021-12-31
    │── Validation: 2022-01-01 → 2023-06-30
    │── Test: 2023-07-01 → present
    │
    ▼
┌─────────────────────────────┐    ┌──────────────────────────────┐
│   Classical ML (sklearn)    │    │   Deep Learning (PyTorch)    │
│                             │    │                              │
│   Regression:               │    │   Regression + Classification│
│   • Linear, Ridge, Lasso    │    │   • MLP                      │
│   • Random Forest           │    │   • LSTM (2-layer, 128 units)│
│   • XGBoost, LightGBM       │    │   • 1D-CNN (64→128→64)       │
│   • SVR                     │    │   • CNN-LSTM hybrid           │
│                             │    │   • Autoencoder (latent=32)   │
│   Classification:           │    │                              │
│   • Logistic Regression     │    │   CUDA when available        │
│   • Random Forest           │    │   Early stopping + LR sched  │
│   • XGBoost, LightGBM       │    │                              │
│   • SVM                     │    │                              │
└──────────────┬──────────────┘    └──────────────┬───────────────┘
               │                                  │
               └──────────┬───────────────────────┘
                          │
                    MLflow Tracking
                   (every run logged)
                          │
                          ▼
               ┌──────────────────┐
               │    Ensemble      │
               │  Stacking (reg)  │
               │  Voting (clf)    │
               │  Weighted blend  │
               └────────┬─────────┘
                        │
                        ▼
               ┌──────────────────┐
               │  Stock Selection │
               │  predicted return│
               │  + buy signal    │
               │  → top 50 stocks │
               └────────┬─────────┘
                        │
                        ▼
               ┌──────────────────┐
               │    Portfolio     │
               │  Mean-Variance   │
               │  Optimization    │
               │  (max Sharpe,    │
               │   ≤5% per stock) │
               └────────┬─────────┘
                        │
                        ▼
               ┌──────────────────┐
               │  Walk-Forward    │
               │  Backtest        │
               │  vs S&P 500     │
               └────────┬─────────┘
                        │
                        ▼
               ┌──────────────────┐
               │   Streamlit      │
               │   Dashboard      │
               └──────────────────┘
```

### Why These Tools

| Tool | Why Chosen Over Alternatives |
|---|---|
| **PyTorch** (not TensorFlow) | Industry standard for research + production. 55%+ market share in 2026. Clean API for custom architectures |
| **XGBoost + LightGBM** (both) | XGBoost = gold standard for tabular. LightGBM = faster training. Comparing both shows rigor |
| **SHAP** (not just feature_importances_) | SHAP gives per-prediction explanations, not just global importance. Shows *how* features affect each stock's prediction |
| **MLflow** (not W&B) | Free, local, no account needed. Stores runs in `mlruns/` -- reviewer can clone repo and browse experiments |
| **Walk-forward backtest** (not random split) | Time-series data cannot be randomly split. Walk-forward simulates real trading: train on past, predict future, rebalance monthly |
| **Streamlit** (not Dash/Flask) | Fastest path to interactive dashboard. Built-in charting, model selector, metric display |

---

## 🧠 ML Models

### Regression Models (Predict Forward 10-Day Returns)

| Model | Framework | Key Hyperparameters | MSE | MAE | R² | Directional Accuracy |
|-------|-----------|--------------------|----|-----|-----|---------------------|
| Linear Regression | scikit-learn | baseline | -- | -- | -- | -- |
| Ridge | scikit-learn | alpha tuned | -- | -- | -- | -- |
| Lasso | scikit-learn | alpha tuned | -- | -- | -- | -- |
| Random Forest | scikit-learn | n_estimators, max_depth | -- | -- | -- | -- |
| XGBoost | xgboost | lr, max_depth, n_estimators | -- | -- | -- | -- |
| LightGBM | lightgbm | lr, num_leaves, n_estimators | -- | -- | -- | -- |
| SVR | scikit-learn | C, gamma, kernel=rbf | -- | -- | -- | -- |
| MLP | PyTorch | 256→128→64, dropout=0.3 | -- | -- | -- | -- |
| LSTM | PyTorch + CUDA | 2-layer, hidden=128 | -- | -- | -- | -- |
| 1D-CNN | PyTorch + CUDA | Conv(64→128→64), MaxPool | -- | -- | -- | -- |
| CNN-LSTM | PyTorch + CUDA | Conv→LSTM→FC | -- | -- | -- | -- |

### Classification Models (Buy / Hold / Sell Signals)

| Model | Framework | Accuracy | Precision | Recall | F1 (Macro) | ROC-AUC |
|-------|-----------|----------|-----------|--------|------------|---------|
| Logistic Regression | scikit-learn | -- | -- | -- | -- | -- |
| Random Forest | scikit-learn | -- | -- | -- | -- | -- |
| XGBoost | xgboost | -- | -- | -- | -- | -- |
| LightGBM | lightgbm | -- | -- | -- | -- | -- |
| SVM | scikit-learn | -- | -- | -- | -- | -- |
| MLP | PyTorch | -- | -- | -- | -- | -- |
| LSTM | PyTorch + CUDA | -- | -- | -- | -- | -- |
| 1D-CNN | PyTorch + CUDA | -- | -- | -- | -- | -- |

### Clustering Models

| Model | Framework | Method | Silhouette Score | Davies-Bouldin |
|-------|-----------|--------|-----------------|----------------|
| K-Means | scikit-learn | Best k by silhouette | -- | -- |
| Hierarchical | scikit-learn | Ward linkage | -- | -- |
| DBSCAN | scikit-learn | Best eps by silhouette | -- | -- |
| Autoencoder + K-Means | PyTorch | Cluster on 32-dim latent space | -- | -- |

> *All metric tables will be updated with actual results after training.*

---

## 📊 Feature Engineering (30+ Features)

| Category | Features | Count |
|----------|----------|-------|
| **Price returns** | Daily returns, log returns, rolling mean returns (5/10/20/60-day), rolling volatility (5/10/20/60-day), price momentum (10/20/60-day), lagged returns (1/2/5-day) | 15 |
| **Technical indicators** | RSI (14), MACD + signal + histogram, Bollinger Bands (upper, lower, bandwidth, %B), SMA (20/50/200), EMA (12/26), ATR (14), OBV, Stochastic (%K, %D) | 17 |
| **Risk metrics** | Beta (vs S&P 500), rolling Sharpe (60-day), rolling Sortino (60-day), rolling max drawdown (60-day), VaR 95% (60-day) | 5 |
| **Volume signals** | Volume MA ratio (current/20-day avg), volume rate of change | 2 |

---

## 🎯 Key Design Decisions & Tradeoffs

**Time-based splitting instead of random split:**
> Financial data is time-ordered. Random shuffling leaks future information into training -- your model sees 2024 data while learning to predict 2022. We use strict cutoffs: train < 2022, validate = 2022-2023H1, test = 2023H2+. This is how every serious quant shop evaluates models.

**Both regression AND classification on the same data:**
> Regression predicts *how much* a stock will return. Classification predicts *whether to act* (buy/hold/sell). Combining both gives a two-stage filter: classification removes bad stocks, regression ranks the remaining ones. This mirrors how production trading systems work.

**12+ models instead of "just pick XGBoost":**
> No single model dominates all datasets. Tree-based models handle tabular features well but miss sequential patterns. LSTMs capture time dependencies but need more data. CNNs detect local patterns in price windows. **Comparing all of them on the same data with the same metrics is the point** -- it demonstrates ML maturity, not just tool knowledge.

**1D-CNN for financial data (unconventional choice):**
> CNNs aren't just for images. Conv1D layers detect local patterns in price windows -- candlestick formations, momentum shifts, volatility clusters. Combined with LSTM in the CNN-LSTM hybrid, you get local pattern detection + sequential memory.

**Ensemble over individual models:**
> Stacking uses predictions from top 3 models as meta-features for a Ridge meta-learner. This captures what each model does well -- tree models handle non-linear feature interactions, LSTMs handle time dependencies, CNNs handle local patterns. The ensemble combines strengths.

**Max 5% weight per stock in portfolio:**
> Unconstrained mean-variance optimization often puts 90%+ into 2-3 stocks. The 5% cap forces diversification -- no single stock can blow up the portfolio. This is a standard constraint in institutional portfolio management.

**Walk-forward backtest with monthly rebalancing:**
> Point-in-time evaluation. At each month, we only use data available up to that date to make predictions. No peeking. The portfolio is rebalanced monthly using fresh predictions -- this simulates how a real fund would operate.

---

## 📈 Portfolio Backtest Results

> *Results will be updated after walk-forward backtesting.*

| Metric | ML Portfolio | S&P 500 Benchmark |
|--------|-------------|-------------------|
| Total Return | -- | -- |
| Annualized Return | -- | -- |
| Annualized Volatility | -- | -- |
| Sharpe Ratio | -- | -- |
| Sortino Ratio | -- | -- |
| Maximum Drawdown | -- | -- |
| Win Rate (monthly) | -- | -- |
| Calmar Ratio | -- | -- |

---

## 📊 MLflow Experiment Tracking

Every model training run is logged to MLflow:

- **Parameters**: model type, all hyperparameters, feature set, date ranges, window size
- **Metrics**: all regression/classification metrics, training time, inference time
- **Artifacts**: trained models (.pkl/.pt), SHAP plots, confusion matrices, loss curves, equity curves

```bash
# browse all experiments locally
mlflow ui --port 5000
```

Experiments are organized as:
- `return-regression` -- all regression model runs
- `signal-classification` -- all classification model runs
- `clustering` -- clustering comparison runs
- `portfolio-backtest` -- final portfolio evaluation

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/BhavyaLikhitha/Portfolio-Optimization-and-Market-Performance-Analysis-using-ML.git
cd Portfolio-Optimization-and-Market-Performance-Analysis-using-ML
pip install -r requirements.txt

# 2. Run full pipeline (data → features → models → backtest)
python main.py

# 3. View MLflow experiment results
mlflow ui --port 5000
# Open http://localhost:5000

# 4. Launch interactive dashboard
streamlit run app.py
# Open http://localhost:8501
```

---

## 📁 Project Structure

```
data/
├── ingestion.py                 # S&P 500 OHLCV data via yfinance
├── feature_engineering.py       # 30+ technical/risk/volume features
├── labeling.py                  # Regression targets + classification labels
└── data_loader.py               # Time-based splits, PyTorch DataLoaders

models/
├── classical/
│   ├── regression.py            # Linear, Ridge, Lasso, RF, XGBoost, LightGBM, SVR
│   └── classification.py       # Logistic, RF, XGBoost, LightGBM, SVM
├── deep_learning/
│   ├── mlp.py                   # Multi-layer perceptron (PyTorch)
│   ├── lstm.py                  # 2-layer LSTM (PyTorch + CUDA)
│   ├── cnn.py                   # 1D-CNN (PyTorch + CUDA)
│   ├── cnn_lstm.py              # CNN-LSTM hybrid (PyTorch + CUDA)
│   └── autoencoder.py           # Feature learning + deep clustering
├── clustering/
│   ├── traditional.py           # K-Means, Hierarchical, DBSCAN
│   └── deep_clustering.py       # Autoencoder + K-Means
└── ensemble.py                  # Stacking, voting, weighted blending

evaluation/
├── metrics.py                   # Regression + classification + portfolio metrics
├── backtester.py                # Walk-forward backtest with monthly rebalancing
└── visualization.py             # All plots: model comparison, SHAP, equity curves

portfolio/
├── optimizer.py                 # Mean-variance optimization (max Sharpe, ≤5%/stock)
└── signals.py                   # Combine model predictions into trading signals

tracking/
└── mlflow_tracking.py           # MLflow logging helpers, experiment setup

config.py                        # All hyperparameters, paths, constants, device setup
main.py                          # Full pipeline orchestration
app.py                           # Streamlit dashboard
requirements.txt
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[PROJECT_PLAN.md](PROJECT_PLAN.md)** | Full implementation plan with model architectures, training details, and phase breakdown |
| **[plan.md](plan.md)** | Step-by-step execution plan -- every function, every dependency, build order |

---

## 📦 Data Source

**S&P 500 Stock Data** via [Yahoo Finance API (yfinance)](https://pypi.org/project/yfinance/)
- ~500 stocks with daily OHLCV data (2015-present)
- S&P 500 index (^GSPC) as benchmark
- Free, publicly available, no API key required

---

*Built to demonstrate production-quality ML engineering across classical ML, deep learning (PyTorch + CUDA), experiment tracking (MLflow), model explainability (SHAP), and proper time-series evaluation -- on real financial data.*
