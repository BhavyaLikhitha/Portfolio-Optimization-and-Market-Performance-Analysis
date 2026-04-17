# Portfolio Optimization & Market Performance Analysis using ML/DL

## Revamp Plan & Implementation Guide

---

## Objective

Predict stock returns (regression) and classify stocks into buy/hold/sell signals (classification) using classical ML and deep learning models, then construct an optimized portfolio benchmarked against the S&P 500.

This project demonstrates the full ML lifecycle: data collection, feature engineering, model training (10+ models), evaluation, ensembling, and deployment via a dashboard.

---

## What This Project Demonstrates (Resume Value)

| Skill Area | What's Covered |
|-----------|---------------|
| Problem Framing | Same dataset used for regression, classification, clustering, and portfolio construction |
| Feature Engineering | 30+ technical/financial indicators, time-series feature design, leakage-free labeling |
| Classical ML | Linear/Ridge/Lasso, Random Forest, XGBoost, LightGBM, SVM/SVR, Logistic Regression |
| Deep Learning | LSTM, 1D-CNN, CNN-LSTM hybrid, MLP, Autoencoder (all in PyTorch with CUDA support) |
| Model Evaluation | MSE, MAE, R², directional accuracy, precision, recall, F1, ROC-AUC, confusion matrices |
| Explainability | SHAP values, feature importance rankings |
| Ensembling | Stacking ensemble combining best models |
| Backtesting | Walk-forward validation, cumulative returns, drawdown analysis |
| Deployment | Streamlit dashboard with interactive model comparison |

### Target Roles

- Data Scientist
- ML Engineer
- Quantitative Analyst / Quant Developer
- Applied Scientist
- Research Engineer
- Data Analyst (with ML focus)

---

## Tech Stack

| Component | Tools |
|-----------|-------|
| Data | yfinance, pandas, numpy |
| Classical ML | scikit-learn, xgboost, lightgbm |
| Deep Learning | PyTorch, CUDA |
| Explainability | SHAP |
| Visualization | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |
| Experiment Tracking | MLflow |

---

## Phase 1: Data Collection & Feature Engineering

### 1.1 Data Collection

- Fetch S&P 500 ticker list from Wikipedia
- Download daily OHLCV (Open, High, Low, Close, Volume) data via yfinance
- Date range: 2015-01-01 to present (enough history for training + backtesting)
- Clean tickers (BRK.B -> BRK-B), drop stocks with insufficient history

### 1.2 Feature Engineering (30+ features per stock)

**Price-Based Features:**
- Daily returns, log returns
- Rolling mean returns (5, 10, 20, 60 day windows)
- Rolling volatility (5, 10, 20, 60 day windows)
- Price momentum (rate of change over 10, 20, 60 days)
- Lagged returns (1-day, 2-day, 5-day lag)

**Technical Indicators:**
- RSI (Relative Strength Index, 14-day)
- MACD (Moving Average Convergence Divergence) + signal line
- Bollinger Bands (upper, lower, bandwidth, %B)
- Moving Averages (SMA 20, SMA 50, SMA 200, EMA 12, EMA 26)
- ATR (Average True Range, 14-day)
- OBV (On-Balance Volume)
- Stochastic Oscillator (%K, %D)

**Risk Metrics:**
- Beta (vs S&P 500 equal-weight proxy)
- Sharpe ratio (rolling 60-day)
- Sortino ratio (rolling 60-day)
- Max drawdown (rolling 60-day)
- Value-at-Risk 95% (rolling 60-day)

**Volume Features:**
- Volume moving average ratio (current volume / 20-day avg)
- Volume rate of change

### 1.3 Label Creation

**Regression targets:**
- Forward 5-day return
- Forward 10-day return
- Forward 30-day return

**Classification targets (based on forward 10-day return):**
- Buy: return > +2%
- Hold: return between -2% and +2%
- Sell: return < -2%

### 1.4 Data Splitting (Time-Based, No Leakage)

- Training: 2015-01-01 to 2021-12-31
- Validation: 2022-01-01 to 2023-06-30
- Test: 2023-07-01 to present

No random shuffling. Strict temporal ordering to prevent look-ahead bias.

---

## Phase 2: Classical ML Models

### 2.1 Regression Models

Train each model to predict forward 10-day returns:

| Model | Key Hyperparameters to Tune |
|-------|---------------------------|
| Linear Regression | baseline, no tuning |
| Ridge Regression | alpha |
| Lasso Regression | alpha |
| Random Forest Regressor | n_estimators, max_depth, min_samples_split |
| XGBoost Regressor | learning_rate, max_depth, n_estimators, subsample |
| LightGBM Regressor | learning_rate, num_leaves, n_estimators |
| SVR (RBF kernel) | C, gamma, epsilon |

**Evaluation metrics:**
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- Directional Accuracy (did we predict the sign correctly?)

### 2.2 Classification Models

Train each model to predict buy/hold/sell:

| Model | Key Hyperparameters to Tune |
|-------|---------------------------|
| Logistic Regression | C, penalty |
| Random Forest Classifier | n_estimators, max_depth |
| XGBoost Classifier | learning_rate, max_depth, n_estimators |
| LightGBM Classifier | learning_rate, num_leaves |
| SVM (RBF kernel) | C, gamma |

**Evaluation metrics:**
- Accuracy
- Precision, Recall, F1-score (per class and macro)
- Confusion Matrix
- ROC-AUC (one-vs-rest for multiclass)
- Classification Report

### 2.3 Feature Importance & Explainability

- SHAP values for XGBoost/LightGBM (both regression and classification)
- Feature importance bar plots from tree-based models
- Identify which technical indicators matter most

---

## Phase 3: Deep Learning Models (PyTorch + CUDA)

### 3.1 Data Preparation for DL

- Create sliding window sequences: 60 trading days of features as input
- Shape: (batch_size, sequence_length=60, num_features=30+)
- Normalize per-window using training set statistics
- Use DataLoader with batching

### 3.2 Models to Build

**LSTM (Long Short-Term Memory):**
- Architecture: 2-layer LSTM (128 units) -> dropout -> fully connected -> output
- For regression: single output (predicted return)
- For classification: 3 outputs (buy/hold/sell probabilities via softmax)

**1D-CNN (Convolutional Neural Network):**
- Architecture: Conv1D (filters: 64, 128) -> BatchNorm -> ReLU -> MaxPool -> flatten -> FC -> output
- Captures local patterns in price windows (chart pattern detection)

**CNN-LSTM Hybrid:**
- Architecture: Conv1D layers -> LSTM layers -> FC -> output
- CNN extracts local features, LSTM models temporal dependencies
- Best of both approaches

**MLP (Multi-Layer Perceptron):**
- Architecture: FC(256) -> ReLU -> Dropout -> FC(128) -> ReLU -> Dropout -> FC(64) -> output
- Baseline deep learning model (no sequential structure)
- Input: flattened feature vector (not windowed)

**Autoencoder (for learned feature representations):**
- Architecture: Encoder(input -> 128 -> 64 -> 32) -> Decoder(32 -> 64 -> 128 -> input)
- Train unsupervised on stock features
- Extract 32-dim latent representations
- Use latent features for clustering (Autoencoder + K-Means)

### 3.3 Training Setup

- Optimizer: Adam (lr=1e-3 with ReduceLROnPlateau scheduler)
- Loss: MSE for regression, CrossEntropy for classification
- Early stopping: patience=10 on validation loss
- CUDA acceleration when GPU available, CPU fallback
- Batch size: 64
- Epochs: up to 100 (with early stopping)
- Dropout: 0.3 for regularization

### 3.4 DL Evaluation

Same metrics as Phase 2, plus:
- Training/validation loss curves
- Overfitting analysis (train vs val gap)
- Inference time comparison across models

---

## Phase 4: Clustering & Stock Grouping

### 4.1 Traditional Clustering (keep from current project)

- K-Means (k=5, 10, 15 -- compare with silhouette score)
- Hierarchical Clustering (ward linkage)
- DBSCAN (tune eps with k-distance plot)

### 4.2 Deep Clustering

- Autoencoder + K-Means: cluster on 32-dim learned representations instead of hand-crafted features
- Compare cluster quality (silhouette score, Davies-Bouldin index) vs traditional

### 4.3 Visualization

- t-SNE / UMAP plots colored by cluster
- Cluster-wise return distribution (box plots)
- Sector composition per cluster

---

## Phase 5: Ensemble & Portfolio Construction

### 5.1 Model Ensembling

- Stacking: use predictions from top 3 regression models as features for a meta-learner
- Voting classifier: majority vote from top 3 classifiers for buy/hold/sell
- Weighted average: blend regression predictions using validation performance

### 5.2 Portfolio Construction

- Use ensemble regression predictions to rank stocks by expected return
- Use ensemble classification to filter (keep only "buy" signals)
- Select top 50 stocks that pass both filters
- Optimize weights via mean-variance optimization (maximize predicted Sharpe)
- Constraint: max 5% weight per stock (diversification)

### 5.3 Backtesting

- Walk-forward backtest on test period (2023-07-01 to present)
- Rebalance monthly using updated model predictions
- Track: cumulative returns, daily returns, portfolio value over time

**Backtest Metrics:**
- Total return vs S&P 500
- Annualized return
- Annualized volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Win rate (% of months with positive return)
- Calmar ratio

---

## Phase 6: Visualization & Dashboard

### 6.1 Model Comparison Plots

- Bar chart: R² / MAE / MSE across all regression models
- Bar chart: F1 / accuracy / AUC across all classification models
- Table: all models ranked by primary metric
- Training time comparison

### 6.2 DL-Specific Plots

- Loss curves (train vs validation) for each DL model
- Predicted vs actual scatter plots
- Residual distribution plots

### 6.3 Explainability Plots

- SHAP summary plots (beeswarm)
- SHAP feature importance bar plots
- Per-stock SHAP waterfall plots

### 6.4 Portfolio Plots

- Cumulative returns: portfolio vs S&P 500
- Drawdown chart over time
- Monthly returns heatmap
- Sector allocation pie chart
- Portfolio weight bar chart

### 6.5 Streamlit Dashboard

Interactive dashboard with:
- Date range selector
- Model selector (compare any two models)
- Live prediction display
- Portfolio allocation viewer
- Backtest results with metrics table

---

## Project Structure

```
data/
    ingestion.py                 # fetch S&P 500 data via yfinance
    feature_engineering.py       # compute 30+ features
    labeling.py                  # create regression + classification targets
    data_loader.py               # PyTorch DataLoader, sliding windows, train/val/test split

models/
    classical/
        regression.py            # Linear, Ridge, Lasso, RF, XGBoost, LightGBM, SVR
        classification.py        # Logistic, RF, XGBoost, LightGBM, SVM
    deep_learning/
        lstm.py                  # LSTM regressor + classifier (PyTorch)
        cnn.py                   # 1D-CNN regressor + classifier (PyTorch)
        cnn_lstm.py              # CNN-LSTM hybrid (PyTorch)
        mlp.py                   # MLP baseline (PyTorch)
        autoencoder.py           # Autoencoder for feature learning (PyTorch)
    clustering/
        traditional.py           # K-Means, Hierarchical, DBSCAN
        deep_clustering.py       # Autoencoder + K-Means
    ensemble.py                  # Stacking, voting, weighted blending

evaluation/
    metrics.py                   # all regression + classification metrics
    backtester.py                # walk-forward portfolio backtest
    visualization.py             # all plots (model comparison, SHAP, portfolio)

portfolio/
    optimizer.py                 # mean-variance optimization with constraints
    signals.py                   # combine model outputs into buy/hold/sell

config.py                        # all hyperparameters, paths, constants
tracking/
    mlflow_tracking.py           # MLflow logging helpers, experiment setup
    mlruns/                      # MLflow run data (gitignored)
main.py                          # orchestrate full pipeline
app.py                           # Streamlit dashboard
requirements.txt                 # all dependencies
```

---

## Implementation Order

| Order | Task | Estimated Complexity |
|-------|------|---------------------|
| 1 | `data/ingestion.py` -- fetch and clean S&P 500 data | Low |
| 2 | `data/feature_engineering.py` -- compute 30+ features | Medium |
| 3 | `data/labeling.py` -- create regression + classification targets | Low |
| 4 | `data/data_loader.py` -- train/val/test split, PyTorch DataLoader | Medium |
| 5 | `models/classical/regression.py` -- all regression models + evaluation | Medium |
| 6 | `models/classical/classification.py` -- all classification models + evaluation | Medium |
| 7 | `evaluation/metrics.py` -- metric computation functions | Low |
| 8 | `evaluation/visualization.py` -- SHAP, model comparison plots | Medium |
| 9 | `models/deep_learning/mlp.py` -- MLP baseline | Medium |
| 10 | `models/deep_learning/lstm.py` -- LSTM model | High |
| 11 | `models/deep_learning/cnn.py` -- 1D-CNN model | High |
| 12 | `models/deep_learning/cnn_lstm.py` -- hybrid model | High |
| 13 | `models/deep_learning/autoencoder.py` -- autoencoder + deep clustering | Medium |
| 14 | `models/clustering/traditional.py` -- K-Means, Hierarchical, DBSCAN | Low |
| 15 | `models/ensemble.py` -- stacking and voting | Medium |
| 16 | `portfolio/signals.py` -- generate trading signals from models | Medium |
| 17 | `portfolio/optimizer.py` -- portfolio weight optimization | Medium |
| 18 | `evaluation/backtester.py` -- walk-forward backtest | High |
| 19 | `main.py` -- full pipeline orchestration | Medium |
| 20 | `app.py` -- Streamlit dashboard | Medium |

---

## MLflow Experiment Tracking

### What Gets Tracked

Every model training run logs the following to MLflow:

**Parameters:**
- Model type, hyperparameters (learning rate, n_estimators, max_depth, etc.)
- Feature set version, window size (for DL models)
- Train/val/test date ranges

**Metrics:**
- Regression: MSE, MAE, R², directional accuracy
- Classification: accuracy, precision, recall, F1, ROC-AUC
- Training time, inference time
- Portfolio backtest metrics (Sharpe, return, drawdown)

**Artifacts:**
- Trained model files (.pkl for sklearn, .pt for PyTorch)
- SHAP plots, confusion matrices, loss curves
- Feature importance rankings
- Backtest equity curves

### MLflow Organization

- Experiment per task: `return-regression`, `signal-classification`, `clustering`, `portfolio-backtest`
- Each model type is a separate run within the experiment
- Best model tagged for production use
- Model registry for versioning the ensemble

### Usage

```bash
# start MLflow UI
mlflow ui --port 5000

# runs are logged automatically during training
python main.py
```

### Integration Points

- `models/classical/regression.py` -- log sklearn model params + metrics
- `models/classical/classification.py` -- log sklearn model params + metrics
- `models/deep_learning/*.py` -- log PyTorch training curves, params, model checkpoints
- `models/ensemble.py` -- log ensemble config + final metrics
- `evaluation/backtester.py` -- log portfolio performance metrics

---

## Key Principles

- **No data leakage**: strict time-based splits, no future information in features
- **Compare everything**: every model gets the same data, same metrics, side-by-side comparison
- **CUDA when available**: all PyTorch models auto-detect GPU, fallback to CPU
- **Reproducibility**: set random seeds everywhere, log all hyperparameters
- **Keep it real**: use real S&P 500 data, realistic transaction assumptions, proper backtesting
