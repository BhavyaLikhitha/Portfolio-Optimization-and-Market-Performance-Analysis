# Execution Plan

Step-by-step implementation plan to revamp the project from start to finish.

---

## Step 0: Cleanup & Scaffolding

### 0.1 Remove old files

Delete the following (replaced by new modular structure):

- `data_ingestion.py`
- `feature_engineering.py`
- `clustering.py`
- `portfolio_optimization.py`
- `sentiment_analysis.py` (hardcoded API key, VADER-based, not reusable)
- `modular_code_top100.py` (monolithic duplicate of entire pipeline)
- `logger.py` (replace with Python logging in config.py)
- `testing.ipynb`
- `final_Testing.ipynb`
- `models/base_model.ipynb`
- `models/multi_cluster_selection.ipynb`
- `models/top100.ipynb`
- `csv_files/` directory
- `output.png`
- `app.py` (will be rewritten from scratch)
- `main.py` (will be rewritten from scratch)

### 0.2 Create directory structure

```
mkdir data models models/classical models/deep_learning models/clustering evaluation portfolio tracking
```

### 0.3 Create config.py

Central configuration file containing:

- `TICKERS_URL` -- Wikipedia S&P 500 URL
- `START_DATE`, `TRAIN_END`, `VAL_END` -- time split boundaries
- `SEQUENCE_LENGTH = 60` -- sliding window for DL models
- `BATCH_SIZE = 64`
- `EPOCHS = 100`
- `LEARNING_RATE = 1e-3`
- `DROPOUT = 0.3`
- `EARLY_STOPPING_PATIENCE = 10`
- `RANDOM_SEED = 42`
- `TOP_N_STOCKS = 50`
- `MAX_WEIGHT_PER_STOCK = 0.05`
- `BUY_THRESHOLD = 0.02`, `SELL_THRESHOLD = -0.02`
- `FORWARD_DAYS = 10` -- prediction horizon
- `DEVICE` -- auto-detect CUDA or CPU
- `MLFLOW_EXPERIMENT_NAMES` -- dict of experiment names
- `FEATURE_COLUMNS` -- list of all 30+ feature names
- Logging setup (replace logger.py)

### 0.4 Update requirements.txt

```
pandas
numpy
yfinance
matplotlib
seaborn
plotly
scikit-learn
scipy
xgboost
lightgbm
torch
shap
mlflow
streamlit
```

### 0.5 Update .gitignore

Add:

```
mlruns/
mlflow.db
*.pt
*.pkl
__pycache__/
portfolio.log
```

---

## Step 1: data/ingestion.py

### Functions to implement:

**`fetch_sp500_tickers()`**
- Scrape Wikipedia for S&P 500 list
- Return DataFrame with Symbol and GICS Sector columns
- Clean tickers (BRK.B -> BRK-B)

**`fetch_stock_data(tickers, start_date, end_date)`**
- Download OHLCV data via `yf.download()`
- Drop tickers with >20% missing data
- Forward-fill remaining gaps
- Return DataFrame with MultiIndex columns (ticker, OHLCV)

**`get_sp500_index(start_date, end_date)`**
- Download ^GSPC (actual S&P 500 index) for benchmarking
- Return Series of adjusted close prices

### Output:
- Raw OHLCV DataFrame for all valid S&P 500 stocks
- S&P 500 index prices for benchmark comparison

---

## Step 2: data/feature_engineering.py

### Functions to implement:

**`compute_returns(close_prices)`**
- Daily returns, log returns
- Rolling mean returns (5, 10, 20, 60 day)
- Rolling volatility (5, 10, 20, 60 day)
- Price momentum / rate of change (10, 20, 60 day)
- Lagged returns (1, 2, 5 day)

**`compute_technical_indicators(ohlcv)`**
- RSI (14-day)
- MACD (12, 26, 9) + signal line + histogram
- Bollinger Bands (20-day, 2 std): upper, lower, bandwidth, %B
- SMA (20, 50, 200)
- EMA (12, 26)
- ATR (14-day)
- OBV (On-Balance Volume)
- Stochastic Oscillator (%K, %D)

**`compute_risk_metrics(returns, market_returns)`**
- Beta vs S&P 500
- Rolling Sharpe ratio (60-day)
- Rolling Sortino ratio (60-day)
- Rolling max drawdown (60-day)
- VaR 95% (rolling 60-day)

**`compute_volume_features(volume)`**
- Volume MA ratio (current / 20-day avg)
- Volume rate of change

**`build_feature_matrix(ohlcv_data, market_returns)`**
- Calls all above functions
- Combines into single DataFrame per stock
- Handles inf/NaN (replace inf -> NaN, drop rows with NaN)
- Standardize with StandardScaler (fit on training data only)
- Return: features DataFrame, fitted scaler object

### Output:
- DataFrame: rows = (date, ticker), columns = 30+ features
- Fitted StandardScaler saved for inference

---

## Step 3: data/labeling.py

### Functions to implement:

**`create_regression_labels(close_prices, forward_days=[5, 10, 30])`**
- For each stock, compute forward N-day return: `(price[t+N] / price[t]) - 1`
- Return DataFrame with columns: `fwd_return_5d`, `fwd_return_10d`, `fwd_return_30d`

**`create_classification_labels(close_prices, forward_days=10, buy_thresh=0.02, sell_thresh=-0.02)`**
- Compute forward 10-day return
- Label: 0 = sell (< -2%), 1 = hold (-2% to +2%), 2 = buy (> +2%)
- Return Series of integer labels

**`merge_features_and_labels(features, reg_labels, clf_labels)`**
- Inner join on (date, ticker)
- Drop any rows where labels are NaN (near end of dataset)
- Return combined DataFrame

### Output:
- Combined DataFrame: features + regression targets + classification target

---

## Step 4: data/data_loader.py

### Functions to implement:

**`split_by_time(df, train_end, val_end)`**
- Training: everything before train_end (2021-12-31)
- Validation: train_end to val_end (2022-01-01 to 2023-06-30)
- Test: after val_end (2023-07-01 onwards)
- Return: train_df, val_df, test_df

**`prepare_tabular_data(train_df, val_df, test_df, target_col)`**
- Separate features (X) and target (y)
- Scale features using training set statistics only
- Return: X_train, y_train, X_val, y_val, X_test, y_test as numpy arrays

**`class StockSequenceDataset(torch.utils.data.Dataset)`**
- Takes DataFrame, target column, sequence_length
- `__getitem__` returns (sequence of shape [seq_len, num_features], target)
- Sequences are per-stock: 60 consecutive days of features for one stock

**`create_dataloaders(train_df, val_df, test_df, target_col, seq_len=60, batch_size=64)`**
- Create StockSequenceDataset for each split
- Wrap in DataLoader with shuffling (train only)
- Return: train_loader, val_loader, test_loader

### Output:
- Tabular data (numpy) for classical ML
- DataLoaders for PyTorch models

---

## Step 5: tracking/mlflow_tracking.py

### Functions to implement:

**`setup_experiment(experiment_name)`**
- Set MLflow tracking URI (local ./mlruns)
- Create or get experiment by name
- Return experiment_id

**`log_sklearn_model(model, model_name, params, metrics, experiment_name)`**
- Start MLflow run
- Log all params (hyperparameters)
- Log all metrics (MSE, MAE, R², etc.)
- Log model artifact (.pkl)
- End run
- Return run_id

**`log_pytorch_model(model, model_name, params, metrics, artifacts, experiment_name)`**
- Start MLflow run
- Log params, metrics
- Log model state_dict (.pt)
- Log any artifact files (loss curves, confusion matrices as images)
- End run
- Return run_id

**`log_backtest_results(metrics, equity_curve_path, experiment_name)`**
- Log portfolio backtest metrics
- Log equity curve plot as artifact

**`get_best_run(experiment_name, metric, maximize=True)`**
- Query MLflow for best run by metric
- Return run_id, params, metrics

---

## Step 6: evaluation/metrics.py

### Functions to implement:

**`regression_metrics(y_true, y_pred)`**
- MSE, MAE, RMSE, R²
- Directional accuracy (% where sign of prediction matches sign of actual)
- Return dict of all metrics

**`classification_metrics(y_true, y_pred, y_prob=None)`**
- Accuracy, precision (macro), recall (macro), F1 (macro)
- Per-class precision, recall, F1
- Confusion matrix
- ROC-AUC (one-vs-rest, if y_prob provided)
- Return dict of all metrics + confusion matrix array

**`portfolio_metrics(portfolio_returns, benchmark_returns)`**
- Total return, annualized return
- Annualized volatility
- Sharpe ratio, Sortino ratio
- Maximum drawdown
- Win rate (% positive days/months)
- Calmar ratio
- Return dict of all metrics

---

## Step 7: models/classical/regression.py

### Functions to implement:

**`train_all_regressors(X_train, y_train, X_val, y_val)`**

Train each of these models:

1. **LinearRegression** -- baseline, no tuning
2. **Ridge** -- tune alpha via validation set: [0.01, 0.1, 1.0, 10.0]
3. **Lasso** -- tune alpha: [0.001, 0.01, 0.1, 1.0]
4. **RandomForestRegressor** -- tune: n_estimators=[100,200], max_depth=[10,20,None]
5. **XGBRegressor** -- tune: lr=[0.01,0.1], max_depth=[3,6], n_estimators=[100,200]
6. **LGBMRegressor** -- tune: lr=[0.01,0.1], num_leaves=[31,63], n_estimators=[100,200]
7. **SVR(kernel='rbf')** -- tune: C=[0.1,1,10], gamma=['scale','auto']

For each model:
- Train on training set
- Evaluate on validation set using `regression_metrics()`
- Log to MLflow via `log_sklearn_model()`
- Store trained model

Return: dict of {model_name: (model, val_metrics)}

**`predict_regression(model, X)`**
- Return predictions array

**`get_best_regressor(results_dict)`**
- Compare validation R² across all models
- Return best model name and object

---

## Step 8: models/classical/classification.py

### Functions to implement:

**`train_all_classifiers(X_train, y_train, X_val, y_val)`**

Train each of these models:

1. **LogisticRegression(multi_class='multinomial')** -- tune C: [0.01, 0.1, 1.0, 10.0]
2. **RandomForestClassifier** -- tune: n_estimators=[100,200], max_depth=[10,20,None]
3. **XGBClassifier** -- tune: lr=[0.01,0.1], max_depth=[3,6], n_estimators=[100,200]
4. **LGBMClassifier** -- tune: lr=[0.01,0.1], num_leaves=[31,63], n_estimators=[100,200]
5. **SVC(kernel='rbf', probability=True)** -- tune: C=[0.1,1,10], gamma=['scale','auto']

For each model:
- Train, evaluate on validation set using `classification_metrics()`
- Log to MLflow
- Store trained model

Return: dict of {model_name: (model, val_metrics)}

**`predict_classification(model, X)`**
- Return predicted classes and probabilities

**`get_best_classifier(results_dict)`**
- Compare validation F1 (macro) across all models
- Return best model name and object

---

## Step 9: models/deep_learning/mlp.py

### Architecture:

```
Input(num_features)
  -> Linear(256) -> BatchNorm -> ReLU -> Dropout(0.3)
  -> Linear(128) -> BatchNorm -> ReLU -> Dropout(0.3)
  -> Linear(64) -> ReLU
  -> Linear(output_size)  # 1 for regression, 3 for classification
```

### Functions to implement:

**`class MLPModel(nn.Module)`**
- `__init__(input_size, hidden_sizes=[256,128,64], output_size=1, dropout=0.3)`
- `forward(x)` -- flatten input, pass through layers

**`train_mlp(train_loader, val_loader, task='regression', epochs=100, lr=1e-3)`**
- Task='regression': MSE loss, single output
- Task='classification': CrossEntropy loss, 3 outputs + softmax
- Adam optimizer + ReduceLROnPlateau
- Early stopping on validation loss (patience=10)
- Log training/validation loss per epoch
- CUDA if available
- Log to MLflow
- Return: trained model, training history

**`evaluate_mlp(model, test_loader, task)`**
- Run inference
- Compute regression_metrics or classification_metrics
- Return metrics dict

---

## Step 10: models/deep_learning/lstm.py

### Architecture:

```
Input(batch, seq_len=60, num_features)
  -> LSTM(input_size=num_features, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
  -> take last hidden state
  -> Linear(128) -> ReLU -> Dropout(0.3)
  -> Linear(output_size)
```

### Functions to implement:

**`class LSTMModel(nn.Module)`**
- `__init__(input_size, hidden_size=128, num_layers=2, output_size=1, dropout=0.3)`
- `forward(x)` -- pass through LSTM, use last timestep output, through FC layers

**`train_lstm(train_loader, val_loader, task='regression', epochs=100, lr=1e-3)`**
- Same training loop pattern as MLP
- Gradient clipping (max_norm=1.0) to prevent exploding gradients
- Log to MLflow
- Return: trained model, training history

**`evaluate_lstm(model, test_loader, task)`**
- Return metrics dict

---

## Step 11: models/deep_learning/cnn.py

### Architecture:

```
Input(batch, seq_len=60, num_features)
  -> permute to (batch, num_features, seq_len)  # channels-first for Conv1D
  -> Conv1d(in=num_features, out=64, kernel=3, padding=1) -> BatchNorm1d -> ReLU -> MaxPool1d(2)
  -> Conv1d(64, 128, kernel=3, padding=1) -> BatchNorm1d -> ReLU -> MaxPool1d(2)
  -> Conv1d(128, 64, kernel=3, padding=1) -> BatchNorm1d -> ReLU
  -> AdaptiveAvgPool1d(1) -> flatten
  -> Linear(64) -> ReLU -> Dropout(0.3)
  -> Linear(output_size)
```

### Functions to implement:

**`class CNN1DModel(nn.Module)`**
- `__init__(input_size, output_size=1, dropout=0.3)`
- `forward(x)` -- permute, conv layers, pool, FC

**`train_cnn(train_loader, val_loader, task='regression', epochs=100, lr=1e-3)`**
- Same training loop pattern
- Log to MLflow
- Return: trained model, training history

**`evaluate_cnn(model, test_loader, task)`**
- Return metrics dict

---

## Step 12: models/deep_learning/cnn_lstm.py

### Architecture:

```
Input(batch, seq_len=60, num_features)
  -> permute to (batch, num_features, seq_len)
  -> Conv1d(num_features, 64, kernel=3, padding=1) -> BatchNorm1d -> ReLU -> MaxPool1d(2)
  -> Conv1d(64, 128, kernel=3, padding=1) -> BatchNorm1d -> ReLU
  -> permute back to (batch, reduced_seq, 128)
  -> LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
  -> take last hidden state
  -> Linear(64) -> ReLU -> Dropout(0.3)
  -> Linear(output_size)
```

### Functions to implement:

**`class CNNLSTMModel(nn.Module)`**
- `__init__(input_size, output_size=1, dropout=0.3)`
- `forward(x)` -- CNN feature extraction, then LSTM sequence modeling, then FC

**`train_cnn_lstm(train_loader, val_loader, task='regression', epochs=100, lr=1e-3)`**
- Same training loop pattern
- Log to MLflow
- Return: trained model, training history

**`evaluate_cnn_lstm(model, test_loader, task)`**
- Return metrics dict

---

## Step 13: models/deep_learning/autoencoder.py

### Architecture:

```
Encoder:
  Input(num_features)
  -> Linear(128) -> ReLU
  -> Linear(64) -> ReLU
  -> Linear(32)  # latent space

Decoder:
  Linear(32) -> ReLU
  -> Linear(64) -> ReLU
  -> Linear(128) -> ReLU
  -> Linear(num_features)
```

### Functions to implement:

**`class Autoencoder(nn.Module)`**
- `__init__(input_size, latent_dim=32)`
- `encode(x)` -- return latent representation
- `decode(z)` -- reconstruct
- `forward(x)` -- encode then decode

**`train_autoencoder(train_loader, val_loader, epochs=100, lr=1e-3)`**
- MSE reconstruction loss
- Log to MLflow
- Return: trained model

**`extract_latent_features(model, data_loader)`**
- Run encoder on all data
- Return: numpy array of latent representations (N x 32)

---

## Step 14: models/clustering/traditional.py

### Functions to implement:

**`run_kmeans(features, k_values=[5, 10, 15])`**
- Run K-Means for each k
- Compute silhouette score and Davies-Bouldin index for each
- Pick best k by silhouette score
- Return: labels, best_k, scores

**`run_hierarchical(features, n_clusters=10)`**
- AgglomerativeClustering with ward linkage
- Return: labels

**`run_dbscan(features, eps_values=[0.5, 1.0, 1.5, 2.0], min_samples=5)`**
- Try multiple eps values
- Pick best by silhouette score (excluding noise label -1)
- Return: labels, best_eps

**`evaluate_clusters(features, labels)`**
- Silhouette score
- Davies-Bouldin index
- Number of clusters, cluster sizes
- Return: metrics dict

---

## Step 15: models/clustering/deep_clustering.py

### Functions to implement:

**`deep_cluster(autoencoder_model, features, k_values=[5, 10, 15])`**
- Extract latent features using trained autoencoder
- Run K-Means on latent space
- Evaluate with silhouette score
- Return: labels, latent_features

**`compare_clustering_methods(traditional_labels, deep_labels, features)`**
- Compute metrics for both
- Return comparison DataFrame

---

## Step 16: models/ensemble.py

### Functions to implement:

**`build_stacking_regressor(base_models, X_train, y_train, X_val, y_val)`**
- Use predictions from top 3 regression models as meta-features
- Meta-learner: Ridge regression
- Train meta-learner on validation predictions
- Log to MLflow
- Return: ensemble object (base models + meta-learner)

**`build_voting_classifier(base_models, X_train, y_train, X_val, y_val)`**
- Majority vote from top 3 classifiers
- Evaluate on validation set
- Log to MLflow
- Return: voting classifier

**`build_weighted_blend(models, weights, X)`**
- Weighted average of regression predictions
- Weights proportional to validation R²
- Return: blended predictions

**`predict_ensemble(ensemble, X, task='regression')`**
- Run all base models, combine
- Return: predictions

---

## Step 17: portfolio/signals.py

### Functions to implement:

**`generate_signals(regression_preds, classification_preds, tickers)`**
- Combine regression (predicted return) and classification (buy/hold/sell)
- Filter: keep only stocks where classification = buy (2)
- Rank remaining by predicted return (descending)
- Select top N stocks (from config.TOP_N_STOCKS)
- Return: DataFrame with ticker, predicted_return, signal

---

## Step 18: portfolio/optimizer.py

### Functions to implement:

**`optimize_portfolio(expected_returns, cov_matrix, max_weight=0.05)`**
- Mean-variance optimization via scipy.optimize.minimize
- Objective: maximize Sharpe ratio
- Constraints: weights sum to 1, each weight in [0, max_weight]
- Return: optimal weights array

**`equal_weight_portfolio(n_stocks)`**
- Return: 1/n weights (baseline comparison)

---

## Step 19: evaluation/backtester.py

### Functions to implement:

**`walk_forward_backtest(test_df, regression_model, classification_model, scaler, rebalance_freq='M')`**

Logic:
1. At each rebalance date (monthly):
   - Get features for all stocks up to that date
   - Scale features using fitted scaler
   - Predict returns and signals using models
   - Generate signals (buy/hold/sell)
   - Select top stocks, optimize weights
2. Between rebalance dates:
   - Track daily portfolio value using actual returns and current weights
3. Compute cumulative portfolio returns over entire test period

Return: daily_portfolio_returns (Series), rebalance_log (list of dicts)

**`compute_benchmark_returns(sp500_prices, test_start, test_end)`**
- Compute daily returns for S&P 500 over test period
- Return: Series

**`backtest_report(portfolio_returns, benchmark_returns)`**
- Compute all portfolio_metrics
- Log to MLflow
- Return: metrics dict

---

## Step 20: evaluation/visualization.py

### Functions to implement:

**`plot_model_comparison_regression(results_dict)`**
- Grouped bar chart: MSE, MAE, R² for each regression model
- Save as image

**`plot_model_comparison_classification(results_dict)`**
- Grouped bar chart: accuracy, F1, ROC-AUC for each classifier
- Save as image

**`plot_confusion_matrices(y_true, y_pred, model_name)`**
- Heatmap confusion matrix
- Save as image

**`plot_roc_curves(y_true, y_prob, model_name)`**
- ROC curve per class (one-vs-rest)
- Save as image

**`plot_training_curves(history, model_name)`**
- Train vs validation loss over epochs
- Save as image

**`plot_shap_summary(model, X, model_name)`**
- SHAP beeswarm plot
- SHAP feature importance bar plot
- Save as images

**`plot_equity_curve(portfolio_returns, benchmark_returns)`**
- Cumulative returns line chart: portfolio vs S&P 500
- Save as image

**`plot_drawdown(portfolio_returns)`**
- Drawdown chart over time
- Save as image

**`plot_monthly_returns_heatmap(portfolio_returns)`**
- Month x Year heatmap of returns
- Save as image

**`plot_portfolio_weights(weights, tickers)`**
- Horizontal bar chart of allocations
- Save as image

**`plot_cluster_visualization(features, labels, method='tsne')`**
- t-SNE or UMAP 2D scatter colored by cluster
- Save as image

---

## Step 21: main.py

### Orchestration script:

```python
def main():
    # Step 1: Data
    tickers_df = fetch_sp500_tickers()
    ohlcv = fetch_stock_data(tickers, START_DATE, end_date)
    sp500 = get_sp500_index(START_DATE, end_date)

    # Step 2: Features
    features, scaler = build_feature_matrix(ohlcv, sp500_returns)

    # Step 3: Labels
    reg_labels = create_regression_labels(close_prices)
    clf_labels = create_classification_labels(close_prices)
    full_df = merge_features_and_labels(features, reg_labels, clf_labels)

    # Step 4: Split
    train_df, val_df, test_df = split_by_time(full_df, TRAIN_END, VAL_END)

    # Step 5: Tabular data for classical ML
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_tabular_data(...)

    # Step 6: Classical regression
    reg_results = train_all_regressors(X_train, y_train_reg, X_val, y_val_reg)

    # Step 7: Classical classification
    clf_results = train_all_classifiers(X_train, y_train_clf, X_val, y_val_clf)

    # Step 8: DataLoaders for DL
    train_loader, val_loader, test_loader = create_dataloaders(...)

    # Step 9: DL models (regression + classification)
    mlp_model = train_mlp(...)
    lstm_model = train_lstm(...)
    cnn_model = train_cnn(...)
    cnn_lstm_model = train_cnn_lstm(...)

    # Step 10: Autoencoder + clustering
    ae_model = train_autoencoder(...)
    latent = extract_latent_features(ae_model, ...)
    cluster_labels = deep_cluster(ae_model, ...)

    # Step 11: Traditional clustering
    kmeans_labels = run_kmeans(...)
    hier_labels = run_hierarchical(...)

    # Step 12: Ensemble
    stacking_reg = build_stacking_regressor(...)
    voting_clf = build_voting_classifier(...)

    # Step 13: Backtest
    portfolio_returns = walk_forward_backtest(test_df, stacking_reg, voting_clf, scaler)
    benchmark_returns = compute_benchmark_returns(sp500, VAL_END)
    metrics = backtest_report(portfolio_returns, benchmark_returns)

    # Step 14: Visualizations
    plot_model_comparison_regression(reg_results)
    plot_model_comparison_classification(clf_results)
    plot_equity_curve(portfolio_returns, benchmark_returns)
    plot_shap_summary(best_model, X_test)
    # ... all other plots

    # Step 15: Print summary
    print_final_report(reg_results, clf_results, metrics)
```

---

## Step 22: app.py (Streamlit Dashboard)

### Pages/Sections:

**Sidebar:**
- Date range selector
- Model type selector (regression/classification)
- Individual model selector

**Page 1 -- Model Comparison:**
- Bar charts comparing all regression models (MSE, R², directional accuracy)
- Bar charts comparing all classification models (F1, AUC)
- Table with all metrics side by side

**Page 2 -- Deep Learning Details:**
- Training loss curves for each DL model
- Predicted vs actual scatter plot
- Model architecture summary

**Page 3 -- Explainability:**
- SHAP summary plot for selected model
- Feature importance bar chart
- Per-stock SHAP waterfall (user selects stock)

**Page 4 -- Clustering:**
- t-SNE visualization colored by cluster
- Cluster distribution histogram
- Cluster quality metrics table

**Page 5 -- Portfolio & Backtest:**
- Equity curve (portfolio vs S&P 500)
- Drawdown chart
- Monthly returns heatmap
- Portfolio weights bar chart
- Metrics summary table (Sharpe, return, drawdown, etc.)

---

## Step 23: Final Testing & README Update

### Testing:
- Run full pipeline end to end: `python main.py`
- Verify MLflow UI shows all runs: `mlflow ui --port 5000`
- Verify Streamlit dashboard works: `streamlit run app.py`
- Check all plots generate correctly
- Verify no data leakage (test metrics should be worse than train, not suspiciously better)

### README Update:
- Fill in all metric placeholder tables with actual results
- Add screenshot of Streamlit dashboard
- Add screenshot of MLflow UI
- Add sample SHAP plot image
- Add equity curve image
- Update requirements.txt if any new dependencies were added

---

## Implementation Order (Build Sequence)

| Order | File | Dependencies |
|-------|------|-------------|
| 1 | `config.py` | None |
| 2 | `requirements.txt` + `.gitignore` | None |
| 3 | `data/ingestion.py` | config |
| 4 | `data/feature_engineering.py` | ingestion |
| 5 | `data/labeling.py` | feature_engineering |
| 6 | `data/data_loader.py` | labeling |
| 7 | `tracking/mlflow_tracking.py` | config |
| 8 | `evaluation/metrics.py` | None |
| 9 | `models/classical/regression.py` | data_loader, metrics, mlflow_tracking |
| 10 | `models/classical/classification.py` | data_loader, metrics, mlflow_tracking |
| 11 | `evaluation/visualization.py` | metrics (partial -- model comparison plots) |
| 12 | `models/deep_learning/mlp.py` | data_loader, metrics, mlflow_tracking |
| 13 | `models/deep_learning/lstm.py` | data_loader, metrics, mlflow_tracking |
| 14 | `models/deep_learning/cnn.py` | data_loader, metrics, mlflow_tracking |
| 15 | `models/deep_learning/cnn_lstm.py` | data_loader, metrics, mlflow_tracking |
| 16 | `models/deep_learning/autoencoder.py` | data_loader, mlflow_tracking |
| 17 | `models/clustering/traditional.py` | feature_engineering, metrics |
| 18 | `models/clustering/deep_clustering.py` | autoencoder, traditional |
| 19 | `models/ensemble.py` | regression, classification, mlflow_tracking |
| 20 | `portfolio/signals.py` | ensemble |
| 21 | `portfolio/optimizer.py` | signals |
| 22 | `evaluation/backtester.py` | signals, optimizer, metrics, mlflow_tracking |
| 23 | `evaluation/visualization.py` | complete all remaining plot functions |
| 24 | `main.py` | everything |
| 25 | `app.py` | everything |
| 26 | Cleanup old files | after main.py works |
| 27 | Final README update with real metrics | after full run |

---

## Notes

- Every `__init__.py` file in each directory should be empty (just marks it as a package)
- All models must set `random_state=42` or `torch.manual_seed(42)` for reproducibility
- Scaler must be fit ONLY on training data, then transform val/test
- No shuffling in time-series splits
- SHAP only works reliably on tree-based models (RF, XGBoost, LightGBM) -- skip for SVM/SVR
- MLflow runs should have descriptive names: `ridge_regression_alpha_0.1`, `lstm_regression_lr_0.001`
- Save all plot images to `outputs/plots/` directory for the dashboard and README
