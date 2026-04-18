import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from config import PLOTS_DIR, get_logger

logger = get_logger(__name__)
os.makedirs(PLOTS_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot: {path}")
    return path


# ------------------------------------------------------------------
# Model comparison
# ------------------------------------------------------------------

def plot_model_comparison_regression(results: dict):
    """Bar chart comparing regression models on MSE, MAE, R², directional accuracy."""
    names = list(results.keys())
    metrics_to_plot = ["mse", "mae", "r2", "directional_accuracy"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, metric in zip(axes, metrics_to_plot):
        values = [results[n][1][metric] for n in names]
        ax.barh(names, values, color=sns.color_palette("coolwarm", len(names)))
        ax.set_title(metric.upper().replace("_", " "))
        ax.set_xlabel(metric)
    fig.suptitle("Regression Model Comparison", fontsize=14)
    fig.tight_layout()
    return _save(fig, "regression_comparison")


def plot_model_comparison_classification(results: dict):
    """Bar chart comparing classification models on accuracy, F1, ROC-AUC."""
    names = list(results.keys())
    metrics_to_plot = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, metric in zip(axes, metrics_to_plot):
        values = [results[n][1].get(metric, 0) for n in names]
        ax.barh(names, values, color=sns.color_palette("viridis", len(names)))
        ax.set_title(metric.upper().replace("_", " "))
    fig.suptitle("Classification Model Comparison", fontsize=14)
    fig.tight_layout()
    return _save(fig, "classification_comparison")


# ------------------------------------------------------------------
# Confusion matrix and ROC
# ------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Sell", "Hold", "Buy"], ax=ax, cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    return _save(fig, f"confusion_matrix_{model_name}")


def plot_roc_curves(y_true, y_prob, model_name):
    """Plot ROC curves (one-vs-rest) for multiclass."""
    fig, ax = plt.subplots(figsize=(7, 6))
    labels = ["Sell", "Hold", "Buy"]
    for i, label in enumerate(labels):
        binary_true = (np.asarray(y_true) == i).astype(int)
        RocCurveDisplay.from_predictions(binary_true, y_prob[:, i], name=label, ax=ax)
    ax.set_title(f"ROC Curves — {model_name}")
    ax.legend()
    return _save(fig, f"roc_curves_{model_name}")


# ------------------------------------------------------------------
# Training curves
# ------------------------------------------------------------------

def plot_training_curves(history, model_name):
    """Plot training vs validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train_loss"], label="Train Loss")
    ax.plot(history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Curves — {model_name}")
    ax.legend()
    return _save(fig, f"training_curves_{model_name}")


# ------------------------------------------------------------------
# SHAP
# ------------------------------------------------------------------

def plot_shap_summary(model, X, feature_names, model_name):
    """SHAP beeswarm and bar plot for tree-based models."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[:1000])  # limit for speed

        # Bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X[:1000], feature_names=feature_names,
                          plot_type="bar", show=False)
        path_bar = _save(plt.gcf(), f"shap_bar_{model_name}")

        # Beeswarm
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X[:1000], feature_names=feature_names, show=False)
        path_bee = _save(plt.gcf(), f"shap_beeswarm_{model_name}")

        return path_bar, path_bee
    except Exception as e:
        logger.warning(f"SHAP failed for {model_name}: {e}")
        return None, None


# ------------------------------------------------------------------
# Portfolio / backtest plots
# ------------------------------------------------------------------

def plot_equity_curve(portfolio_returns, benchmark_returns):
    """Cumulative returns: portfolio vs S&P 500."""
    fig, ax = plt.subplots(figsize=(12, 6))
    port_cum = (1 + portfolio_returns).cumprod()
    bench_cum = (1 + benchmark_returns).cumprod()

    ax.plot(port_cum.index, port_cum.values, label="ML Portfolio", linewidth=2)
    ax.plot(bench_cum.index, bench_cum.values, label="S&P 500", linewidth=2, alpha=0.7)
    ax.set_title("Cumulative Returns: Portfolio vs S&P 500")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, "equity_curve")


def plot_drawdown(portfolio_returns):
    """Drawdown chart over time."""
    cumulative = (1 + portfolio_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.4)
    ax.set_title("Portfolio Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    return _save(fig, "drawdown")


def plot_monthly_returns_heatmap(portfolio_returns):
    """Month x Year heatmap of returns."""
    returns_series = pd.Series(portfolio_returns.values, index=portfolio_returns.index)
    monthly = returns_series.resample("M").sum()
    pivot = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    }).pivot(index="year", columns="month", values="return")

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, annot=True, fmt=".2%", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Monthly Returns Heatmap")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    return _save(fig, "monthly_returns_heatmap")


def plot_portfolio_weights(weights, tickers):
    """Horizontal bar chart of portfolio allocations."""
    # Sort and show top 20
    sorted_idx = np.argsort(weights)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [tickers[i] for i in sorted_idx],
        [weights[i] for i in sorted_idx],
        color=sns.color_palette("coolwarm", 20),
    )
    ax.set_xlabel("Weight")
    ax.set_title("Portfolio Allocation (Top 20)")
    ax.invert_yaxis()
    return _save(fig, "portfolio_weights")


def plot_cluster_visualization(features, labels, method="tsne"):
    """2D scatter plot of clusters using t-SNE or PCA."""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=min(30, len(features) - 1), random_state=42)
    else:
        reducer = PCA(n_components=2)

    coords = reducer.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", alpha=0.6, s=20)
    ax.set_title(f"Stock Clusters ({method.upper()})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    return _save(fig, f"cluster_{method}")
