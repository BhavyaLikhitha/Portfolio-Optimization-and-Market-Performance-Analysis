import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from config import OUTPUTS_DIR, PLOTS_DIR, START_DATE, VAL_END, get_logger
from data.data_loader import create_dataloaders, prepare_tabular_data, split_by_time
from data.feature_engineering import build_feature_matrix
from data.ingestion import fetch_sp500_tickers, fetch_stock_data, get_close_prices, get_sp500_index
from data.labeling import (
    create_classification_labels,
    create_regression_labels,
    merge_features_and_labels,
)
from evaluation.backtester import backtest_report, compute_benchmark_returns, walk_forward_backtest
from evaluation.metrics import classification_metrics, regression_metrics
from evaluation.visualization import (
    plot_cluster_visualization,
    plot_confusion_matrix,
    plot_drawdown,
    plot_equity_curve,
    plot_model_comparison_classification,
    plot_model_comparison_regression,
    plot_monthly_returns_heatmap,
    plot_portfolio_weights,
    plot_roc_curves,
    plot_shap_summary,
)
from models.classical.classification import get_best_classifier, train_all_classifiers
from models.classical.regression import get_best_regressor, train_all_regressors
from models.clustering.traditional import run_dbscan, run_hierarchical, run_kmeans
from models.ensemble import (
    build_stacking_regressor,
    build_voting_classifier,
    predict_stacking,
    predict_voting,
)

logger = get_logger(__name__)


class StackingPredictor:
    def __init__(self, ensemble):
        self.ensemble = ensemble

    def predict(self, X):
        return predict_stacking(self.ensemble, X)


class VotingPredictor:
    def __init__(self, ensemble):
        self.ensemble = ensemble

    def predict(self, X):
        return predict_voting(self.ensemble, X)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the modular portfolio-optimization pipeline.")
    parser.add_argument("--start-date", default=START_DATE, help="Pipeline start date.")
    parser.add_argument(
        "--end-date",
        default=datetime.today().strftime("%Y-%m-%d"),
        help="Pipeline end date.",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=75,
        help="Limit the universe size for faster experiments.",
    )
    parser.add_argument(
        "--rebalance-freq",
        default="M",
        help="Pandas rebalance frequency used by the backtester.",
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip autoencoder and clustering stages.",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip walk-forward backtesting and portfolio plots.",
    )
    parser.add_argument(
        "--skip-deep-learning",
        action="store_true",
        help="Skip PyTorch-based model training even if torch is available.",
    )
    parser.add_argument(
        "--dl-epochs",
        type=int,
        default=20,
        help="Epochs for deep-learning models in the orchestrated pipeline.",
    )
    return parser.parse_args()


def ensure_output_dirs():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def dump_json(payload, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(make_json_safe(payload), handle, indent=2)


def summarize_model_results(results):
    summary = {}
    for name, (_, metrics, params) in results.items():
        summary[name] = {
            "metrics": metrics,
            "params": params,
        }
    return summary


def top_models(results, metric_name, top_k=3):
    ranked = sorted(results.items(), key=lambda item: item[1][1][metric_name], reverse=True)
    return [(name, payload[0]) for name, payload in ranked[:top_k]]


def average_probabilities(base_models, X):
    probabilities = [model.predict_proba(X) for _, model in base_models if hasattr(model, "predict_proba")]
    if not probabilities:
        return None
    return np.mean(probabilities, axis=0)


def maybe_plot_shap(model_name, model, X, feature_cols):
    shap_supported = any(token in model_name for token in ["random_forest", "xgboost", "lightgbm"])
    if not shap_supported:
        return {}

    bar_path, beeswarm_path = plot_shap_summary(model, X, feature_cols, model_name)
    artifacts = {}
    if bar_path:
        artifacts["shap_bar"] = bar_path
    if beeswarm_path:
        artifacts["shap_beeswarm"] = beeswarm_path
    return artifacts


def build_cluster_frame(full_df, feature_cols, scaler):
    latest = (
        full_df.reset_index()
        .sort_values(["ticker", "date"])
        .groupby("ticker", as_index=False)
        .tail(1)
        .set_index("ticker")
    )
    latest_features = scaler.transform(latest[feature_cols].values)
    return latest, latest_features


def apply_scaler_to_split(df, feature_cols, scaler):
    scaled = df.copy()
    scaled.loc[:, feature_cols] = scaler.transform(scaled[feature_cols].values)
    return scaled


def run_deep_learning_suite(
    train_df,
    val_df,
    test_df,
    feature_cols,
    epochs,
):
    from models.deep_learning.cnn import evaluate_cnn, train_cnn
    from models.deep_learning.cnn_lstm import evaluate_cnn_lstm, train_cnn_lstm
    from models.deep_learning.lstm import evaluate_lstm, train_lstm
    from models.deep_learning.mlp import evaluate_mlp, train_mlp

    dl_summary = {"regression": {}, "classification": {}}
    artifacts = {}

    reg_train_loader, reg_val_loader, reg_test_loader = create_dataloaders(
        train_df, val_df, test_df, target_col="fwd_return_10d", feature_cols=feature_cols
    )
    clf_train_loader, clf_val_loader, clf_test_loader = create_dataloaders(
        train_df, val_df, test_df, target_col="signal", feature_cols=feature_cols
    )

    model_specs = [
        ("mlp", train_mlp, evaluate_mlp),
        ("lstm", train_lstm, evaluate_lstm),
        ("cnn", train_cnn, evaluate_cnn),
        ("cnn_lstm", train_cnn_lstm, evaluate_cnn_lstm),
    ]

    input_size = len(feature_cols)
    for model_name, train_fn, eval_fn in model_specs:
        model, history, train_meta = train_fn(
            reg_train_loader, reg_val_loader, input_size=input_size, task="regression", epochs=epochs
        )
        test_metrics = eval_fn(model, reg_test_loader, task="regression")
        history_path = os.path.join(OUTPUTS_DIR, f"{model_name}_regression_history.json")
        dump_json(history, history_path)
        dl_summary["regression"][model_name] = {
            "training": train_meta,
            "test_metrics": test_metrics,
            "history_path": history_path,
        }
        artifacts[f"{model_name}_regression_curve"] = os.path.join(PLOTS_DIR, f"training_curves_{model_name}_regression.png")

        model, history, train_meta = train_fn(
            clf_train_loader, clf_val_loader, input_size=input_size, task="classification", epochs=epochs
        )
        test_metrics = eval_fn(model, clf_test_loader, task="classification")
        history_path = os.path.join(OUTPUTS_DIR, f"{model_name}_classification_history.json")
        dump_json(history, history_path)
        dl_summary["classification"][model_name] = {
            "training": train_meta,
            "test_metrics": {k: v for k, v in test_metrics.items() if k != "confusion_matrix"},
            "history_path": history_path,
        }
        artifacts[f"{model_name}_classification_curve"] = os.path.join(
            PLOTS_DIR, f"training_curves_{model_name}_classification.png"
        )

    return dl_summary, artifacts


def run_pipeline(args):
    ensure_output_dirs()
    summary = {
        "metadata": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "max_tickers": args.max_tickers,
            "rebalance_freq": args.rebalance_freq,
        },
        "artifacts": {},
    }

    logger.info("Stage 1/7 - downloading market data")
    tickers_df = fetch_sp500_tickers()
    tickers = tickers_df["Symbol"].tolist()[: args.max_tickers]
    ohlcv = fetch_stock_data(tickers, args.start_date, args.end_date)
    close_prices = get_close_prices(ohlcv)
    sp500_prices = get_sp500_index(args.start_date, args.end_date)

    logger.info("Stage 2/7 - feature engineering and labels")
    features_df, _ = build_feature_matrix(ohlcv, scale=False)
    regression_labels = create_regression_labels(close_prices)
    classification_labels = create_classification_labels(close_prices)
    full_df = merge_features_and_labels(features_df, regression_labels, classification_labels)
    train_df, val_df, test_df = split_by_time(full_df)

    (
        X_train_reg,
        y_train_reg,
        X_val_reg,
        y_val_reg,
        X_test_reg,
        y_test_reg,
        scaler,
        feature_cols,
    ) = prepare_tabular_data(train_df, val_df, test_df, target_col="fwd_return_10d")
    (
        X_train_clf,
        y_train_clf,
        X_val_clf,
        y_val_clf,
        X_test_clf,
        y_test_clf,
        _,
        _,
    ) = prepare_tabular_data(train_df, val_df, test_df, target_col="signal", feature_cols=feature_cols)

    summary["dataset"] = {
        "n_tickers": len(close_prices.columns),
        "feature_count": len(feature_cols),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "test_start": str(test_df.index.get_level_values("date").min().date()),
        "test_end": str(test_df.index.get_level_values("date").max().date()),
    }

    logger.info("Stage 3/7 - classical models and ensembles")
    regression_results = train_all_regressors(X_train_reg, y_train_reg, X_val_reg, y_val_reg)
    classification_results = train_all_classifiers(X_train_clf, y_train_clf, X_val_clf, y_val_clf)

    best_reg_name, best_reg_model = get_best_regressor(regression_results)
    best_clf_name, best_clf_model = get_best_classifier(classification_results)

    best_reg_test_preds = best_reg_model.predict(X_test_reg)
    best_reg_test_metrics = regression_metrics(y_test_reg, best_reg_test_preds)

    best_clf_test_preds = best_clf_model.predict(X_test_clf)
    best_clf_test_probs = best_clf_model.predict_proba(X_test_clf) if hasattr(best_clf_model, "predict_proba") else None
    best_clf_test_metrics = classification_metrics(y_test_clf, best_clf_test_preds, best_clf_test_probs)

    top_regressors = top_models(regression_results, "r2")
    top_classifiers = top_models(classification_results, "f1_macro")

    stacking = build_stacking_regressor(top_regressors, X_train_reg, y_train_reg, X_val_reg, y_val_reg)
    stacking_test_preds = predict_stacking(stacking, X_test_reg)
    stacking_test_metrics = regression_metrics(y_test_reg, stacking_test_preds)

    voting = build_voting_classifier(top_classifiers, X_val_clf, y_val_clf)
    voting_test_preds = predict_voting(voting, X_test_clf)
    voting_test_probs = average_probabilities(top_classifiers, X_test_clf)
    voting_test_metrics = classification_metrics(y_test_clf, voting_test_preds, voting_test_probs)

    summary["regression"] = {
        "validation": summarize_model_results(regression_results),
        "best_model": best_reg_name,
        "best_test_metrics": best_reg_test_metrics,
        "stacking_test_metrics": stacking_test_metrics,
    }
    summary["classification"] = {
        "validation": summarize_model_results(classification_results),
        "best_model": best_clf_name,
        "best_test_metrics": {k: v for k, v in best_clf_test_metrics.items() if k != "confusion_matrix"},
        "voting_test_metrics": {k: v for k, v in voting_test_metrics.items() if k != "confusion_matrix"},
    }

    regression_plot = plot_model_comparison_regression(regression_results)
    classification_plot = plot_model_comparison_classification(classification_results)
    confusion_plot = plot_confusion_matrix(y_test_clf, voting_test_preds, "voting_classifier")

    summary["artifacts"]["regression_plot"] = regression_plot
    summary["artifacts"]["classification_plot"] = classification_plot
    summary["artifacts"]["confusion_matrix_plot"] = confusion_plot

    if voting_test_probs is not None:
        roc_plot = plot_roc_curves(y_test_clf, voting_test_probs, "voting_classifier")
        summary["artifacts"]["roc_plot"] = roc_plot

    summary["artifacts"].update(maybe_plot_shap(best_reg_name, best_reg_model, X_test_reg, feature_cols))

    logger.info("Stage 4/7 - deep learning")
    if not args.skip_deep_learning:
        try:
            import torch

            _ = torch.__version__
            scaled_train_df = apply_scaler_to_split(train_df, feature_cols, scaler)
            scaled_val_df = apply_scaler_to_split(val_df, feature_cols, scaler)
            scaled_test_df = apply_scaler_to_split(test_df, feature_cols, scaler)
            dl_summary, dl_artifacts = run_deep_learning_suite(
                scaled_train_df,
                scaled_val_df,
                scaled_test_df,
                feature_cols,
                epochs=args.dl_epochs,
            )
            summary["deep_learning"] = dl_summary
            summary["artifacts"].update(dl_artifacts)
        except Exception as exc:
            logger.warning("Skipping deep-learning suite: %s", exc)
            summary["deep_learning"] = {"status": f"skipped: {exc}"}
    else:
        summary["deep_learning"] = {"status": "skipped by flag"}

    logger.info("Stage 5/7 - clustering")
    if not args.skip_clustering:
        try:
            from models.clustering.deep_clustering import deep_cluster
            from models.deep_learning.autoencoder import train_autoencoder

            latest_frame, latest_features = build_cluster_frame(full_df, feature_cols, scaler)
            kmeans_labels, best_k, kmeans_scores = run_kmeans(latest_features)
            hier_labels, hier_metrics = run_hierarchical(latest_features, n_clusters=best_k or 10)
            dbscan_labels, best_eps, dbscan_scores = run_dbscan(latest_features)

            autoencoder, _, _ = train_autoencoder(X_train_reg, X_val_reg, input_size=len(feature_cols))
            deep_labels, latent_features, deep_k, deep_scores = deep_cluster(autoencoder, latest_features)

            cluster_frame = pd.DataFrame(
                {
                    "ticker": latest_frame.index.tolist(),
                    "date": latest_frame["date"].dt.strftime("%Y-%m-%d").tolist(),
                    "kmeans_cluster": kmeans_labels,
                    "hierarchical_cluster": hier_labels,
                    "dbscan_cluster": dbscan_labels,
                    "deep_cluster": deep_labels,
                }
            )
            cluster_csv = os.path.join(OUTPUTS_DIR, "cluster_membership.csv")
            cluster_frame.to_csv(cluster_csv, index=False)

            summary["clustering"] = {
                "best_kmeans_k": best_k,
                "best_dbscan_eps": best_eps,
                "best_deep_k": deep_k,
                "kmeans_scores": kmeans_scores,
                "hierarchical_metrics": hier_metrics,
                "dbscan_scores": dbscan_scores,
                "deep_scores": deep_scores,
                "cluster_membership_path": cluster_csv,
            }

            summary["artifacts"]["cluster_tsne_plot"] = plot_cluster_visualization(latest_features, kmeans_labels, method="tsne")
            summary["artifacts"]["cluster_pca_plot"] = plot_cluster_visualization(latent_features, deep_labels, method="pca")
        except Exception as exc:
            logger.warning("Skipping clustering suite: %s", exc)
            summary["clustering"] = {"status": f"skipped: {exc}"}
    else:
        summary["clustering"] = {"status": "skipped by flag"}

    logger.info("Stage 6/7 - backtest")
    if not args.skip_backtest:
        portfolio_returns, rebalance_log = walk_forward_backtest(
            test_df=test_df,
            close_prices=close_prices,
            regression_model=StackingPredictor(stacking),
            classification_model=VotingPredictor(voting),
            feature_cols=feature_cols,
            scaler=scaler,
            rebalance_freq=args.rebalance_freq,
        )

        benchmark_returns = compute_benchmark_returns(
            sp500_prices=sp500_prices,
            test_start=test_df.index.get_level_values("date").min(),
            test_end=test_df.index.get_level_values("date").max(),
        )
        backtest_metrics = backtest_report(portfolio_returns, benchmark_returns)

        returns_path = os.path.join(OUTPUTS_DIR, "backtest_returns.csv")
        portfolio_returns.rename("portfolio_return").to_csv(returns_path, index_label="date")

        rebalance_frame = pd.DataFrame(rebalance_log)
        rebalance_path = os.path.join(OUTPUTS_DIR, "rebalance_log.csv")
        rebalance_frame.to_csv(rebalance_path, index=False)

        summary["backtest"] = {
            "metrics": backtest_metrics,
            "returns_path": returns_path,
            "rebalance_log_path": rebalance_path,
        }

        summary["artifacts"]["equity_curve_plot"] = plot_equity_curve(portfolio_returns, benchmark_returns)
        summary["artifacts"]["drawdown_plot"] = plot_drawdown(portfolio_returns)
        summary["artifacts"]["monthly_heatmap_plot"] = plot_monthly_returns_heatmap(portfolio_returns)

        if rebalance_log:
            last_rebalance = rebalance_log[-1]
            summary["artifacts"]["weights_plot"] = plot_portfolio_weights(
                np.array(last_rebalance["weights"]),
                last_rebalance["all_tickers"],
            )
    else:
        summary["backtest"] = {"status": "skipped by flag"}

    logger.info("Stage 7/7 - writing summary artifact")
    summary_path = os.path.join(OUTPUTS_DIR, "pipeline_summary.json")
    dump_json(summary, summary_path)
    logger.info("Pipeline complete. Summary written to %s", summary_path)
    return summary_path, summary


def print_summary(summary):
    print("\nPipeline complete.")
    print(f"Best regression model: {summary['regression']['best_model']}")
    print(f"Best classification model: {summary['classification']['best_model']}")
    if "backtest" in summary:
        sharpe = summary["backtest"]["metrics"]["sharpe_ratio"]
        total_return = summary["backtest"]["metrics"]["total_return"]
        print(f"Backtest Sharpe: {sharpe:.3f}")
        print(f"Backtest total return: {total_return:.2%}")


if __name__ == "__main__":
    arguments = parse_args()
    _, pipeline_summary = run_pipeline(arguments)
    print_summary(pipeline_summary)
