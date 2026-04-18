import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from config import OUTPUTS_DIR

SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "pipeline_summary.json")

st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def load_summary():
    with open(SUMMARY_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_csv(path):
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def metrics_frame(section):
    rows = []
    for model_name, payload in section["validation"].items():
        row = {"model": model_name}
        row.update(payload["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)


def show_artifact(path, caption):
    if path and os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"{caption} is not available in the current run.")


def show_overview(summary):
    st.subheader("Run Summary")
    metadata = summary.get("metadata", {})
    dataset = summary.get("dataset", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Start Date", metadata.get("start_date", "-"))
    col2.metric("End Date", metadata.get("end_date", "-"))
    col3.metric("Tickers", dataset.get("n_tickers", 0))
    col4.metric("Features", dataset.get("feature_count", 0))

    backtest = summary.get("backtest", {})
    if isinstance(backtest, dict) and "metrics" in backtest:
        backtest = backtest["metrics"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return", f"{backtest['total_return']:.2%}")
        col2.metric("Sharpe", f"{backtest['sharpe_ratio']:.2f}")
        col3.metric("Max Drawdown", f"{backtest['max_drawdown']:.2%}")
        col4.metric("Benchmark Return", f"{backtest['benchmark_total_return']:.2%}")


def show_model_comparison(summary):
    st.subheader("Regression Models")
    reg_df = metrics_frame(summary["regression"]).sort_values("r2", ascending=False)
    st.dataframe(reg_df, use_container_width=True)
    show_artifact(summary["artifacts"].get("regression_plot"), "Regression comparison")

    st.subheader("Classification Models")
    clf_df = metrics_frame(summary["classification"]).sort_values("f1_macro", ascending=False)
    st.dataframe(clf_df, use_container_width=True)
    show_artifact(summary["artifacts"].get("classification_plot"), "Classification comparison")


def show_explainability(summary):
    st.subheader("Classification Diagnostics")
    col1, col2 = st.columns(2)
    with col1:
        show_artifact(summary["artifacts"].get("confusion_matrix_plot"), "Confusion matrix")
    with col2:
        show_artifact(summary["artifacts"].get("roc_plot"), "ROC curves")

    st.subheader("SHAP")
    col1, col2 = st.columns(2)
    with col1:
        show_artifact(summary["artifacts"].get("shap_bar"), "SHAP feature importance")
    with col2:
        show_artifact(summary["artifacts"].get("shap_beeswarm"), "SHAP beeswarm")


def show_clustering(summary):
    clustering = summary.get("clustering")
    if not clustering:
        st.info("Clustering artifacts were not generated for this run.")
        return
    if isinstance(clustering, dict) and "status" in clustering:
        st.info(clustering["status"])
        return

    scores = {
        "kmeans_best_k": clustering.get("best_kmeans_k"),
        "dbscan_best_eps": clustering.get("best_dbscan_eps"),
        "deep_best_k": clustering.get("best_deep_k"),
    }
    st.json(scores)

    cluster_membership = load_csv(clustering.get("cluster_membership_path"))
    if not cluster_membership.empty:
        st.subheader("Cluster Membership")
        st.dataframe(cluster_membership.head(50), use_container_width=True)

        counts = cluster_membership["kmeans_cluster"].value_counts().sort_index().reset_index()
        counts.columns = ["cluster", "count"]
        fig = px.bar(counts, x="cluster", y="count", title="K-Means Cluster Distribution")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        show_artifact(summary["artifacts"].get("cluster_tsne_plot"), "Cluster visualization (t-SNE)")
    with col2:
        show_artifact(summary["artifacts"].get("cluster_pca_plot"), "Deep cluster visualization (PCA)")


def show_backtest(summary):
    backtest = summary.get("backtest")
    if not backtest:
        st.info("Backtest artifacts were not generated for this run.")
        return
    if isinstance(backtest, dict) and "status" in backtest:
        st.info(backtest["status"])
        return

    returns_df = load_csv(backtest.get("returns_path"))
    rebalance_df = load_csv(backtest.get("rebalance_log_path"))

    if not returns_df.empty:
        returns_df["date"] = pd.to_datetime(returns_df["date"])
        returns_df["equity_curve"] = (1 + returns_df["portfolio_return"]).cumprod()
        fig = px.line(returns_df, x="date", y="equity_curve", title="Portfolio Equity Curve")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        show_artifact(summary["artifacts"].get("drawdown_plot"), "Drawdown")
    with col2:
        show_artifact(summary["artifacts"].get("monthly_heatmap_plot"), "Monthly returns heatmap")

    show_artifact(summary["artifacts"].get("weights_plot"), "Latest portfolio weights")

    if not rebalance_df.empty:
        st.subheader("Recent Rebalances")
        st.dataframe(rebalance_df.tail(12), use_container_width=True)


def show_deep_learning(summary):
    st.subheader("Deep Learning")
    dl = summary.get("deep_learning")
    if not dl:
        st.info("No deep-learning results were captured for this run.")
        return
    if isinstance(dl, dict) and "status" in dl:
        st.info(dl["status"])
        return

    task = st.selectbox("Task", ["regression", "classification"])
    task_results = dl.get(task, {})
    if not task_results:
        st.info(f"No {task} deep-learning results were captured for this run.")
        return

    rows = []
    for model_name, payload in task_results.items():
        row = {"model": model_name}
        row.update(payload.get("test_metrics", {}))
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    selected_model = st.selectbox("Model", list(task_results.keys()))
    payload = task_results[selected_model]
    history = load_json(payload.get("history_path"))
    if history:
        history_df = pd.DataFrame(history)
        fig = px.line(history_df, title=f"{selected_model} {task.title()} History")
        st.plotly_chart(fig, use_container_width=True)

    curve_key = f"{selected_model}_{task}_curve"
    show_artifact(summary["artifacts"].get(curve_key), f"{selected_model} {task} training curve")


st.title("Portfolio Optimization Dashboard")

if not os.path.exists(SUMMARY_PATH):
    st.error("No pipeline artifacts found. Run `python main.py` first to generate `outputs/pipeline_summary.json`.")
    st.stop()

summary = load_summary()

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Section",
    [
        "Overview",
        "Model Comparison",
        "Deep Learning",
        "Explainability",
        "Clustering",
        "Portfolio & Backtest",
    ],
)

show_overview(summary)

if page == "Model Comparison":
    show_model_comparison(summary)
elif page == "Deep Learning":
    show_deep_learning(summary)
elif page == "Explainability":
    show_explainability(summary)
elif page == "Clustering":
    show_clustering(summary)
elif page == "Portfolio & Backtest":
    show_backtest(summary)
else:
    st.subheader("Best Models")
    col1, col2 = st.columns(2)
    col1.metric("Best Regressor", summary["regression"]["best_model"])
    col2.metric("Best Classifier", summary["classification"]["best_model"])
    show_artifact(summary["artifacts"].get("equity_curve_plot"), "Equity curve")
