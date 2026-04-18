import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from config import RANDOM_SEED, get_logger

logger = get_logger(__name__)


def evaluate_clusters(features, labels) -> dict:
    """Compute clustering quality metrics."""
    # Filter out noise label (-1) for metrics
    mask = labels >= 0
    n_clusters = len(set(labels[mask]))
    if n_clusters < 2:
        return {"silhouette": 0.0, "davies_bouldin": float("inf"), "n_clusters": n_clusters}

    return {
        "silhouette": float(silhouette_score(features[mask], labels[mask])),
        "davies_bouldin": float(davies_bouldin_score(features[mask], labels[mask])),
        "n_clusters": n_clusters,
    }


def run_kmeans(features, k_values=None):
    """Run K-Means for multiple k values, pick best by silhouette score.

    Returns:
        labels (best k), best_k, all_scores dict
    """
    if k_values is None:
        k_values = [5, 10, 15]

    best_score = -1
    best_labels = None
    best_k = None
    all_scores = {}

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(features)
        m = evaluate_clusters(features, labels)
        all_scores[k] = m
        logger.info(f"  KMeans k={k}: silhouette={m['silhouette']:.4f}")
        if m["silhouette"] > best_score:
            best_score = m["silhouette"]
            best_labels = labels
            best_k = k

    logger.info(f"Best KMeans: k={best_k}, silhouette={best_score:.4f}")
    return best_labels, best_k, all_scores


def run_hierarchical(features, n_clusters=10):
    """Agglomerative clustering with ward linkage."""
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = hc.fit_predict(features)
    m = evaluate_clusters(features, labels)
    logger.info(f"Hierarchical (n={n_clusters}): silhouette={m['silhouette']:.4f}")
    return labels, m


def run_dbscan(features, eps_values=None, min_samples=5):
    """Run DBSCAN for multiple eps values, pick best by silhouette score.

    Returns:
        labels (best eps), best_eps, all_scores dict
    """
    if eps_values is None:
        eps_values = [0.5, 1.0, 1.5, 2.0]

    best_score = -1
    best_labels = None
    best_eps = None
    all_scores = {}

    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(features)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            logger.info(f"  DBSCAN eps={eps}: only {n_clusters} cluster(s), skipping")
            continue
        m = evaluate_clusters(features, labels)
        all_scores[eps] = m
        logger.info(f"  DBSCAN eps={eps}: silhouette={m['silhouette']:.4f}, clusters={n_clusters}")
        if m["silhouette"] > best_score:
            best_score = m["silhouette"]
            best_labels = labels
            best_eps = eps

    if best_labels is None:
        logger.warning("DBSCAN found no valid clustering. Returning single cluster.")
        best_labels = np.zeros(len(features), dtype=int)
        best_eps = eps_values[0]

    logger.info(f"Best DBSCAN: eps={best_eps}, silhouette={best_score:.4f}")
    return best_labels, best_eps, all_scores
