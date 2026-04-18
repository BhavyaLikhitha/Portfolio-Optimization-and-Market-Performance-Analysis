import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from models.deep_learning.autoencoder import extract_latent_features
from config import RANDOM_SEED, get_logger

logger = get_logger(__name__)


def deep_cluster(autoencoder_model, features, k_values=None):
    """Cluster on autoencoder latent representations.

    Args:
        autoencoder_model: trained Autoencoder
        features: numpy array of raw features

    Returns:
        labels (best k), latent_features, best_k, scores
    """
    if k_values is None:
        k_values = [5, 10, 15]

    latent = extract_latent_features(autoencoder_model, features)
    logger.info(f"Latent features shape: {latent.shape}")

    best_score = -1
    best_labels = None
    best_k = None
    scores = {}

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(latent)
        sil = silhouette_score(latent, labels)
        db = davies_bouldin_score(latent, labels)
        scores[k] = {"silhouette": float(sil), "davies_bouldin": float(db)}
        logger.info(f"  Deep KMeans k={k}: silhouette={sil:.4f}")
        if sil > best_score:
            best_score = sil
            best_labels = labels
            best_k = k

    logger.info(f"Best deep cluster: k={best_k}, silhouette={best_score:.4f}")
    return best_labels, latent, best_k, scores


def compare_clustering_methods(traditional_scores, deep_scores) -> dict:
    """Compare traditional vs deep clustering results."""
    comparison = {
        "traditional": traditional_scores,
        "deep": deep_scores,
    }
    logger.info("Clustering comparison logged")
    return comparison
