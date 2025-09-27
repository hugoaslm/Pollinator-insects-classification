# -*- coding: utf-8 -*-
"""Clustering algorithms implementation"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from config import KMEANS_N_INIT, DBSCAN_EPS, DBSCAN_MIN_SAMPLES

def visualize_clustering_results(X, y_pred, y_true, method_name):
    """Visualize clustering results using PCA"""
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot clustering results
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
    ax1.set_title(f'{method_name} Clustering Results')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.grid(True, linestyle='--', alpha=0.7)
    legend1 = ax1.legend(*scatter1.legend_elements(), title="Clusters")
    ax1.add_artist(legend1)

    # Plot true labels
    # Convert string labels to numerical labels for color mapping
    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true)

    scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true_encoded, cmap='tab10', alpha=0.7)
    ax2.set_title('True Labels')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Use the original labels in the legend
    legend2 = ax2.legend(*scatter2.legend_elements(), title="True Labels", labels=le.classes_)
    ax2.add_artist(legend2)

    plt.tight_layout()
    plt.savefig(f'clustering_{method_name.lower()}.png')

def apply_kmeans(X, y_true):
    """Apply K-Means clustering"""
    # Determine number of clusters from true labels
    n_clusters = len(np.unique(y_true))

    # Train KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=KMEANS_N_INIT)
    y_pred = kmeans.fit_predict(X)

    # Evaluate clustering
    silhouette = silhouette_score(X, y_pred)

    # Visualize clustering results with PCA
    visualize_clustering_results(X, y_pred, y_true, "K-Means")

    return {
        'model': kmeans,
        'predictions': y_pred,
        'silhouette_score': silhouette
    }

def apply_agglomerative(X, y_true):
    """Apply Agglomerative clustering"""
    # Determine number of clusters from true labels
    n_clusters = len(np.unique(y_true))

    # Train Agglomerative clustering
    agglom = AgglomerativeClustering(n_clusters=n_clusters)
    y_pred = agglom.fit_predict(X)

    # Evaluate clustering
    silhouette = silhouette_score(X, y_pred)

    # Visualize clustering results with PCA
    visualize_clustering_results(X, y_pred, y_true, "Agglomerative")

    return {
        'model': agglom,
        'predictions': y_pred,
        'silhouette_score': silhouette
    }

def apply_dbscan(X, y_true):
    """Apply DBSCAN clustering"""
    # Train DBSCAN
    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    y_pred = dbscan.fit_predict(X)

    # Evaluate clustering if more than one cluster was found
    silhouette = None
    if len(np.unique(y_pred)) > 1 and -1 not in y_pred:
        silhouette = silhouette_score(X, y_pred)

    # Visualize clustering results with PCA if clusters were found
    if len(np.unique(y_pred)) > 1:
        visualize_clustering_results(X, y_pred, y_true, "DBSCAN")

    return {
        'model': dbscan,
        'predictions': y_pred,
        'silhouette_score': silhouette
    }

def apply_spectral_clustering(X, y_true):
    """Apply Spectral clustering"""
    # Train spectral clustering
    spectral = SpectralClustering(n_clusters=len(np.unique(y_true)), affinity='nearest_neighbors')
    y_pred = spectral.fit_predict(X)

    # Evaluate clustering
    silhouette = silhouette_score(X, y_pred)

    # Visualize clustering results with PCA
    visualize_clustering_results(X, y_pred, y_true, "Spectral")

    return {
        'model': spectral,
        'predictions': y_pred,
        'silhouette_score': silhouette
    }

def perform_clustering(X, y_true):
    results = {}

    # 1. K-Means clustering
    kmeans_results = apply_kmeans(X, y_true)
    results['kmeans'] = kmeans_results

    # 2. Agglomerative clustering
    agglom_results = apply_agglomerative(X, y_true)
    results['agglomerative'] = agglom_results

    # 3. DBSCAN (optional additional clustering)
    dbscan_results = apply_dbscan(X, y_true)
    results['dbscan'] = dbscan_results

    # 4. Spectral clustering
    spectral_results = apply_spectral_clustering(X, y_true)
    results['spectral'] = spectral_results

    return results