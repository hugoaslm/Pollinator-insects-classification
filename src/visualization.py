# -*- coding: utf-8 -*-
"""Data visualization utilities"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap

def visualize_pca(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    for class_label in np.unique(y):
        mask = y == class_label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=class_label, alpha=0.7)

    plt.title('PCA Projection of Bug Features')
    plt.xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2f})')
    plt.ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2f})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('pca_projection.png')

def visualize_tsne(X, y):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    for class_label in np.unique(y):
        mask = y == class_label
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=class_label, alpha=0.7)

    plt.title('t-SNE Projection of Bug Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('tsne_projection.png')

def visualize_isomap(X, y):
    """Create Isomap visualization"""
    isomap = Isomap(n_components=2, n_neighbors=10)
    X_isomap = isomap.fit_transform(X)

    plt.figure(figsize=(10, 8))
    for class_label in np.unique(y):
        mask = y == class_label
        plt.scatter(X_isomap[mask, 0], X_isomap[mask, 1], label=class_label, alpha=0.7)

    plt.title('Isomap Projection of Bug Features')
    plt.xlabel('Isomap Dimension 1')
    plt.ylabel('Isomap Dimension 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('isomap_projection.png')

def visualize_data(X_scaled, y, classification_df):
    # 1. Distribution of bug types
    plt.figure(figsize=(12, 6))
    sns.countplot(data=classification_df, x='bug type')
    plt.title('Distribution of Bug Types')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('bug_type_distribution.png')

    # 2. Distribution of species
    plt.figure(figsize=(14, 8))
    sns.countplot(data=classification_df, x='species')
    plt.title('Distribution of Species')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('species_distribution.png')

    # 3. PCA projection
    visualize_pca(X_scaled, y)

    # 4. t-SNE projection (non-linear)
    visualize_tsne(X_scaled, y)

    # 5. Isomap projection (non-linear)
    visualize_isomap(X_scaled, y)