# To Bee or Not to Bee — Pollinator Classification with ML & Optimization

**IG.2412 — Machine Learning (2024/2025)**
Master's Project (Engineering cycle, ISEP Paris) — Machine Learning and Optimization.

## Overview
We tackle **pollinator insect classification** from images and binary masks, addressing two tasks:
- **Binary:** Bee vs. Bumblebee
- **Multiclass:** Bee, Bumblebee, Others (Hover fly, Wasp, Butterfly grouped)

The pipeline combines **classical machine learning** on engineered features and a **deep learning** approach based on a pre-trained vision backbone. The goal is to compare approaches, visualize separability, and quantify performance.

## Data & Preprocessing
- Images with **pixel masks** and class labels (IDs 1–347).
- **Largest connected component** kept from the mask; image cropped to its bounding box to remove background.
- All feature extraction and deep models operate on the **masked regions**.

## Feature Engineering
From masks and masked RGB/grayscale images we derive:
- **Shape:** area, perimeter, circularity, eccentricity, aspect ratio, solidity.
- **Texture (GLCM):** homogeneity, energy (computed at angles 0, 45, 90, 135 degrees).
- **Color:** per-channel min, max, mean, median, standard deviation; "bug ratio" (sum RGB / mask size).
- **Geometric:**  
  - **Largest inscribed circle** inside the mask using the **Euclidean Distance Transform** for initialization and **Nelder–Mead** to optimize center and radius under constraints.  
  - **Best symmetry axis** via 1D rotation search around the circle center: rotate, mirror around the vertical through the rotated center, and **minimize pixel-wise asymmetry** with Nelder–Mead.

## Classical ML
- **Feature selection:** SFS, RFE, and SelectKBest (ANOVA and Mutual Information). SFS found a compact subset (≈6–7 features) with strong validation accuracy.
- **Models evaluated:** k-NN, Decision Tree, SVM, Random Forest, XGBoost, Logistic Regression.
- **Highlights:**  
  - **Binary:** 3-NN with SFS-selected features reached **93% validation accuracy**.  
  - **Multiclass:** XGBoost with all features reached **88% validation accuracy**.

## Unsupervised Clustering
K-Means, Agglomerative, Spectral clustering were explored. Silhouette scores remained **low (~0.17–0.26)**, reflecting overlapping class structure and noisy separability.

## Deep Learning
- **Backbone:** Pre-trained **DINOv2** used as a frozen feature extractor.  
- **Head:** Two linear layers with ReLU and dropout on the CLS token output.
- **Training:** 15 epochs, AdamW, cosine annealing LR, standard augmentations; masked images resized to 224×224; class imbalance mitigated by augmentation and oversampling.
- **Results:** **95% accuracy** (binary) and **94%** (multiclass) on validation; the DL approach outperformed classical baselines while remaining lightweight to train.

## Takeaways
- **Simple geometric and texture features** already offer competitive baselines (especially for binary).  
- **DINOv2 features** provide a strong boost for both tasks with minimal fine-tuning.  
- Class overlap and label imbalance challenge unsupervised clustering and linear separability.

## Acknowledgments
Course staff and public pretrained backbones (DINOv2).
