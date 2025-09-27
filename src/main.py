# -*- coding: utf-8 -*-
"""
ML and Optim project - Main script

Automatically generated from Colab notebook.
Original file is located at:
    https://colab.research.google.com/drive/1woWrebCRvpDRMu2xmSAeUtxkL0f2oOoD
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Import custom modules
from data_loader import load_classification_data, load_image_and_mask
from preprocessing import preprocess_test_images
from inscribed_circle import compute_inscribed_circles
from symmetry_plane import compute_symmetry_planes
from feature_extraction import create_feature_matrix
from feature_selection import (perform_sequential_feature_selection, 
                              perform_rfe_selection, perform_selectkbest_selection,
                              evaluate_feature_selection, plot_f1_scores)
from visualization import visualize_data
from ml_models import train_and_evaluate_ml_models
from clustering import perform_clustering
from deep_learning import load_and_prepare_data, train_dino_model, train_vit_model
from utils import create_submission_file, save_results, load_results

warnings.filterwarnings('ignore')

def main():
    print("Starting ML and Optimization project...")
    
    # Clone repository if needed
    if not os.path.exists('Pollinator-insects-classification'):
        os.system('git clone https://github.com/hugoaslm/Pollinator-insects-classification')
    
    # Load classification data
    print("Loading classification data...")
    classification_df = load_classification_data()
    print(f"Bug types: {classification_df['bug type'].unique()}")
    
    # Preprocess test images (251-347)
    print("Preprocessing test images...")
    # processed_images = preprocess_test_images()
    
    # Compute inscribed circles
    print("Computing inscribed circles...")
    # circle_features_array = compute_inscribed_circles()
    
    # Compute symmetry planes
    print("Computing symmetry planes...")
    # extra_features_array = compute_symmetry_planes()
    
    # Load or compute features
    print("Loading/computing features...")
    train_image_ids = classification_df['ID'].unique()
    
    # Option 1: Extract features from scratch
    # texture_color = create_feature_matrix(train_image_ids)
    # all_features_array = np.load('extra_features_3_class.npy')
    # X_train = np.concatenate((texture_color, all_features_array), axis=1)
    
    # Option 2: Load precomputed features
    X_train = np.load('all_feats_3_class_88.npy')
    
    # Get labels
    y_train = classification_df['bug type'].values
    
    # Prepare data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Label encoder
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_scaled, y_train_encoded, test_size=0.2, random_state=0, 
        stratify=y_train_encoded
    )
    
    # Visualization
    print("Creating visualizations...")
    visualize_data(X_train_scaled, y_train, classification_df)
    
    # Feature selection (optional)
    print("Performing feature selection...")
    best_features_sfs_indices, best_num_features, sfs_scores = perform_sequential_feature_selection(
        X_train_split, y_train_split, X_val, y_val)
    
    rfe_selected_features_indices = perform_rfe_selection(
        X_train_split, y_train_split, X_val, y_val, best_num_features)
    
    filter_selected_features_indices = perform_selectkbest_selection(
        X_train_split, y_train_split, X_val, y_val, best_num_features)
    
    k_range, scores = evaluate_feature_selection(X_train_split, y_train_split, X_val, y_val)
    plot_f1_scores(k_range, scores)
    
    # Machine Learning methods
    print("Training machine learning models...")
    ml_results = train_and_evaluate_ml_models(X_train_split, y_train_split, X_val, y_val)
    
    # Clustering methods
    print("Performing clustering...")
    clustering_results = perform_clustering(X_train_scaled, y_train)
    
    # Save results
    save_results(ml_results, 'ml_results_3_class.pkl')
    save_results(clustering_results, 'clustering_results_3_class.pkl')
    
    # Generate predictions for test set (images 251-347)
    print("Generating test predictions...")
    test_image_ids = list(range(251, 348))
    X_test = create_feature_matrix(test_image_ids)
    X_test_scaled = scaler.transform(X_test)
    
    best_model = ml_results['best_model']
    test_predictions = best_model.predict(X_test_scaled)
    test_predictions = label_encoder.inverse_transform(test_predictions)
    
    create_submission_file(test_image_ids, test_predictions)
    
    print("Project completed successfully!")

def run_deep_learning():
    """Run deep learning experiments separately"""
    print("Starting deep learning experiments...")
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Train DINO model
    print("Training DINO model...")
    dino_model = train_dino_model(df)
    
    # Train ViT model (optional)
    print("Training ViT model...")
    # vit_model = train_vit_model(df)
    
    print("Deep learning experiments completed!")

if __name__ == "__main__":
    # Run traditional ML pipeline
    main()
    
    # Uncomment to run deep learning experiments
    # run_deep_learning()