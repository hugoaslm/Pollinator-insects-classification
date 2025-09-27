# -*- coding: utf-8 -*-
"""Feature selection utilities"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SequentialFeatureSelector, RFE
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm

def select_features(X, y, method='anova', k=10):
    """
    Select top k features using ANOVA F-test or Mutual Information.

    Parameters:
    - X : feature matrix (n_samples x n_features)
    - y : labels
    - method : 'anova' or 'mutual_info'
    - k : number of top features to select

    Returns:
    - X_new : reduced feature matrix
    - selector : fitted selector object
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == 'anova':
        selector = SelectKBest(score_func=f_classif, k=k)
    elif method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    else:
        raise ValueError("Method must be 'anova' or 'mutual_info'")

    X_new = selector.fit_transform(X_scaled, y)

    # Plot the scores of the features
    plt.figure(figsize=(10, 6))
    scores = selector.scores_
    plt.bar(range(len(scores)), scores)
    plt.xlabel('Feature Index')
    plt.ylabel('Score')
    plt.title(f'Feature Scores using {method}')
    plt.show()

    return X_new, selector

def perform_sequential_feature_selection(X_train_split, y_train_split, X_val, y_val):
    num_features = X_train_split.shape[1]
    feature_names = [f'feature_{i}' for i in range(num_features)]

    print(f"Assumed number of features: {len(feature_names)}")
    print(f"X_train_split shape: {X_train_split.shape}")
    print(f"y_train_split shape: {y_train_split.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")

    estimator = RandomForestClassifier(n_estimators=50, random_state=0)

    print("\n--- Performing Sequential Feature Selection (Forward) ---")

    best_score = -1
    best_num_features = 0
    best_features_sfs_indices = None
    sfs_scores = []

    # Define the cross-validation strategy for evaluation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for k in tqdm(range(1, X_train_split.shape[1]), desc="Testing feature subsets (SFS)"):
        sfs = SequentialFeatureSelector(
            estimator,
            n_features_to_select=k,
            direction='forward',
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )

        sfs.fit(X_train_split, y_train_split)

        # Get the indices of the features selected by SFS for this specific k
        selected_features_indices_k = sfs.get_support(indices=True)

        # Select the subset of training data using these indices
        X_train_selected_k = X_train_split[:, selected_features_indices_k]

        # Evaluate the estimator using cross-validation
        scores = cross_val_score(estimator, X_train_selected_k, y_train_split, cv=cv, scoring='accuracy', n_jobs=-1)
        current_cv_score = scores.mean()

        sfs_scores.append(current_cv_score)

        if current_cv_score > best_score:
            best_score = current_cv_score
            best_num_features = k
            best_features_sfs_indices = selected_features_indices_k

    # Get names using the provided feature_names list and the stored indices for the overall best k
    best_features_sfs_names = [feature_names[i] for i in best_features_sfs_indices]

    print(f"\nSFS Results:")
    print(f"Best number of features (based on internal CV mean): {best_num_features}")
    print(f"Best SFS Cross-Validation Score (mean across folds for best k): {best_score:.4f}")
    print(f"Best SFS Selected feature names: {best_features_sfs_names}")

    print("\n--- Evaluating Final Model with SFS Selected Features on Validation Set ---")

    # Select the best features from X_train_split and X_val using the determined indices
    X_train_selected_sfs = X_train_split[:, best_features_sfs_indices]
    X_val_selected_sfs = X_val[:, best_features_sfs_indices]

    # Train the final model on the selected features
    final_model_sfs = RandomForestClassifier(n_estimators=50, random_state=0)
    final_model_sfs.fit(X_train_selected_sfs, y_train_split)

    # Evaluate the final model on the validation set
    val_accuracy_sfs = final_model_sfs.score(X_val_selected_sfs, y_val)
    print(f"Validation Accuracy with SFS selected features: {val_accuracy_sfs:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, X_train_split.shape[1]), sfs_scores, marker='o')
    plt.title('Sequential Feature Selector (SFS) Cross-Validation Scores vs. Number of Features')
    plt.xlabel('Number of Features Selected (k)')
    plt.ylabel('Mean Cross-Validation Accuracy (on X_train_split subset)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axvline(x=best_num_features, color='red', linestyle='--', label=f'Best k = {best_num_features}')
    plt.legend()
    plt.show()

    return best_features_sfs_indices, best_num_features, sfs_scores

def perform_rfe_selection(X_train_split, y_train_split, X_val, y_val, best_num_features):
    num_features = X_train_split.shape[1]
    feature_names = [f'feature_{i}' for i in range(num_features)]

    print("\n--- Performing Recursive Feature Elimination (RFE) ---")

    rfe_estimator = RandomForestClassifier(n_estimators=50, random_state=0)
    rfe_selector = RFE(estimator=rfe_estimator,
                       n_features_to_select=best_num_features,
                       step=1,
                       verbose=0)

    rfe_selector.fit(X_train_split, y_train_split)

    rfe_selected_features_indices = rfe_selector.get_support(indices=True)
    rfe_selected_features_names = [feature_names[i] for i in rfe_selected_features_indices]

    print(f"RFE Selected feature names: {rfe_selected_features_names}")

    # Evaluate RFE results
    X_train_selected_rfe = X_train_split[:, rfe_selected_features_indices]
    X_val_selected_rfe = X_val[:, rfe_selected_features_indices]

    final_model_rfe = RandomForestClassifier(n_estimators=50, random_state=0)
    final_model_rfe.fit(X_train_selected_rfe, y_train_split)
    val_accuracy_rfe = final_model_rfe.score(X_val_selected_rfe, y_val)
    print(f"Validation Accuracy with RFE selected features: {val_accuracy_rfe:.4f}")

    return rfe_selected_features_indices

def perform_selectkbest_selection(X_train_split, y_train_split, X_val, y_val, best_num_features):
    num_features = X_train_split.shape[1]
    feature_names = [f'feature_{i}' for i in range(num_features)]

    print("\n--- Performing SelectKBest (Filter Method) ---")

    select_k_best = SelectKBest(score_func=f_classif, k=best_num_features) # Use the best k from SFS
    select_k_best.fit(X_train_split, y_train_split)

    filter_selected_features_indices = select_k_best.get_support(indices=True)
    filter_selected_features_names = [feature_names[i] for i in filter_selected_features_indices]

    print(f"SelectKBest Selected feature names: {filter_selected_features_names}")

    # Evaluate SelectKBest results
    X_train_selected_filter = X_train_split[:, filter_selected_features_indices]
    X_val_selected_filter = X_val[:, filter_selected_features_indices]

    final_model_filter = RandomForestClassifier(n_estimators=50, random_state=0)
    final_model_filter.fit(X_train_selected_filter, y_train_split)
    val_accuracy_filter = final_model_filter.score(X_val_selected_filter, y_val)
    print(f"Validation Accuracy with SelectKBest features: {val_accuracy_filter:.4f}")

    feature_scores = pd.DataFrame({'Feature': feature_names, 'Score': select_k_best.scores_})
    feature_scores = feature_scores.sort_values(by='Score', ascending=False)
    print("\nTop 10 Feature Scores (SelectKBest - f_classif):")
    print(feature_scores.head(10))

    return filter_selected_features_indices

def evaluate_feature_selection(X_train, y_train, X_test, y_test, max_k=None):
    from sklearn.feature_selection import chi2

    methods = {
        'ANOVA (f_classif)': f_classif,
        'Chi2': chi2,
        'Mutual Info': mutual_info_classif
    }

    if max_k is None:
        max_k = X_train.shape[1]

    # Scale features between 0 and 1 (required for Chi² and often helpful)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    k_range = range(1, max_k + 1)
    scores = {method: [] for method in methods}

    for method_name, score_func in methods.items():
        print(f"Evaluating {method_name}...")
        for k in k_range:
            selector = SelectKBest(score_func=score_func, k=k)
            X_train_k = selector.fit_transform(X_train_scaled, y_train)
            X_test_k = selector.transform(X_test_scaled)

            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train_k, y_train)
            y_pred = clf.predict(X_test_k)

            f1 = f1_score(y_test, y_pred, average='macro')
            scores[method_name].append(f1)

    return k_range, scores

def plot_f1_scores(k_range, scores):
    plt.figure(figsize=(10, 6))
    for method_name, f1_scores in scores.items():
        plt.plot(k_range, f1_scores, label=method_name)

    plt.xlabel("Number of Selected Features (k)")
    plt.ylabel("F1 Score (macro)")
    plt.title("Impact of Univariate Feature Selection on Classification Performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()