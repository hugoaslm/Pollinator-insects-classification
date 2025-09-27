# -*- coding: utf-8 -*-
"""Machine Learning models training and evaluation"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from config import N_JOBS, RANDOM_STATE

def train_with_grid_search(model, param_grid, X_train, y_train):
    """Train model using Grid Search for hyperparameter optimization"""
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=N_JOBS)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

def train_with_randomized_search(model, param_distributions, X_train, y_train):
    """Train model using Randomized Search for hyperparameter optimization"""
    random_search = RandomizedSearchCV(model, param_distributions, n_iter=20, cv=5, scoring='accuracy', n_jobs=N_JOBS, random_state=RANDOM_STATE)
    random_search.fit(X_train, y_train)

    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")

    return random_search.best_estimator_

def evaluate_model(model, X_val, y_val, model_name):
    """Evaluate model performance on validation set"""
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"{model_name} Accuracy: {accuracy:.4f}")

    # Generate classification report
    report = classification_report(y_val, y_pred, output_dict=True)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_val, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y_val), yticklabels=np.unique(y_val))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')

    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'model': model
    }

def train_and_evaluate_ml_models(X_train, y_train, X_val, y_val):
    results = {}

    # 0. k-Nearest Neighbors
    print("Training k-Nearest Neighbors...")
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    }
    knn_model = train_with_grid_search(KNeighborsClassifier(), param_grid_knn, X_train, y_train)
    knn_eval = evaluate_model(knn_model, X_val, y_val, "k-Nearest Neighbors")
    results['knn'] = knn_eval

    # 1. Logistic Regression (non-ensemble, non-deep learning) ; change to decision trees or naives bayes
    print("Training decision trees ...")
    param_grid_dt = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    dt_model = train_with_randomized_search(DecisionTreeClassifier(), param_grid_dt, X_train, y_train)
    dt_eval = evaluate_model(dt_model, X_val, y_val, "Decision Trees")
    results['decision_trees'] = dt_eval

    # 2. SVM (non-ensemble, non-deep learning)
    print("Training SVM...")
    param_grid_svm = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }
    svm_model = train_with_grid_search(SVC(probability=True), param_grid_svm, X_train, y_train)
    svm_eval = evaluate_model(svm_model, X_val, y_val, "Support Vector Machine")
    results['svm'] = svm_eval

    # 3. Random Forest (ensemble learning)
    print("Training Random Forest...")
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_model = train_with_randomized_search(RandomForestClassifier(), param_grid_rf, X_train, y_train)
    rf_eval = evaluate_model(rf_model, X_val, y_val, "Random Forest")
    results['random_forest'] = rf_eval

    # 3.bis XGBoost (ensemble learning)
    print("Training XGBoost...")
    param_grid_xgb = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    xgb_model = train_with_randomized_search(XGBClassifier(), param_grid_xgb, X_train, y_train)
    xgb_eval = evaluate_model(xgb_model, X_val, y_val, "XGBoost")
    results['xgboost'] = xgb_eval

    # Logistic regression (supervised method studied in AI & optimization IG.2411)
    print("Training Logistic Regression...")
    param_grid_lr = {
        'C': [0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [1000]
    }
    lr_model = train_with_grid_search(LogisticRegression(), param_grid_lr, X_train, y_train)
    lr_eval = evaluate_model(lr_model, X_val, y_val, "Logistic Regression")
    results['logistic_regression'] = lr_eval

    # Find the best model based on validation accuracy
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = None

    if best_model_name == 'logistic_regression':
        best_model = lr_model
    elif best_model_name == 'svm':
        best_model = svm_model
    elif best_model_name == 'random_forest':
        best_model = rf_model
    elif best_model_name == 'decision_trees':
        best_model = dt_model
    elif best_model_name == 'xgboost':
        best_model = xgb_model
    elif best_model_name == 'knn':
        best_model = knn_model

    results['best_model'] = best_model
    results['best_model_name'] = best_model_name

    print(f"Best model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")

    return results