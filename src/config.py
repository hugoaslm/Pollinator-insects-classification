# -*- coding: utf-8 -*-
"""Configuration file for ML Optimization project"""

import os

# Paths
IMAGE_DIR = 'Pollinator-insects-classification/'
MASK_DIR = 'Pollinator-insects-classification/'
EXCEL_FILE = 'Pollinator-insects-classification/data/classif.xlsx'

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_JOBS = -1

# Feature extraction parameters
LBP_P = 8
LBP_R = 1
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, 3.14159/4, 3.14159/2, 3*3.14159/4]

# Clustering parameters
KMEANS_N_INIT = 10
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5