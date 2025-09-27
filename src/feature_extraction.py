# -*- coding: utf-8 -*-
"""Feature extraction utilities"""

import numpy as np
import cv2
from skimage import measure
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.cluster import KMeans
from data_loader import load_image_and_mask
from config import GLCM_DISTANCES, GLCM_ANGLES, LBP_P, LBP_R

# Function to extract shape features
def extract_shape_features(mask):
    labeled_mask = measure.label(mask)
    regions = measure.regionprops(labeled_mask)
    if not regions:
        return np.zeros(2)  # Return zeros if no regions found

    region = max(regions, key=lambda r: r.area)
    area = region.area
    perimeter = region.perimeter
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    eccentricity = region.eccentricity
    aspect_ratio = region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else 0
    solidity = region.solidity

    return np.array([area, perimeter, circularity, eccentricity, aspect_ratio, solidity])

# Function to extract texture features
def extract_texture_features(image, mask):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    masked_gray = gray_image * mask

    rows, cols = np.where(mask)
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    roi = masked_gray[min_row:max_row+1, min_col:max_col+1]

    if roi.max() > 0:
        roi_norm = (roi * 255 / roi.max()).astype(np.uint8)
        distances = GLCM_DISTANCES
        angles = GLCM_ANGLES
        glcm = graycomatrix(roi_norm, distances, angles, symmetric=True, normed=True)

        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()

    return np.array([homogeneity, energy])

def extract_color_features(image, mask):
    # Extract pixels belonging to the bug
    bug_pixels = image[mask]

    # Ratio feature
    bug_ratio = np.sum(bug_pixels) / mask.size
    bug_ratio = np.array([bug_ratio])

    # Extract features
    min_rgb = np.min(bug_pixels, axis=0) if bug_pixels.size > 0 else np.zeros(3)
    max_rgb = np.max(bug_pixels, axis=0) if bug_pixels.size > 0 else np.zeros(3)
    mean_rgb = np.mean(bug_pixels, axis=0) if bug_pixels.size > 0 else np.zeros(3)
    median_rgb = np.median(bug_pixels, axis=0) if bug_pixels.size > 0 else np.zeros(3)
    std_rgb = np.std(bug_pixels, axis=0) if bug_pixels.size > 0 else np.zeros(3)

    return np.concatenate([min_rgb, max_rgb, mean_rgb, median_rgb, std_rgb, bug_ratio])

def extract_hairiness_features(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    masked_gray = gray * mask

    # Apply LBP
    lbp = local_binary_pattern(masked_gray, P=LBP_P, R=LBP_R, method='uniform')
    lbp_values = lbp[mask > 0]

    lbp_hist, _ = np.histogram(lbp_values, bins=np.arange(0, 11), density=True)
    lbp_mean = np.mean(lbp_values)
    lbp_std = np.std(lbp_values)

    return np.concatenate([lbp_hist, [lbp_mean, lbp_std]])

def extract_color_clusters(image, mask, k=3):
    bug_pixels = image[mask]
    if bug_pixels.shape[0] < k:
        return np.zeros(k * 3)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(bug_pixels)
    return kmeans.cluster_centers_.flatten()

# Function to extract all features for a given image
def extract_all_features(image_id):
    image, mask = load_image_and_mask(image_id)

    shape_features = extract_shape_features(mask)
    texture_features = extract_texture_features(image, mask)
    color_features = extract_color_features(image, mask)

    return np.concatenate([color_features, texture_features, shape_features])

# Function to create a feature matrix for multiple images
def create_feature_matrix(image_ids):
    features_list = []
    for img_id in image_ids:
        features = extract_all_features(img_id)
        features_list.append(features)
        if img_id % 50 == 0:
            print(f"Image processed: {img_id}.JPG")

    return np.array(features_list)