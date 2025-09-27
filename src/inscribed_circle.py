# -*- coding: utf-8 -*-
"""Best inscribed circle computation"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize
from matplotlib.patches import Circle
from PIL import Image
import os
import pandas as pd
from data_loader import load_classification_data

# Function to load images and masks
def load_preprocessed_image_and_mask(image_id):
    image_path = os.path.join("Pollinator-insects-classification/preprocessed", f"preprocessed_{image_id}.png")
    mask_path = os.path.join("Pollinator-insects-classification/preprocessed/masks", f"preprocessed_mask_{image_id}.png")

    image = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path).convert('L')) > 0

    return image, mask

def loss_function(params, mask_data, dist_map_data):
    cx, cy, r = params
    h, w = mask_data.shape

    # Radius must be non-negative
    if r < 0:
        return np.inf

    # Center must be within image bounds.
    if not (0 <= cx < w and 0 <= cy < h):
        return np.inf

    # Round and convert center to integer coordinates for mask/dist_map indexing
    int_cx = int(np.round(cx))
    int_cy = int(np.round(cy))
    int_cx = np.clip(int_cx, 0, w - 1)
    int_cy = np.clip(int_cy, 0, h - 1)

    # Calculate objective (maximize r -> minimize -r)
    objective = -r

    # --- Penalty for circle points outside the mask (circumference check) ---
    num_points = 100
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    # Convert circle coordinates to pixel indices
    circle_x = np.round(cx + r * np.cos(theta)).astype(int)
    circle_y = np.round(cy + r * np.sin(theta)).astype(int)

    # Check if any part of the circle is outside image boundaries
    out_of_bounds = (circle_x < 0) | (circle_x >= w) | \
                    (circle_y < 0) | (circle_y >= h)
    if np.any(out_of_bounds):
        return np.inf

    # Check if sampled points on circumference are within the mask
    points_outside_mask = ~mask_data[circle_y, circle_x]

    if np.any(points_outside_mask):
        return np.inf

    # --- Incorporating EDT for better guidance (lighter penalty) ---
    max_r_at_current_center = dist_map_data[int_cy, int_cx]

    if r > max_r_at_current_center:
        penalty = (r - max_r_at_current_center) * 1000
        return objective + penalty
    else:
        return objective

def compute_inscribed_circles():
    # Load classification data
    classification_df = load_classification_data()

    # Get training image IDs (1-250)
    train_image_ids = classification_df['ID'].unique()

    circle_features = []  # [image_id, cx, cy, r] for each mask

    for image_id in train_image_ids:
        print(f"\nProcessing image ID: {image_id}")

        # Load image and mask for the current ID
        image, mask = load_preprocessed_image_and_mask(image_id)
        h, w = mask.shape

        print(f"Mask loaded with shape: {mask.shape}")

        ### Step 1: Initialization Relying on the Centroid

        # Calculate the centroid of the loaded mask
        y_coords, x_coords = np.where(mask)
        if len(x_coords) == 0:
            print(f"Mask {image_id} is empty. Skipping.")
            circle_features.append([image_id, 0, 0, 0])
            continue

        initial_cx = np.mean(x_coords)
        initial_cy = np.mean(y_coords)

        # Calculate the Euclidean Distance Transform (EDT) of the mask
        dist_map = distance_transform_edt(mask)

        # Initial radius: Use the EDT value at the centroid.
        initial_cy_int = int(round(initial_cy))
        initial_cx_int = int(round(initial_cx))

        # Clip to ensure valid indices in case centroid is exactly on boundary
        initial_cy_int = np.clip(initial_cy_int, 0, h - 1)
        initial_cx_int = np.clip(initial_cx_int, 0, w - 1)

        initial_r = dist_map[initial_cy_int, initial_cx_int]
        if initial_r == 0:
            initial_r = 1

        initial_params = [initial_cx, initial_cy, initial_r]

        #### Initializatio not relying on the centroid of the mask
        distance_map = distance_transform_edt(mask)
        initial_radius = np.max(distance_map)
        max_pos = np.unravel_index(np.argmax(distance_map), distance_map.shape)
        initial_center = (max_pos[1], max_pos[0])
        initial_params = np.array(initial_center + (initial_radius,))
        print(initial_params)

        print(f"Initial centroid: ({initial_cx:.2f}, {initial_cy:.2f})")
        print(f"Initial radius (at centroid): {initial_r:.2f}")

        # Bounds for the optimization parameters
        bounds = [(0, w - 1), (0, h - 1), (0, min(h, w) / 2)]

        print("\n--- Starting Optimization ---")
        result = minimize(
            fun=loss_function,
            x0=initial_params,
            args=(mask, dist_map),
            method='Nelder-Mead',
            bounds=bounds,
            options={'disp': True, 'return_all': True}
        )

        best_cx, best_cy, best_r = result.x
        best_r = max(0, best_r)

        circle_features.append([image_id, best_cx, best_cy, best_r])

        print("\n--- Optimization Results ---")
        print(f"Optimization Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Optimal center: ({best_cx:.2f}, {best_cy:.2f})")
        print(f"Optimal radius: {best_r:.2f}")
        print(f"Final loss value: {result.fun:.2f}")

        # --- Visualization ---
        plt.figure(figsize=(6, 4))
        plt.imshow(mask, cmap='gray', alpha=0.7)
        plt.title(f'Best Inscribed Circle in Insect Mask - Image ID: {image_id}')
        plt.axis('off')

        plt.plot(initial_cx, initial_cy, 'gx', markersize=10, label='Initial Centroid')
        plt.plot(best_cx, best_cy, 'rx', markersize=10, label='Optimal Center')

        initial_circle = Circle(
            (initial_cx, initial_cy), initial_r,
            color='blue', fill=False, linestyle='--', linewidth=1, label='Initial Circle'
        )
        plt.gca().add_patch(initial_circle)

        best_circle = Circle(
            (best_cx, best_cy), best_r,
            color='red', fill=False, linewidth=2, label='Best Inscribed Circle'
        )
        plt.gca().add_patch(best_circle)

        plt.legend()
        plt.tight_layout()
        plt.show()

    circle_features_array = np.array(circle_features)
    np.save("inscribed_circle_features_3_class.npy", circle_features_array)
    
    return circle_features_array