# -*- coding: utf-8 -*-
"""Symmetry plane computation"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from data_loader import load_classification_data
from inscribed_circle import load_preprocessed_image_and_mask
from tqdm import tqdm

def rotate_image(theta_degree, xc, yc, arr):
    theta_degree = float(theta_degree)

    h, w = arr.shape

    # Get the rotation matrix for calculating the new bounds
    M_temp = cv2.getRotationMatrix2D((xc, yc), theta_degree, 1.0)

    # Apply rotation to the four corners of the image to find new dimensions
    corners = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]], dtype=np.float32)
    corners_homog = np.c_[corners, np.ones(4)]
    rotated_corners = (M_temp @ corners_homog.T).T

    min_x = np.min(rotated_corners[:, 0])
    max_x = np.max(rotated_corners[:, 0])
    min_y = np.min(rotated_corners[:, 1])
    max_y = np.max(rotated_corners[:, 1])

    new_w = int(np.ceil(max_x - min_x))
    new_h = int(np.ceil(max_y - min_y))

    # Calculate translation to move top-left corner to (0,0) in new image
    tx = -min_x
    ty = -min_y

    # Adjust the rotation matrix to include this translation
    M_adjusted = cv2.getRotationMatrix2D((xc, yc), theta_degree, 1.0)
    M_adjusted[0, 2] += tx # Add translation to x
    M_adjusted[1, 2] += ty # Add translation to y

    # Calculate the new coordinates of the rotation center (xc, yc) in the *output* image
    original_center_homog = np.array([[xc, yc, 1]], dtype=np.float32).T # Column vector
    transformed_center = M_adjusted @ original_center_homog
    new_xc = transformed_center[0, 0]
    new_yc = transformed_center[1, 0]

    # Apply the rotation with the adjusted matrix
    original_dtype = arr.dtype
    rot_arr = cv2.warpAffine(arr.astype(np.float32), M_adjusted, (new_w, new_h),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if original_dtype == bool:
        rot_arr = rot_arr > 0.5

    return rot_arr, new_xc, new_yc

# --- Symmetric Image Creation Function ---
def create_symmetric_image(image_arr, symmetry_center_x):
    h, w = image_arr.shape
    symmetric_arr = np.zeros_like(image_arr, dtype=image_arr.dtype)

    for dest_x in range(w):
        src_x = int(round(2 * symmetry_center_x - dest_x)) # Calculate source x-coordinate

        if 0 <= src_x < w:
            symmetric_arr[:, dest_x] = image_arr[:, src_x]

    return symmetric_arr

# --- Symmetry Loss Function ---
def symmetry_loss_function(theta_degree_param, original_mask, rotation_center_x_orig, rotation_center_y_orig):
    theta_degree = theta_degree_param[0]

    # Rotate the original mask and get the new center coordinates
    rotated_mask_temp, new_xc, new_yc = rotate_image(theta_degree, rotation_center_x_orig, rotation_center_y_orig, original_mask.astype(np.float32))
    rotated_mask_binary = rotated_mask_temp > 0.5

    # Create the symmetric version of the rotated mask using the new x-coordinate of the center
    symmetric_rotated_mask = create_symmetric_image(rotated_mask_binary, new_xc)

    # Calculate  symmetry loss
    loss = np.sum(np.abs(rotated_mask_binary.astype(int) - symmetric_rotated_mask.astype(int)))

    return loss

def compute_symmetry_planes():
    # load inscribed npy
    circle_features = np.load("inscribed_circle_features_new.npy")
    circle_features_array = np.array(circle_features)

    # Load classification data
    classification_df = load_classification_data()

    # Get training image IDs (1-250)
    train_image_ids = classification_df['ID'].unique()

    # --- Main Processing Loop ---
    all_image_features = []

    for image_id in train_image_ids:
        print(f"\nProcessing image ID: {image_id}")

        current_image_features = circle_features_array[circle_features_array[:, 0] == image_id]
        if len(current_image_features) == 0:
            print(f"No inscribed circle features found for image ID: {image_id}. Skipping.")
            continue

        _, best_cx, best_cy, best_r = current_image_features[0]
        print(f"Retrieved inscribed circle: Center=({best_cx:.2f}, {best_cy:.2f}), Radius={best_r:.2f}")

        image, mask = load_preprocessed_image_and_mask(image_id)
        if mask is None:
            print(f"Could not load mask for image ID: {image_id}. Skipping.")
            continue
        mask = mask.astype(bool) # Ensure mask is boolean

        # --- Improved Initialization for Symmetry Optimization ---
        print("--- Initializing Symmetry Search ---")

        # Candidate initial angles to test
        candidate_angles = np.array(range(0, 181, 20))

        best_prelim_loss = np.inf
        best_prelim_angle = 0.0

        for angle in candidate_angles:
            loss = symmetry_loss_function([angle], mask, best_cx, best_cy)
            print(f"  Candidate angle: {angle:.1f}°, Preliminary Loss: {loss:.2f}")
            if loss < best_prelim_loss:
                best_prelim_loss = loss
                best_prelim_angle = angle

        initial_theta_for_minimize = best_prelim_angle
        print(f"Selected best preliminary angle: {initial_theta_for_minimize:.1f}° with loss: {best_prelim_loss:.2f}")

        # --- Symmetry Plane Optimization (using the best preliminary angle) ---

        print("--- Starting Full Symmetry Optimization ---")
        symmetry_result = minimize(
            fun=symmetry_loss_function,
            x0=[initial_theta_for_minimize],
            args=(mask, best_cx, best_cy),
            method='Nelder-Mead',
            options={'disp': True, 'return_all': True}
        )

        optimal_theta = symmetry_result.x[0]

        print(f"Symmetry Optimization Success: {symmetry_result.success}")
        print(f"Message: {symmetry_result.message}")
        print(f"Optimal symmetry angle: {optimal_theta:.2f} degrees")
        print(f"Final symmetry loss: {symmetry_result.fun:.2f}")

        # 4. Store results
        all_image_features.append([image_id, best_cx, best_cy, best_r, optimal_theta])

        # --- Visualization for Symmetry ---
        rotated_for_symmetry_mask, final_rotated_cx, final_rotated_cy = rotate_image(optimal_theta, best_cx, best_cy, mask.astype(np.float32))
        rotated_for_symmetry_mask = rotated_for_symmetry_mask > 0.5

        plt.figure(figsize=(14, 7))

        # Original mask with inscribed circle
        plt.subplot(1, 3, 1)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Original Mask (ID: {image_id})')
        from matplotlib.patches import Circle
        circle_patch = Circle((best_cx, best_cy), best_r, color='red', fill=False, linewidth=2)
        plt.gca().add_patch(circle_patch)
        plt.plot(best_cx, best_cy, 'rx', markersize=10, label='Center')
        plt.axis('off')

        # Rotated mask with symmetry axis
        plt.subplot(1, 3, 2)
        plt.imshow(rotated_for_symmetry_mask, cmap='gray')
        plt.axvline(x=final_rotated_cx, color='lime', linestyle='--', linewidth=2, label='Symmetry Axis')
        # Plot the transformed center point itself
        plt.plot(final_rotated_cx, final_rotated_cy, 'gx', markersize=10, label='Rotated Center')
        plt.title(f'Rotated Mask ({optimal_theta:.1f}°)')
        plt.axis('off')
        plt.legend()

        # Symmetric version of rotated mask
        symmetric_rotated_mask_vis = create_symmetric_image(rotated_for_symmetry_mask, final_rotated_cx)
        plt.subplot(1, 3, 3)
        plt.imshow(symmetric_rotated_mask_vis, cmap='gray')
        plt.axvline(x=final_rotated_cx, color='lime', linestyle='--', linewidth=2, label='Symmetry Axis')
        plt.plot(final_rotated_cx, final_rotated_cy, 'gx', markersize=10, label='Rotated Center')
        plt.title('Symmetric Mask')
        plt.axis('off')
        plt.legend()

        plt.tight_layout()
        plt.show()

    extra_features_array = np.array(all_image_features)
    np.save("extra_features_3_class.npy", extra_features_array)
    
    return extra_features_array