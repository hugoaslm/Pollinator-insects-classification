# -*- coding: utf-8 -*-
"""Image preprocessing functions"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure
from data_loader import load_image_and_mask, load_classification_data

def process_and_mask_image(image_id):
    image, mask = load_image_and_mask(image_id)

    # Computing the connected components of the binary mask
    labeled_mask = measure.label(mask)
    regions = measure.regionprops(labeled_mask)

    # Restricting the binary mask to its connected component of highest area
    max_area_region = max(regions, key=lambda r: r.area)
    max_area_mask = max_area_region.image

    # Restricting both mask and image to the bounding box of the cleaned binary mask
    cleaned_mask = max_area_mask

    # Get the bounding box coordinates from the region property
    min_row, min_col, max_row, max_col = max_area_region.bbox

    # Restricting both mask and image to the bounding box of the cleaned binary mask
    cleaned_image_region = image[min_row:max_row, min_col:max_col, :]

    cleaned_image = cleaned_image_region

    return image, mask, cleaned_image, cleaned_mask

def preprocess_test_images():
    # Load classification data
    classification_df = load_classification_data()

    # Get training image IDs (1-250)
    # get id from 251 to 347
    train_image_ids = [i for i in range(251, 348)]

    processed_images = {} # Dictionary to store results: {image_id: (original_image, original_mask, cleaned_image)}

    # Create the directories if they don't exist
    os.makedirs('test_preprocessed', exist_ok=True)
    os.makedirs('test_preprocessed/masks', exist_ok=True)

    for img_id in train_image_ids:
        original_img, original_mask, cleaned_img, cleaned_mask = process_and_mask_image(img_id)
        processed_images[img_id] = (original_img, original_mask, cleaned_img, cleaned_mask)

        # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        # axs[0].imshow(cleaned_mask)
        # axs[0].set_title(f'ID {img_id} Original Image')
        # axs[1].imshow(original_mask, cmap='gray') # Use gray colormap for mask
        # axs[1].set_title(f'ID {img_id} Original Mask')
        # axs[2].imshow(cleaned_img)
        # axs[2].set_title(f'ID {img_id} Cleaned Image (cropped to bbox)')
        #plt.show()

        # save only the cleaned images
        if cleaned_img is not None:
            cleaned_img_pil = Image.fromarray(cleaned_img)
            cleaned_img_pil.save(f'test_preprocessed/test_{img_id}.png')

        # save only the cleaned masks in binary
        if cleaned_mask is not None:
            cleaned_mask_uint8 = (cleaned_mask * 255).astype(np.uint8)
            cleaned_mask_pil = Image.fromarray(cleaned_mask_uint8, mode='L') # 'L' mode for grayscale
            cleaned_mask_pil.save(f'test_preprocessed/masks/test_mask_{img_id}.png')

    # save folder in zip
    shutil.make_archive('test_preprocessed', 'zip', 'test_preprocessed')

    print(f"\nFinished processing. Successfully processed {len(processed_images)} images.")
    
    return processed_images