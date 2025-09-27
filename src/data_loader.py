# -*- coding: utf-8 -*-
"""Data loading and preprocessing utilities"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from config import IMAGE_DIR, MASK_DIR, EXCEL_FILE

def load_image_and_mask(image_id):
    # image_path = os.path.join(IMAGE_DIR, f"{image_id}.JPG")
    # mask_path = os.path.join(MASK_DIR, f"binary_{image_id}.tif")

    image_path = f"test_preprocessed/test_{image_id}.png"
    mask_path = f"test_preprocessed/masks/test_mask_{image_id}.png"

    image = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path).convert('L')) > 0

    return image, mask

def load_classification_data():
    df = pd.read_excel(EXCEL_FILE)

    bug_type_counts = df['bug type'].value_counts()
    bugs_to_keep = bug_type_counts[bug_type_counts > 1].index
    df = df[df['bug type'].isin(bugs_to_keep)]

    # # delete columns 'Hover fly', 'Wasp', 'Butterfly'
    # df = df[df['bug type'] != 'Hover fly']
    # df = df[df['bug type'] != 'Wasp']
    # df = df[df['bug type'] != 'Butterfly']

    df['bug type'] = df['bug type'].replace(['Hover fly', 'Wasp', 'Butterfly'], 'Others')

    return df