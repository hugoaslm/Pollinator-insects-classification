# -*- coding: utf-8 -*-
"""Utility functions"""

import pandas as pd
import pickle

def create_submission_file(test_image_ids, predictions):
    """Create CSV submission file"""
    submission_df = pd.DataFrame({
        'ID': test_image_ids,
        'bug type': predictions
    })

    submission_df.to_csv('submission.csv', index=False)
    print("Submission file created successfully.")

def save_results(results, filename):
    """Save results to pickle file"""
    with open(filename, 'wb') as file:
        pickle.dump(results, file)
    print(f"Results saved to {filename}")

def load_results(filename):
    """Load results from pickle file"""
    with open(filename, 'rb') as file:
        results = pickle.load(file)
    print(f"Results loaded from {filename}")
    return results