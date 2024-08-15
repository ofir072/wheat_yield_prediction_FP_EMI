import os
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from tabulate import tabulate
import numpy as np


def suggest_variance_threshold(df, percentile=10):
    variances = df.var()
    threshold = variances.quantile(percentile / 100)
    return threshold


def remove_low_variance_features(df, threshold=0.0):
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df)
    features_to_keep = selector.get_support(indices=True)
    removed_features = df.columns[~selector.get_support()].tolist()
    df_reduced = df.iloc[:, features_to_keep]
    return df_reduced, removed_features, features_to_keep


def process_train_folder_by_variance_percentile_threshold(folder_path, threshold=None, percentile=10):
    features_to_keep_dict = {}
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return features_to_keep_dict
    train_summary = []
    removed_features_dict = {}  # Dictionary to store removed features for all datasets
    max_len = 0  # Maximum length of removed features lists
    for filename in os.listdir(folder_path):
        if filename.endswith("_train.xlsx"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_excel(file_path)
            if threshold is None:
                # Calculate the threshold for all features to find the 10% percentile to be removed
                threshold = suggest_variance_threshold(df, percentile)
                print(f"Suggested threshold for {filename}: {threshold}")
            # Remove the features that there all variance dont hold the threshold
            df_reduced, removed_features, features_to_keep = remove_low_variance_features(df, threshold)
            output_file_path = os.path.join(folder_path, "reduced_" + filename)
            df_reduced.to_excel(output_file_path, index=False)
            train_summary.append([filename, threshold, df.shape[1], df_reduced.shape[1]])
            # Save the feature that dropped to be applied on the test set
            features_to_keep_dict[filename] = features_to_keep
            # Update max_len if necessary
            max_len = max(max_len, len(removed_features))
            # Add removed features to the dictionary
            removed_features_dict[filename] = removed_features
    print(tabulate(train_summary, headers=["Filename", "Suggested Threshold", "Original Features", "Reduced Features"],
                   tablefmt="pretty", colalign=("center", "center", "center", "center")))
    # Pad removed features lists with NaN values to ensure all arrays have the same length
    for key in removed_features_dict:
        removed_features_dict[key] += [np.nan] * (max_len - len(removed_features_dict[key]))
    # Create DataFrame with removed features
    removed_features_df = pd.DataFrame(removed_features_dict)
    # Save removed features to an Excel file with each dataset's removed features in a separate column
    path = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\models_sets'
    removed_features_file_path = os.path.join(path, "removed_features.xlsx")
    removed_features_df.to_excel(removed_features_file_path, index=False)
    return features_to_keep_dict


def process_test_folder(folder_path, features_to_keep_dict):

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        if filename.endswith("_test.xlsx"):
            train_filename = filename.replace("_test.xlsx", "_train.xlsx")
            if train_filename in features_to_keep_dict:
                file_path = os.path.join(folder_path, filename)
                df = pd.read_excel(file_path)
                features_to_keep = features_to_keep_dict[train_filename]
                df_reduced = df.iloc[:, features_to_keep]
                output_file_path = os.path.join(folder_path, "reduced_" + filename)
                df_reduced.to_excel(output_file_path, index=False)
                print(f"Processed {filename} based on features from {train_filename}:")
                print(f"- Original number of features: {df.shape[1]}")
                print(f"- Number of features after reduction: {df_reduced.shape[1]}\n")


train_folder = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\models_sets\train'
test_folder = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\models_sets\test'
variance_threshold = None  # Set to None to use suggested threshold

print("Processing train folder...")
features_to_keep_dict = process_train_folder_by_variance_percentile_threshold(train_folder, threshold=variance_threshold)

print("Processing test folder...")
process_test_folder(test_folder, features_to_keep_dict)
