import os
import numpy as np
import pandas as pd
from pys.models.scores.feature_importance_analysis import calculate_and_save_feature_importance
from sklearn.svm import SVR
from pys.models.wraps.ml_regressor_pipeline import model_with_fs_optimized

# Configuration details
config = {
    'C': list(np.linspace(0.01, 200, num=30)),  # Broadened range from 0.01 to 200 with 30 values
    'epsilon': list(np.linspace(0.001, 10, num=15)),  # Broadened range from 0.001 to 10 with 15 values
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
}


def run_svr_all_models():
    # Define paths
    data_dir = r'/data/models_sets/train'
    results_base_dir = r'/data/results'

    param_grid = {
        'C': config['C'],
        'epsilon': config['epsilon'],
        'kernel': config['kernel'],
    }

    # Loop through each dataset in the data directory
    for filename in os.listdir(data_dir):
        if "reduced" in filename:
            if "חיטה" not in filename:
                dataset_path = os.path.join(data_dir, filename)
                dataset_name = "SVR_" + os.path.splitext(filename)[0]
                print(f'Runing model {dataset_name}...')

                # Create a result directory for the current dataset
                results_dir = os.path.join(results_base_dir, dataset_name)
                os.makedirs(results_dir, exist_ok=True)

                # Data set load
                df_to_train = pd.read_excel(dataset_path)
                if "גרעינים" in filename:
                    df_to_train = df_to_train.drop(columns=["יבול קש (ק\"ג/ד')"])
                print("Starting the full model evaluation and hyperparameter tuning pipeline...")

                best_model, x_val_selected, y_val = model_with_fs_optimized(df_to_train, SVR(),
                                                                            "SVR", results_dir,
                                                                            dataset_name, param_grid, 'R2')
                calculate_and_save_feature_importance(best_model, x_val_selected, y_val, results_dir,
                                                      dataset_name, 'predict', 'auto')

            else:
                print(f'The current file: {filename} have missing value.')
                continue


run_svr_all_models()
