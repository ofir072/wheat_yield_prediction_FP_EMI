import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from pys.models.scores.feature_importance_analysis import calculate_and_save_feature_importance
from pys.models.wraps.ml_regressor_pipeline import model_with_fs_optimized


# Configuration details
config = {
    'learning_rate': [round(x, 2) for x in np.arange(0.1, 0.52, 0.02)],
    'max_depth': [None] + list(range(2, 20)),
    'num_leaves': list(range(31, 51))
}


def run_lgb_all_models():
    # Define paths
    data_dir = r'/data/models_sets/train'
    results_base_dir = r'/data/results'

    param_grid = {
        'learning_rate': config['learning_rate'],
        'max_depth': config['max_depth'],
        'num_leaves': config['num_leaves']
    }

    # Loop through each dataset in the data directory
    for filename in os.listdir(data_dir):
        if "reduced" in filename and "חיטה" in filename:
            if "יחס" in filename or "שחת" in filename:
                dataset_path = os.path.join(data_dir, filename)
                dataset_name = "LightGBMRegressor_" + os.path.splitext(filename)[0]
                print(f'Runing model {dataset_name}...')

                # Create a result directory for the current dataset
                results_dir = os.path.join(results_base_dir, dataset_name)
                os.makedirs(results_dir, exist_ok=True)

                # Data set load
                df_to_train = pd.read_excel(dataset_path)
                if "גרעינים" in filename and "יחס" not in filename:
                    df_to_train = df_to_train.drop(columns=["יבול קש (ק\"ג/ד')"])
                print("Starting the full model evaluation and hyperparameter tuning pipeline...")

                best_model, x_val_selected, y_val = model_with_fs_optimized(df_to_train, lgb.LGBMRegressor(
                                                                            random_state=config["KFold random_state"],
                                                                            verbose=-1),
                                                                            "LightGBMRegressor", results_dir,
                                                                            dataset_name, param_grid, 'R2')
                calculate_and_save_feature_importance(best_model, x_val_selected, y_val, results_dir,
                                                      dataset_name, 'predict', 'auto')
            else:
                print(f'The current file: {filename} have missing value.')
                continue


run_lgb_all_models()
