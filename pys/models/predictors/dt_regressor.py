import os
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from pys.models.scores.feature_importance_analysis import calculate_and_save_feature_importance
from pys.models.wraps.ml_regressor_pipeline import model_with_fs_optimized

# Configuration details
config = {
    'max_depth': [None],
    'min_samples_split': [2],
    'min_samples_leaf': [2]
}


def run_dt_all_models():

    # Define paths
    data_dir = r'/data/models_sets/train'
    results_base_dir = r'/data/results'

    param_grid = {
        'max_depth': config['max_depth'],
        'min_samples_split': config['min_samples_split'],
        'min_samples_leaf': config['min_samples_leaf']
    }

    # Loop through each dataset in the data directory
    for filename in os.listdir(data_dir):
        if "reduced" in filename and "אחוזן" not in filename:
            dataset_path = os.path.join(data_dir, filename)
            dataset_name = "DecisionTree_" + os.path.splitext(filename)[0]
            print(f'Runing model {dataset_name}...')

            # Create a result directory for the current dataset
            results_dir = os.path.join(results_base_dir, dataset_name)
            os.makedirs(results_dir, exist_ok=True)

            # Data set load
            df_to_train = pd.read_excel(dataset_path)
            if "גרעינים" in filename and "יחס" not in filename:
                df_to_train = df_to_train.drop(columns=["יבול קש (ק\"ג/ד')"])
            print("Starting the full model evaluation and hyperparameter tuning pipeline...")

            best_model, x_val_selected, y_val = model_with_fs_optimized(df_to_train, DecisionTreeRegressor(),
                                                                        "DecisionTreeRegressor", results_dir,
                                                                        dataset_name, param_grid, 'R2')
            calculate_and_save_feature_importance(best_model, x_val_selected, y_val, results_dir,
                                                  dataset_name, 'predict', 'auto')


run_dt_all_models()
