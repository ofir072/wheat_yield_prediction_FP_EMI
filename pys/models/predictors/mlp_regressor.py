import os
import pandas as pd
from pys.models.scores.feature_importance_analysis import calculate_and_save_feature_importance
from sklearn.neural_network import MLPRegressor
from pys.models.wraps.ml_regressor_pipeline import model_with_fs_optimized

# Configuration details
config = {
    'hidden_layer_sizes': [(x,) for x in range(30, 151, 20)] + [(x, x) for x in range(30, 151, 20)] + [(x, x, x) for x in range(30, 151, 20)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
    'learning_rate': ['constant', 'adaptive'],
}


def run_mlp_all_models():

    # Define paths
    data_dir = r'/data/models_sets/train'
    results_base_dir = r'/data/results'

    param_grid = {
        'hidden_layer_sizes': config['hidden_layer_sizes'],
        'activation': config['activation'],
        'solver': config['solver'],
        'alpha': config['alpha'],
        'learning_rate': config['learning_rate']
    }

    # Loop through each dataset in the data directory
    for filename in os.listdir(data_dir):
        if "reduced" in filename:
            if "חיטה" not in filename:
                dataset_path = os.path.join(data_dir, filename)
                dataset_name = "MLP_" + os.path.splitext(filename)[0]
                print(f'Runing model {dataset_name}...')

                # Create a result directory for the current dataset
                results_dir = os.path.join(results_base_dir, dataset_name)
                os.makedirs(results_dir, exist_ok=True)

                # Data set load
                df_to_train = pd.read_excel(dataset_path)
                if "גרעינים" in filename:
                    df_to_train = df_to_train.drop(columns=["יבול קש (ק\"ג/ד')"])
                print("Starting the full model evaluation and hyperparameter tuning pipeline...")

                best_model, x_val_selected, y_val = model_with_fs_optimized(df_to_train,
                                                                            MLPRegressor(max_iter=1000, random_state=42),
                                                                            "MLP", results_dir, dataset_name,
                                                                            param_grid, 'R2')
                calculate_and_save_feature_importance(best_model, x_val_selected, y_val, results_dir,
                                                      dataset_name, 'predict', 'auto')
            else:
                print(f'The current file: {filename} have missing value.')
                continue


run_mlp_all_models()
