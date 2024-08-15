import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from pys.models.scores.feature_importance_analysis import calculate_and_save_feature_importance
from pys.models.wraps.ml_classifier_pipeline import model_with_fs_optimized

# Configuration details
config = {
    'max_depth': [None] + list(range(2, 21)),
    'min_samples_split': list(range(2, 21)),
    'min_samples_leaf': list(range(1, 16)),
}

def save_tree_plot(best_regressor, feature_names, results_dir, dataset_name):
    # Visualize the tree with limited depth
    plt.figure(figsize=(20, 10))
    plot_tree(best_regressor, feature_names=feature_names, filled=True, max_depth=3)
    filename = os.path.join(results_dir, f'tree_plot_{dataset_name}.png')
    plt.savefig(filename)
    plt.close()
    print(f"Tree plot saved to {filename}")


def run_dt_all_models():
    # Define paths
    data_dir = r'/data/models_sets/train'
    results_base_dir = r'/data/results'
    # Get the range for each from the predefine parameter grid to search on
    param_grid = {
        'max_depth': config['max_depth'],
        'min_samples_split': config['min_samples_split'],
        'min_samples_leaf': config['min_samples_leaf']
    }
    # Loop through each dataset in the data directory
    for filename in os.listdir(data_dir):
        if "reduced" in filename and "אחוזון" in filename and ("שחת" in filename or "שעורה" in filename):
            dataset_path = os.path.join(data_dir, filename)
            dataset_name = "DecisionTree_" + os.path.splitext(filename)[0]
            print(f'Running model {dataset_name}...')
            # Create a result directory for the current dataset
            results_dir = os.path.join(results_base_dir, dataset_name)
            os.makedirs(results_dir, exist_ok=True)
            # Data set load
            df_to_train = pd.read_excel(dataset_path)
            if "גרעינים" in filename and "יחס" not in filename:
                df_to_train = df_to_train.drop(columns=["יבול קש (ק\"ג/ד')"])
            print("Starting the full model evaluation and hyperparameter tuning pipeline...")
            # Call the generic ml classification flow, gets bask the best model and the suit data set
            best_model, x_val_selected, y_val = model_with_fs_optimized(df_to_train, DecisionTreeClassifier(),
                                                                        "DecisionTreeClassifier", results_dir,
                                                                        dataset_name, param_grid, 'ROC AUC')
            # Calculate the PDP's and the feature importance scores for the best model
            calculate_and_save_feature_importance(best_model, x_val_selected, y_val, results_dir,
                                                  dataset_name, 'predict_proba', 'auto')
            # Get the correct feature names
            feature_names = x_val_selected.columns
            save_tree_plot(best_model, feature_names, results_dir, dataset_name)

run_dt_all_models()
