import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
from sklearn.inspection import permutation_importance
from feature_importance_analysis import calculate_and_save_feature_importance
import os

# Configuration
CONFIG = {
    'dt_param_grid': {
        'max_depth': [None] + list(range(2, 3)),
        'min_samples_split': list(range(2, 3)),
        'min_samples_leaf': list(range(1, 2))
    },
    'hgb_param_grid': {
        'learning_rate': [round(x, 2) for x in np.arange(0.1, 0.52, 0.02)],
        'max_depth': [None] + list(range(2, 20)),
        'min_samples_leaf': list(range(1, 16))
    },
    'lgb_param_grid': {
        'learning_rate': [round(x, 2) for x in np.arange(0.1, 0.52, 0.02)],
        'max_depth': [None] + list(range(2, 20)),
        'num_leaves': list(range(31, 51))
    },
    'k_values': [5, 10, 20]
}


# Function to perform feature selection with HistGradientBoostingRegressor
def feature_selection_with_hgb(x_train, y_train, k):
    selector = HistGradientBoostingRegressor()
    selector.fit(x_train, y_train)
    result = permutation_importance(selector, x_train, y_train, n_repeats=10, random_state=42, n_jobs=2)
    importances = result.importances_mean
    indices = np.argsort(importances)[::-1]
    selected_indices = indices[:k]
    return selected_indices, importances


# Function to evaluate model with feature selection
def evaluate_model_with_fs(x, y, model, param_grid, k):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # Feature selection
    selected_features, importances = feature_selection_with_hgb(x_train, y_train, k)
    x_train_fs = x_train[:, selected_features]
    x_val_fs = x_val[:, selected_features]

    # GridSearchCV for model
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(x_train_fs, y_train)

    # Best estimator
    best_model = grid_search.best_estimator_

    # Predictions and metrics
    y_pred = best_model.predict(x_val_fs)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    n = x_val_fs.shape[0]
    p = x_val_fs.shape[1]
    r2_adj = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    cv_scores = cross_val_score(best_model, x[:, selected_features], y, cv=5, scoring='neg_mean_squared_error')
    cv_mean = -np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    return grid_search.best_params_, mse, r2, r2_adj, cv_mean, cv_std, selected_features, importances, best_model, x_val_fs, y_val


# Function to flatten the configuration dictionary for saving
def flatten_config(config):
    flat_config = []
    for key, subdict in config.items():
        if isinstance(subdict, dict):
            for subkey, value in subdict.items():
                flat_config.append({'Parameter': f"{key}.{subkey}", 'Value': value})
        else:
            flat_config.append({'Parameter': key, 'Value': subdict})
    return pd.DataFrame(flat_config)


# Main function to run the models with feature selection
def run_models_with_fs(df, k_values):
    x = df.drop(columns=["יבול-  (ק\"ג/ד')"]).values
    y = df["יבול-  (ק\"ג/ד')"].values

    dt_param_grid = CONFIG['dt_param_grid']
    hgb_param_grid = CONFIG['hgb_param_grid']
    lgb_param_grid = CONFIG['lgb_param_grid']

    models = [
        (DecisionTreeRegressor(random_state=42), dt_param_grid, "DecisionTreeRegressor"),
        (HistGradientBoostingRegressor(random_state=42), hgb_param_grid, "HistGradientBoostingRegressor"),
        (lgb.LGBMRegressor(random_state=42), lgb_param_grid, "LightGBMRegressor")
    ]

    best_model_info = None

    for model, param_grid, model_name in models:
        all_results = []
        for k in k_values:
            print(f"Running model {model_name} with feature selector HistGradientBoostingRegressor and k={k}")
            best_params, mse, r2, r2_adj, cv_mean, cv_std, selected_features, importances, best_model, x_val_selected, y_val = evaluate_model_with_fs(x, y, model, param_grid, k)
            result = (model_name, 'HistGradientBoostingRegressor', k, best_params, mse, r2, r2_adj, cv_mean, cv_std)
            all_results.append(result)

            # Check if this model is the best so far based on R²
            if best_model_info is None or r2 > best_model_info['r2']:
                best_model_info = {
                    'model_name': model_name,
                    'k': k,
                    'best_params': best_params,
                    'mse': mse,
                    'r2': r2,
                    'r2_adj': r2_adj,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'selected_features': selected_features,
                    'importances': importances,
                    'best_model': best_model
                }

        # Save results for each model to separate Excel files
        results_df = pd.DataFrame(all_results, columns=['Model', 'Feature Selector', 'K', 'Best Parameters', 'MSE', 'R2', 'R2 Adjusted', 'CV MSE Mean', 'CV MSE Std'])
        config_df = flatten_config(CONFIG)
        feature_importances_df = pd.DataFrame({'Feature': df.drop(columns=["יבול-  (ק\"ג/ד')"]).columns[selected_features], 'Importance': importances[selected_features]}).sort_values(by='Importance', ascending=False)

        filename = 'reduced_אפונה_בעל_train'
        results_base_dir = r'/data/results'
        dataset_name = f'{model_name}_' + os.path.splitext(filename)[0]

        # Create a result directory for the current dataset
        results_dir = os.path.join(results_base_dir, dataset_name)
        os.makedirs(results_dir, exist_ok=True)

        file_name = results_dir + f'{dataset_name}_results.xlsx'
        with pd.ExcelWriter(file_name) as writer:
            results_df.to_excel(writer, sheet_name='All Results', index=False)
            feature_importances_df.to_excel(writer, sheet_name='Feature Importance', index=False)
            config_df.to_excel(writer, sheet_name='Config', index=False)

        dataset_name = f'{model_name}' + os.path.splitext(filename)[0]
        print(f'Runing model {dataset_name}...')

        # Create a result directory for the current dataset
        results_dir = os.path.join(results_base_dir, dataset_name)
        os.makedirs(results_dir, exist_ok=True)

        calculate_and_save_feature_importance(best_model, x_val_selected, y_val, results_dir, dataset_name)

    return best_model_info


# Load data
location_file = r'/data/models_sets/train/reduced_אפונה_בעל_train.xlsx'
df_to_train = pd.read_excel(location_file)

# Run models with feature selection
best_model_info = run_models_with_fs(df_to_train, CONFIG['k_values'])
print("Best model info:", best_model_info)
