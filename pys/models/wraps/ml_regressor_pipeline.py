import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from pys.data_preperation.pre_train_process import replace_with_nan_based_on_rik

# Configuration details
config = {
    "KFold random_state": 42,
    "KFold n_splits": 5
}


def calculate_metrics(y_val, y_pred, x_val_selected, best_regressor, x_train_selected, y_train):
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    n = len(y_val)
    p = x_val_selected.shape[1]
    r2_adj = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    cv_scores = cross_val_score(best_regressor, x_train_selected, y_train,
                                cv=KFold(n_splits=config["KFold n_splits"], shuffle=True,
                                         random_state=config["KFold random_state"]), scoring='neg_mean_squared_error')
    cv_mean = -np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    return mse, r2, r2_adj, cv_mean, cv_std


def perform_search(x_train_selected, y_train, x_val_selected, y_val, model, model_name,
                   k, feature_selector_name, param_grid):
    search = GridSearchCV(estimator=model, param_grid=param_grid,
                          cv=KFold(n_splits=config["KFold n_splits"], shuffle=True,
                                   random_state=config["KFold random_state"]), scoring='neg_mean_squared_error',
                          return_train_score=True)

    search.fit(x_train_selected, y_train)
    best_regressor = search.best_estimator_
    y_pred = best_regressor.predict(x_val_selected)
    mse, r2, r2_adj, cv_mean, cv_std = calculate_metrics(y_val, y_pred, x_val_selected, best_regressor,
                                                         x_train_selected, y_train)

    results = [{
        "Model": model_name,
        "MSE": mse,
        "R2": r2,
        "R2 Adjusted": r2_adj,
        "CV MSE Mean": cv_mean,
        "CV MSE Std": cv_std,
        "Best Params": search.best_params_,
        "Number of Features": k,
        "Feature Selection Method": feature_selector_name
    }]

    print(f"Best Params: {search.best_params_}")

    return results, best_regressor


def evaluate_model_with_fs_optimized(x_train, y_train, x_val, y_val, feature_selector, k, model, model_name, param_grid):
    if feature_selector.__class__.__name__ in ['ExtraTreesRegressor']:
        selector = feature_selector.fit(x_train, y_train)
        feature_importances = selector.feature_importances_
        indices = np.argsort(feature_importances)[-k:]
    elif feature_selector.__class__.__name__ == 'SelectKBest':
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        selector = selector.fit(x_train, y_train)
        indices = selector.get_support(indices=True)
    elif feature_selector.__class__.__name__ in ['HistGradientBoostingRegressor']:
        selector = feature_selector.fit(x_train, y_train)
        result = permutation_importance(selector, x_train, y_train, n_repeats=10, random_state=42, n_jobs=2)
        importances = result.importances_mean
        selected_indices = np.argsort(importances)[::-1]
        indices = selected_indices[:k]

    x_train_selected = x_train.iloc[:, indices]
    x_val_selected = x_val.iloc[:, indices]

    print(f"Running broad search for {model_name} with {feature_selector.__class__.__name__} and k={k}")
    results, best_regressor = perform_search(x_train_selected, y_train,
                                             x_val_selected, y_val, model, model_name, k,
                                             feature_selector.__class__.__name__, param_grid)
    return results, best_regressor, x_train_selected, x_val_selected


def model_with_fs_optimized(df, model, model_name, results_dir, dataset_name, param_grid, score):
    df1 = replace_with_nan_based_on_rik(df)
    x = df1.drop(columns=["יבול-  (ק\"ג/ד')"])
    y = df1["יבול-  (ק\"ג/ד')"]

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=config["KFold random_state"])

    feature_selectors = [
        HistGradientBoostingRegressor(),
    ]
    ks = [5, 10, 20]

    results = []
    best_regressor_overall = None
    best_r2_score = -np.inf
    best_x_val_selected = None

    for feature_selector in feature_selectors:
        for k in ks:
            print(f"Running model {model_name} with feature selector {feature_selector.__class__.__name__} and k={k}")
            model_results, best_regressor, x_train_selected, x_val_selected = evaluate_model_with_fs_optimized(
                x_train, y_train, x_val, y_val, feature_selector, k, model, model_name, param_grid
            )
            results.extend(model_results)
            if model_results[-1][score] > best_r2_score:
                best_r2_score = model_results[-1][score]
                best_regressor_overall = best_regressor
                best_x_val_selected = x_val_selected

        # Save results to an Excel file
        results_df_optimized = pd.DataFrame(results)

        # Convert the config dictionary to a DataFrame
        config_df = pd.DataFrame(list(param_grid.items()), columns=['Configuration', 'Value'])

        feature_importances = best_regressor_overall.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': best_x_val_selected.columns,
            'Importance': feature_importances
                                     }).sort_values(by='Importance', ascending=False)

        # Write both the config DataFrame and the results DataFrame to the Excel file
        os.makedirs(results_dir, exist_ok=True)
        with pd.ExcelWriter(
                os.path.join(results_dir, f'results_with_config_{dataset_name}.xlsx')) as writer:
            config_df.to_excel(writer, sheet_name='Config', index=False)
            results_df_optimized.to_excel(writer, sheet_name='Results', index=False)
            importance_df.to_excel(writer, sheet_name='Feature Importances', index=False)

        print("Model training complete. Best model and results saved.")

        return best_regressor_overall, best_x_val_selected, y_val
