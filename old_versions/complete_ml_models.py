from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pys.data_preperation.pre_train_process import replace_with_nan_based_on_rik
import os


def evaluate_model_with_fs(x_train, y_train, x_val, y_val, feature_selector, k, model, param_grid, model_name, output_dir):
    if feature_selector.__class__.__name__ in ['ExtraTreesRegressor']:
        selector = feature_selector.fit(x_train, y_train)
        feature_importances = selector.feature_importances_
        indices = np.argsort(feature_importances)[-k:]
    elif feature_selector.__class__.__name__ == 'SelectKBest':
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        selector = selector.fit(x_train, y_train)
        indices = selector.get_support(indices=True)

    x_train_selected = x_train.iloc[:, indices]
    x_val_selected = x_val.iloc[:, indices]

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_search.fit(x_train_selected, y_train)

    results = []

    best_regressor = grid_search.best_estimator_
    y_pred = best_regressor.predict(x_val_selected)

    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    n = len(y_val)
    p = x_val_selected.shape[1]
    r2_adj = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    cv_scores = cross_val_score(best_regressor, x_train_selected, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_mean = -np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    results.append({
            "Model": model_name,
            "Feature Selector": feature_selector.__class__.__name__,
            "K": k,
            "MSE": mse,
            "R2": r2,
            "R2 Adjusted": r2_adj,
            "CV MSE Mean": cv_mean,
            "CV MSE Std": cv_std
        })

    # Specific actions for DecisionTreeRegressor
    if isinstance(best_regressor, DecisionTreeRegressor):
        # Save the plot of the tree
        plt.figure(figsize=(20, 10))
        plot_tree(best_regressor, filled=True, feature_names=x_train_selected.columns, rounded=True)
        tree_plot_path = os.path.join(output_dir, f"{model_name}_tree_plot_k_{k}.png")
        plt.savefig(tree_plot_path)
        plt.close()
        print(f"Decision tree plot saved to {tree_plot_path}")

        # Save the feature importances
        feature_importances = best_regressor.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': x_train_selected.columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
        feature_importance_path = os.path.join(output_dir, f"{model_name}_feature_importances_k_{k}.xlsx")
        feature_importance_df.to_excel(feature_importance_path, index=False)
        print(f"Feature importances saved to {feature_importance_path}")

    return results


def model_with_fs(df, model, param_grid, model_name, output_dir):
    df1 = replace_with_nan_based_on_rik(df)
    x = df1.drop(columns=["יבול-  (ק\"ג/ד')"])
    y = df1["יבול-  (ק\"ג/ד')"]

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    feature_selectors = [
        ExtraTreesRegressor(n_estimators=100, random_state=42),
        SelectKBest(score_func=mutual_info_regression, k=10)
    ]
    ks = [5, 10, 20]

    results = []
    for feature_selector in feature_selectors:
        for k in ks:
            print(f"Running model {model_name} with feature selector {feature_selector.__class__.__name__} and k={k}")
            model_results = evaluate_model_with_fs(
                x_train, y_train, x_val, y_val, feature_selector, k, model, param_grid, model_name, output_dir
            )
            results.extend(model_results)

    return results


location_file = r'/data/models_sets/train/reduced_אפונה_בעל_train.xlsx'
df_to_train = pd.read_excel(location_file)

output_dir = r'/data/results'  # Set your output directory here

# DecisionTreeRegressor
dt_param_grid = {
    'max_depth': list(range(2, 31)) + [None],
    'min_samples_split': list(range(2, 21)),
    'min_samples_leaf': list(range(1, 18))
}
best_dt_results = model_with_fs(df_to_train, DecisionTreeRegressor(random_state=42), dt_param_grid, "DecisionTreeRegressor", output_dir)

# SVR
svr_param_grid = {
    'C': list(np.linspace(0.1, 150, num=20)),
    'epsilon': [0.01, 0.1, 1, 2, 5],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}
best_svr_results = model_with_fs(df_to_train, SVR(), svr_param_grid, "SVR", output_dir)

# MLPRegressor (ANN)
ann_param_grid = {
    'hidden_layer_sizes': [(x,) for x in range(30, 151, 20)] + [(x, x) for x in range(30, 151, 20)] + [(x, x, x) for x in range(30, 151, 20)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
    'learning_rate': ['constant', 'adaptive']
}
best_ann_results = model_with_fs(df_to_train, MLPRegressor(max_iter=1000, random_state=42), ann_param_grid, "MLPRegressor", output_dir)

# Combine all results
all_results = best_dt_results + best_svr_results + best_ann_results

# Convert results to a DataFrame
results_df = pd.DataFrame(all_results)

# Save results to an Excel file
results_df_path = os.path.join(output_dir, 'model_שעורה_results.xlsx')
results_df.to_excel(results_df_path, index=False)
print(f"Results saved to {results_df_path}")
