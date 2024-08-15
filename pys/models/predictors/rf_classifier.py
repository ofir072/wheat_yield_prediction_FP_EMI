import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
from pys.data_preperation.pre_train_process import replace_with_nan_based_on_rik
from pys.models.scores.feature_importance_analysis import calculate_and_save_feature_importance
from pys.models.wraps.ml_classifier_pipeline import extract_hebrew_and_underscores_exclude, plot_class_distribution

# Configuration details
config = {
    "RandomizedSearchCV n_iter": 75,
    "KFold random_state": 42,
    "KFold n_splits": 5
}

# RandomForestClassifier initial parameter grid
rf_initial_param_grid = {
        'n_estimators': list(range(50, 200, 10)),
        'max_depth': list(range(2, 21)) + [None],
        'min_samples_split': list(range(2, 21)),
        'min_samples_leaf': list(range(1, 16))
}


def calculate_metrics(y_val, y_pred, x_val_selected, best_classifier, x_train_selected, y_train):
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_val, best_classifier.predict_proba(x_val_selected)[:, 1])

    cv_scores = cross_val_score(best_classifier, x_train_selected, y_train,
                                cv=KFold(n_splits=config["KFold n_splits"], shuffle=True,
                                         random_state=config["KFold random_state"]), scoring='roc_auc')
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    return accuracy, f1, roc_auc, cv_mean, cv_std


def perform_search(x_train_selected, y_train, x_val_selected, y_val, model, param_grid, model_name, run_detail, k,
                   feature_selector_name, search_type='grid'):
    if search_type == 'grid':
        search = GridSearchCV(estimator=model, param_grid=param_grid,
                              cv=KFold(n_splits=config["KFold n_splits"], shuffle=True,
                                       random_state=config["KFold random_state"]), scoring='neg_mean_squared_error',
                              return_train_score=True)
    else:
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                                    n_iter=config["RandomizedSearchCV n_iter"],
                                    cv=KFold(n_splits=config["KFold n_splits"], shuffle=True,
                                             random_state=config["KFold random_state"]),
                                    scoring='neg_mean_squared_error', return_train_score=True,
                                    random_state=config["KFold random_state"])

    search.fit(x_train_selected, y_train)
    best_classifier = search.best_estimator_
    y_pred = best_classifier.predict(x_val_selected)
    accuracy, f1, roc_auc, cv_mean, cv_std = calculate_metrics(y_val, y_pred, x_val_selected, best_classifier,
                                                               x_train_selected, y_train)

    results = [{
        "Model": model_name,
        "Run Detail": run_detail,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "ROC AUC": roc_auc,
        "CV ROC AUC Mean": cv_mean,
        "CV ROC AUC Std": cv_std,
        "Best Params": search.best_params_,
        "Number of Features": k,
        "Feature Selection Method": feature_selector_name
    }]

    print(f"Best Params: {search.best_params_}")

    return results, best_classifier, search.best_params_


def generate_param_grid(best_params, n_estimators_range, max_depth_range, min_samples_split_range,
                        min_samples_leaf_range):
    further_refined_rf_param_grid = {
        'n_estimators': list(range(max(50, best_params['n_estimators'] - n_estimators_range),
                                   min(200, best_params['n_estimators'] + n_estimators_range), 2)),
        'max_depth': [best_params['max_depth']] if best_params['max_depth'] is None else list(
            range(max(2, best_params['max_depth'] - max_depth_range),
                  min(30, best_params['max_depth'] + max_depth_range))),
        'min_samples_split': list(range(max(2, best_params['min_samples_split'] - min_samples_split_range),
                                        min(20, best_params['min_samples_split'] + min_samples_split_range))),
        'min_samples_leaf': list(range(max(1, best_params['min_samples_leaf'] - min_samples_leaf_range),
                                       min(16, best_params['min_samples_leaf'] + min_samples_leaf_range)))
    }
    return further_refined_rf_param_grid


def evaluate_model_with_fs_optimized(x_train, y_train, x_val, y_val, feature_selector, k, model, initial_param_grid,
                                     model_name):
    if feature_selector.__class__.__name__ in ['ExtraTreesClassifier']:
        selector = feature_selector.fit(x_train, y_train)
        feature_importances = selector.feature_importances_
        indices = np.argsort(feature_importances)[-k:]
    elif feature_selector.__class__.__name__ == 'SelectKBest':
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        selector = selector.fit(x_train, y_train)
        indices = selector.get_support(indices=True)
    elif feature_selector.__class__.__name__ in ['HistGradientBoostingClassifier']:
        selector = feature_selector.fit(x_train, y_train)
        result = permutation_importance(selector, x_train, y_train, n_repeats=10, random_state=42, n_jobs=2)
        importances = result.importances_mean
        selected_indices = np.argsort(importances)[::-1]
        indices = selected_indices[:k]

    x_train_selected = x_train.iloc[:, indices]
    x_val_selected = x_val.iloc[:, indices]

    # Initial Broad Search (Randomized)
    print(f"Running initial broad search for {model_name} with {feature_selector.__class__.__name__} and k={k}")
    initial_results, best_classifier, best_initial_params = perform_search(x_train_selected, y_train, x_val_selected,
                                                                           y_val, model, initial_param_grid, model_name,
                                                                           "Initial Broad Search", k,
                                                                           feature_selector.__class__.__name__,
                                                                           search_type='random')

    # Refined Search (Randomized)
    refined_rf_param_grid = generate_param_grid(best_initial_params, n_estimators_range=25, max_depth_range=5,
                                                min_samples_split_range=5, min_samples_leaf_range=5)
    print(f"Running refined search for {model_name} with {feature_selector.__class__.__name__} and k={k}")
    refined_results, best_classifier, best_refined_params = perform_search(x_train_selected, y_train, x_val_selected,
                                                                           y_val, model, refined_rf_param_grid,
                                                                           model_name, "Refined Search", k,
                                                                           feature_selector.__class__.__name__,
                                                                           search_type='random')

    # Further Refined Search (Grid)
    further_refined_rf_param_grid = generate_param_grid(best_refined_params, n_estimators_range=10, max_depth_range=2,
                                                        min_samples_split_range=2, min_samples_leaf_range=2)
    print(f"Running final grid search for {model_name} with {feature_selector.__class__.__name__} and k={k}")
    final_results, best_classifier, _ = perform_search(x_train_selected, y_train, x_val_selected, y_val, model,
                                                       further_refined_rf_param_grid, model_name, "Final Grid Search",
                                                       k, feature_selector.__class__.__name__, search_type='grid')

    return initial_results + refined_results + final_results, best_classifier, x_train_selected, x_val_selected


def model_with_fs_optimized(df, model, initial_param_grid, model_name, results_dir, dataset_name):
    splits_dir = r'/data/results/Classification/conclusion_table.xlsx'
    splits_df = pd.read_excel(splits_dir)

    cleaned_model_name = extract_hebrew_and_underscores_exclude(dataset_name, "אחוזון")

    specific_dataset = splits_df[splits_df['Dataset'].str.contains(cleaned_model_name)]
    split_value = float(specific_dataset['Best ROC AUC Split'].values[0])  # Ensure split_value is a single float

    df1 = replace_with_nan_based_on_rik(df)

    threshold = float(df1["יבול-  (ק\"ג/ד')"].quantile(1 - split_value))  # Ensure threshold is a single float value
    df1["יבול-  (ק\"ג/ד')"] = (df1["יבול-  (ק\"ג/ד')"] >= threshold).astype(int)

    x = df1.drop(columns=["יבול-  (ק\"ג/ד')"])
    y = df1["יבול-  (ק\"ג/ד')"]

    plot_class_distribution(y, split_value, dataset_name)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=config["KFold random_state"])

    feature_selectors = [
        HistGradientBoostingClassifier(),
    ]
    ks = [5, 10, 20]

    results = []
    best_classifier_overall = None
    best_roc_auc = -np.inf
    best_x_val_selected = None

    for feature_selector in feature_selectors:
        for k in ks:
            print(f"Running model {model_name} with feature selector {feature_selector.__class__.__name__} and k={k}")
            model_results, best_classifier, x_train_selected, x_val_selected = evaluate_model_with_fs_optimized(
                x_train, y_train, x_val, y_val, feature_selector, k, model, initial_param_grid, model_name
            )
            results.extend(model_results)
            if model_results[-1]['ROC AUC'] > best_roc_auc:
                best_roc_auc = model_results[-1]['ROC AUC']
                best_classifier_overall = best_classifier
                best_x_val_selected = x_val_selected

    # Save results to an Excel file
    results_df_optimized = pd.DataFrame(results)

    # Convert the config dictionary to a DataFrame
    config_df = pd.DataFrame(list(rf_initial_param_grid.items()), columns=['Configuration', 'Value'])

    feature_importances = best_classifier_overall.feature_importances_
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

    return best_classifier_overall, best_x_val_selected, y_val


def run_rf_all_models():
    # Define paths
    data_dir = r'/data/models_sets/train'
    results_base_dir = r'/data/results'

    # Loop through each dataset in the data directory
    for filename in os.listdir(data_dir):
        if "reduced" in filename and "אחוזון" in filename:
            dataset_path = os.path.join(data_dir, filename)
            dataset_name = "RandomForest_" + os.path.splitext(filename)[0]
            print(f'Runing model {dataset_name}...')

            # Create a result directory for the current dataset
            results_dir = os.path.join(results_base_dir, dataset_name)
            os.makedirs(results_dir, exist_ok=True)

            # Data set load
            df_to_train = pd.read_excel(dataset_path)
            if "גרעינים" in filename and "יחס" not in filename:
                df_to_train = df_to_train.drop(columns=["יבול קש (ק\"ג/ד')"])
            print("Starting the full model evaluation and hyperparameter tuning pipeline...")

            best_model, x_val_selected, y_val = model_with_fs_optimized(df_to_train,
                                                                        RandomForestClassifier(
                                                                            random_state=config["KFold random_state"]),
                                                                        rf_initial_param_grid, "RandomForestClassifier",
                                                                        results_dir, dataset_name)
            calculate_and_save_feature_importance(best_model, x_val_selected, y_val, results_dir,
                                                  dataset_name, 'predict_proba', 'auto')


run_rf_all_models()
