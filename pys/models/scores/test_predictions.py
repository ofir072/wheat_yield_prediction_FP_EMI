import numpy as np
import pandas as pd
import re
import ast
import os
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from pys.data_preperation.pre_train_process import replace_with_nan_based_on_rik

# Configuration details
config = {
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier(random_state=42),
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=42),
    "LightGBMClassifier": lgb.LGBMClassifier(random_state=42, verbose=-1),
    "LightGBMRegressor": lgb.LGBMRegressor(random_state=42, verbose=-1),
    "MLP": MLPRegressor(max_iter=1000, random_state=42),
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "RandomForestRegressor": RandomForestRegressor(random_state=42),
    "SVR": SVR()
}

results = {
    'Classification': [],
    'Regression': []
}

# Define the find_model function
def find_model(dataset_name):
    print(f"Finding best model for dataset: {dataset_name}")
    results_dir = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\performance\best_models_summary.xlsx'

    # Load both sheets
    df_classification = pd.read_excel(results_dir, sheet_name='Classification Models')
    df_regression = pd.read_excel(results_dir, sheet_name='Regression Models')

    # Determine if the task is classification
    is_classification = 'אחוזון' in dataset_name

    # Extract the relevant part of the dataset name
    cleaned_dataset_name = '_'.join([part for part in dataset_name.split('_') if re.search(r'[\u0590-\u05FF]', part)])

    # Select the appropriate dataframe
    if is_classification:
        df = df_classification[df_classification['Dataset'].str.contains(cleaned_dataset_name, regex=False)]
        if df.empty:
            raise ValueError(f"No matching dataset found in the classification sheet for {cleaned_dataset_name}")
        best_model_row = df.loc[df['ROC AUC'].idxmax()].copy()
    else:
        df = df_regression[df_regression['Dataset'].str.contains(cleaned_dataset_name, regex=False)]
        if df.empty:
            raise ValueError(f"No matching dataset found in the regression sheet for {cleaned_dataset_name}")
        best_model_row = df.loc[df['R2'].idxmax()].copy()

    best_model = config[f"{best_model_row['Model']}"]
    best_param_grid = ast.literal_eval(best_model_row['Best Params'])  # Convert string to dictionary
    best_k = int(best_model_row['Number of Features'])

    # Ensure all values in param_grid are lists
    for key, value in best_param_grid.items():
        if not isinstance(value, list):
            best_param_grid[key] = [value]

    print(f"Best model: {best_model.__class__.__name__}, Params: {best_param_grid}, K: {best_k}, Classification: {is_classification}")
    return best_model, best_param_grid, best_k, is_classification

def extract_hebrew_and_underscores_exclude(text, exclude_word):
    text = text.replace(exclude_word, '')
    pattern = r'[א-ת_]+'
    result = re.findall(pattern, text)
    name = ''.join(result)
    return name.replace('__', '_')

def calculate_metrics(y_original, y_pred, x_val_selected, model, is_classification):
    metrics = {}
    if is_classification:
        metrics['accuracy'] = accuracy_score(y_original, y_pred)
        metrics['f1'] = f1_score(y_original, y_pred, average='weighted')
        metrics['roc_auc'] = roc_auc_score(y_original, model.predict_proba(x_val_selected)[:, 1])
    else:
        metrics['mse'] = mean_squared_error(y_original, y_pred)
        metrics['r2'] = r2_score(y_original, y_pred)
        n = len(y_original)
        p = x_val_selected.shape[1]
        metrics['r2_adj'] = 1 - ((1 - metrics['r2']) * (n - 1)) / (n - p - 1)
    return metrics


def best_model_test_prediction(df_train, df_test, model, dataset_name, param_grid, k, is_classification):
    print(f"Starting model training and prediction for dataset: {dataset_name}")
    # Handle possible nan value to prevent run time error
    df_tr = replace_with_nan_based_on_rik(df_train)
    df_te = replace_with_nan_based_on_rik(df_test)
    # Create the class target variable base on the learned percentile for the classification models
    if is_classification:
        splits_dir = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\results\
                        Classification\conclusion_table.xlsx'
        splits_df = pd.read_excel(splits_dir)
        cleaned_model_name = extract_hebrew_and_underscores_exclude(dataset_name, "אחוזון")
        specific_dataset = splits_df[splits_df['Dataset'].str.contains(cleaned_model_name)]
        split_value = float(specific_dataset['Best ROC AUC Split'].values[0])
        threshold = float(df_tr["יבול-  (ק\"ג/ד')"].quantile(1 - split_value))
        df_tr["יבול-  (ק\"ג/ד')"] = (df_tr["יבול-  (ק\"ג/ד')"] >= threshold).astype(int)
        df_te["יבול-  (ק\"ג/ד')"] = (df_te["יבול-  (ק\"ג/ד')"] >= threshold).astype(int)

    if "LightGBM" in dataset_name:
        # Clean feature names for the "LightGBM" sensitive algorithm
        df_tr = df_tr.rename(
            columns=lambda x: x.replace(' ', '_').replace('(', '').replace(')', '').replace("'", '').replace('"', ''))
        df_tr.columns = df_tr.columns.str.replace(r'[^\w\s]', '', regex=True).str.replace(' ', '_')
        df_te = df_te.rename(
            columns=lambda x: x.replace(' ', '_').replace('(', '').replace(')', '').replace("'", '').replace('"', ''))
        df_te.columns = df_te.columns.str.replace(r'[^\w\s]', '', regex=True).str.replace(' ', '_')
        x_train = df_tr.drop(columns=["יבול-__קג/ד"])
        y_train = df_tr["יבול-__קג/ד"]
        x_test = df_te.drop(columns=["יבול-__קג/ד"])
        y_test = df_te["יבול-__קג/ד"]

    else:
        x_train = df_tr.drop(columns=["יבול-  (ק\"ג/ד')"])
        y_train = df_tr["יבול-  (ק\"ג/ד')"]
        x_test = df_te.drop(columns=["יבול-  (ק\"ג/ד')"])
        y_test = df_te["יבול-  (ק\"ג/ד')"]
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    # Calculate the top K features the model learned on
    feature_selector = HistGradientBoostingRegressor()
    selector = feature_selector.fit(x_train, y_train)
    result = permutation_importance(selector, x_train, y_train, n_repeats=10, random_state=42, n_jobs=2)
    importances = result.importances_mean
    selected_indices = np.argsort(importances)[::-1]
    indices = selected_indices[:k]
    # Filter the data set to only the K-best features
    x_train_selected = x_train.iloc[:, indices]
    x_val_selected = x_val.iloc[:, indices]
    x_test_selected = x_test.iloc[:, indices]
    scoring = 'roc_auc' if is_classification else 'r2'
    # Preform the best model training and implement the CV approach
    search = GridSearchCV(estimator=model, param_grid=param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                          scoring=scoring, return_train_score=True)
    print(f"Starting GridSearchCV for dataset: {dataset_name} with params: {param_grid}")
    search.fit(x_train_selected, y_train)
    best_model = search.best_estimator_
    #  Predict the target variable for each data set
    train_pred = best_model.predict(x_train_selected)
    val_pred = best_model.predict(x_val_selected)
    test_pred = best_model.predict(x_test_selected)
    #  Calculate the metrics for each data set based on the model kind
    train_metrics = calculate_metrics(y_train, train_pred, x_train_selected, best_model, is_classification)
    val_metrics = calculate_metrics(y_val, val_pred, x_val_selected, best_model, is_classification)
    test_metrics = calculate_metrics(y_test, test_pred, x_test_selected, best_model, is_classification)
    result = {
        'Model': model.__class__.__name__,
        'Dataset': dataset_name,
        'K': k,
        'Params': search.best_params_,
        'Train Metrics': train_metrics,
        'Validation Metrics': val_metrics,
        'Test Metrics': test_metrics
    }
    print(f"Appending results for dataset: {dataset_name}")
    if is_classification:
        results['Classification'].append(result)
    else:
        results['Regression'].append(result)
    print(f"Completed training and evaluation for dataset: {dataset_name}")
    print(f"Current results: {results}")

def save_results_to_excel(results, output_path):
    print(f"Saving results to Excel at: {output_path}")
    with pd.ExcelWriter(output_path) as writer:
        for key, value in results.items():
            df = pd.DataFrame(value)
            df.to_excel(writer, sheet_name=key, index=False)
    print("Results successfully saved to Excel")

if __name__ == "__main__":
    data_dir = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\models_sets\train'
    for filename in os.listdir(data_dir):
        if "reduced" in filename:
            dataset_name = os.path.splitext(filename)[0]
            dataset_train_path = fr'{data_dir}\{dataset_name}.xlsx'
            dataset_test_path = fr'{data_dir.replace("train", "test")}\{dataset_name.replace("train", "test")}.xlsx'
            df_to_train = pd.read_excel(dataset_train_path)
            df_to_test = pd.read_excel(dataset_test_path)

            if "גרעינים" in dataset_train_path and "יחס" not in dataset_train_path:
                df_to_train = df_to_train.drop(columns=["יבול קש (ק\"ג/ד')"])
                df_to_test = df_to_test.drop(columns=["יבול קש (ק\"ג/ד')"])

            try:
                model, param_grid, k, class_or_reg = find_model(dataset_name)
                best_model_test_prediction(df_to_train, df_to_test, model, dataset_name, param_grid, k, class_or_reg)
            except Exception as e:
                print(f"Error processing dataset {dataset_name}: {e}")

    print(f"Final results before saving: {results}")
    save_results_to_excel(results, r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\performance\model_predictions_summary.xlsx')
    print("Script execution completed.")
