import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
from pys.data_preperation.pre_train_process import replace_with_nan_based_on_rik

# Configuration details
config = {
    "KFold random_state": 42,
    "KFold n_splits": 5
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


def perform_search(x_train_selected, y_train, x_val_selected, y_val, model,
                   model_name, k, feature_selector_name, param_grid):
    search = GridSearchCV(estimator=model, param_grid=param_grid,
                          cv=KFold(n_splits=config["KFold n_splits"], shuffle=True,
                                   random_state=config["KFold random_state"]), scoring='roc_auc',
                          return_train_score=True)
    search.fit(x_train_selected, y_train)
    best_classifier = search.best_estimator_
    y_pred = best_classifier.predict(x_val_selected)
    accuracy, f1, roc_auc, cv_mean, cv_std = calculate_metrics(y_val, y_pred, x_val_selected, best_classifier,
                                                               x_train_selected, y_train)
    results = [{
        "Model": model_name,
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
    return results, best_classifier


def evaluate_model_with_fs_optimized(x_train, y_train, x_val, y_val, feature_selector,
                                     k, model, model_name, param_grid):
    # Train naive HistGradientBoosting model to extract K-best informative features
    selector = feature_selector.fit(x_train, y_train)
    result = permutation_importance(selector, x_train, y_train, n_repeats=10, random_state=42, n_jobs=2)
    importances = result.importances_mean
    selected_indices = np.argsort(importances)[::-1]
    indices = selected_indices[:k]
    # Filter only the K-best features founded
    x_train_selected = x_train.iloc[:, indices]
    x_val_selected = x_val.iloc[:, indices]
    print(f"Running broad search for {model_name} with {feature_selector.__class__.__name__} and k={k}")
    # Set the set to the search phase to do parameter tuning for the model
    results, best_classifier = perform_search(x_train_selected, y_train,
                                              x_val_selected, y_val, model, model_name, k,
                                              feature_selector.__class__.__name__, param_grid)
    return results, best_classifier, x_train_selected, x_val_selected


def extract_hebrew_and_underscores_exclude(text, exclude_word):
    # Remove the word from the text
    text = text.replace(exclude_word, '')
    # This regex pattern matches only Hebrew letters (Aleph to Tav) and underscores
    pattern = r'[א-ת_]+'
    result = re.findall(pattern, text)
    # Join the matched parts to form the final string
    name = ''.join(result)
    return name.replace('__', '_')


def plot_class_distribution(y, split_value, dataset_name):

    plots_dir = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\plots'

    class_counts = y.value_counts()
    plt.figure(figsize=(8, 6))

    # Custom class labels based on split value
    class_labels = {0: f"Class Bottom Grain", 1: f"Class Top Grain"}
    bars = plt.bar(class_counts.index, class_counts.values, width=0.4, color=['skyblue', 'salmon'])

    # Adding value annotations on the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, str(int(yval)), ha='center', va='bottom')

    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title(rf'Class Distribution - {split_value * 100:.1f}% Percentile', fontsize=16)
    plt.xticks(class_counts.index, [class_labels[idx] for idx in class_counts.index], fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plot_filename = os.path.join(plots_dir, f'class_distribution_split_{split_value * 100:.1f}_{dataset_name}.png')
    plt.savefig(plot_filename)
    plt.close()

    print(f"Plot saved to {plot_filename}")


def model_with_fs_optimized(df, model, model_name, results_dir, dataset_name, param_grid, score):
    # Extract the percentile learned to set the classes by it
    splits_dir = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\results\Classification\
                    conclusion_table.xlsx'
    splits_df = pd.read_excel(splits_dir)
    cleaned_model_name = extract_hebrew_and_underscores_exclude(dataset_name, "אחוזון")
    specific_dataset = splits_df[splits_df['Dataset'].str.contains(cleaned_model_name)]
    split_value = float(specific_dataset['Best ROC AUC Split'].values[0])  # Ensure split_value is a single float
    df1 = replace_with_nan_based_on_rik(df)
    threshold = float(df1["יבול-  (ק\"ג/ד')"].quantile(1 - split_value))  # Ensure threshold is a single float value
    # Define the target variable to the classification task by the percentile value
    df1["יבול-  (ק\"ג/ד')"] = (df1["יבול-  (ק\"ג/ד')"] >= threshold).astype(int)
    if "LightGBM" in model_name:
        # Clean feature names
        df1 = df.rename(columns=lambda x: x.replace(' ', '_').replace('(', '').replace(')', '').replace("'", '')
                        .replace('"', ''))
        x = df1.drop(columns=["יבול-__קג/ד"])
        y = df1["יבול-__קג/ד"]
        df1.columns = df1.columns.str.replace(r'[^\w\s]', '', regex=True).str.replace(' ', '_')
    else:
        x = df1.drop(columns=["יבול-  (ק\"ג/ד')"])
        y = df1["יבול-  (ק\"ג/ד')"]
    # Self control plot to ensure equal class distribution
    plot_class_distribution(y, split_value, dataset_name)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2,
                                                      random_state=config["KFold random_state"])
    feature_selectors = [
        HistGradientBoostingClassifier(),
    ]
    # Top K best feature range to train
    ks = [5, 10, 20]
    results = []
    best_classifier_overall = None
    best_roc_auc = -np.inf
    best_x_val_selected = None
    for feature_selector in feature_selectors:
        for k in ks:
            print(f"Running model {model_name} with feature selector {feature_selector.__class__.__name__} and k={k}")
            # Run the model to train phase
            model_results, best_classifier, x_train_selected, x_val_selected = evaluate_model_with_fs_optimized(
                x_train, y_train, x_val, y_val, feature_selector, k, model, model_name, param_grid)
            results.extend(model_results)
            # Save the best model to make the trained model feature exploration
            if model_results[-1][score] > best_roc_auc:
                best_roc_auc = model_results[-1][score]
                best_classifier_overall = best_classifier
                best_x_val_selected = x_val_selected
    # Save results to an Excel file
    results_df_optimized = pd.DataFrame(results)
    # Convert the config dictionary to a DataFrame
    config_df = pd.DataFrame(list(param_grid.items()), columns=['Configuration', 'Value'])
    importance_df = pd.DataFrame()
    if "HistGradientBoosting" not in model_name:
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
        if "HistGradientBoosting" not in model_name:
            importance_df.to_excel(writer, sheet_name='Feature Importances', index=False)
    print("Model training complete. Best model and results saved.")
    return best_classifier_overall, best_x_val_selected, y_val
