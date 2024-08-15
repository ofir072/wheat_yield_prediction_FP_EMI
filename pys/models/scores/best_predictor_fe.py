import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from pys.data_preperation.pre_train_process import replace_with_nan_based_on_rik
from sklearn.tree import plot_tree, DecisionTreeRegressor, DecisionTreeClassifier
from pys.models.scores.feature_importance_analysis import ice_histogram


# Configuration details
config = {
    "KFold random_state": 42,
    "KFold n_splits": 5,
}


def save_tree_plot(best_regressor, feature_names, results_dir, dataset_name):

    # Visualize the tree with limited depth
    plt.figure(figsize=(20, 10))
    plot_tree(best_regressor, feature_names=feature_names, filled=True, max_depth=3)
    filename = os.path.join(results_dir, f'tree_plot_{dataset_name}.png')
    plt.savefig(filename)
    plt.close()
    print(f"Tree plot saved to {filename}")


def extract_hebrew_and_underscores_exclude(text, exclude_word):
    # Remove the word from the text
    text = text.replace(exclude_word, '')
    # This regex pattern matches only Hebrew letters (Aleph to Tav) and underscores
    pattern = r'[א-ת_]+'
    result = re.findall(pattern, text)
    # Join the matched parts to form the final string
    name = ''.join(result)
    return name.replace('__', '_')


def best_model_feature_importance(df, model, model_name, dataset_name, param_grid, results_dir, classifiaction_trend):

    df1 = replace_with_nan_based_on_rik(df)

    if classifiaction_trend is True:
        splits_dir = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\results\Classification\conclusion_table.xlsx'
        splits_df = pd.read_excel(splits_dir)

        cleaned_model_name = extract_hebrew_and_underscores_exclude(dataset_name, "אחוזון")

        specific_dataset = splits_df[splits_df['Dataset'].str.contains(cleaned_model_name)]
        split_value = float(specific_dataset['Best ROC AUC Split'].values[0])  # Ensure split_value is a single float

        threshold = float(df1["יבול-  (ק\"ג/ד')"].quantile(1 - split_value))  # Ensure threshold is a single float value
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

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2,
                                                      random_state=config["KFold random_state"])

    feature_selector = HistGradientBoostingClassifier()
    k = 10

    selector = feature_selector.fit(x_train, y_train)
    result = permutation_importance(selector, x_train, y_train, n_repeats=10, random_state=42, n_jobs=2)
    importances = result.importances_mean
    selected_indices = np.argsort(importances)[::-1]
    indices = selected_indices[:k]

    x_train_selected = x_train.iloc[:, indices]
    x_val_selected = x_val.iloc[:, indices]

    search = GridSearchCV(estimator=model, param_grid=param_grid,
                          cv=KFold(n_splits=config["KFold n_splits"], shuffle=True,
                                   random_state=config["KFold random_state"]), scoring='roc_auc',
                          return_train_score=True)

    search.fit(x_train_selected, y_train)
    best_model = search.best_estimator_

    # Get the correct feature names
    # feature_names = x_val_selected.columns
    # save_tree_plot(best_model, feature_names, results_dir, dataset_name)

    ice_histogram(best_model, x_val_selected, y_val, results_dir, model_name)


if __name__ == "__main__":
    set_name = 'שעורה_אחוזון_בעל'
    dataset_path = fr'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\models_sets\train\reduced_{set_name}_train.xlsx'
    # Data set load
    df_to_train = pd.read_excel(dataset_path)
    if "גרעינים" in dataset_path and "יחס" not in dataset_path:
        df_to_train = df_to_train.drop(columns=["יבול קש (ק\"ג/ד')"])
    model = RandomForestClassifier()
    model_name = "RandomForestClassifier"
    dataset_name = fr'RandomForestClassifier_reduced_{set_name}_train'
    param_grid = {'max_depth': [2], 'min_samples_leaf': [1], 'min_samples_split': [2], 'n_estimators': [76]}
    results_dir = fr'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\results\RandomForest_reduced_{set_name}_train'
    best_model_feature_importance(df_to_train, model, model_name, dataset_name, param_grid, results_dir, True)
