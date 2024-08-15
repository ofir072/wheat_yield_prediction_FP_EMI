from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pys.data_preperation.pre_train_process import replace_with_nan_based_on_rik


def dt(df):
    """
    Summary - Decision trees in scikit-learn can handle missing values:
        * Have built-in support for missing values when splitter='best' and criterion is 'squared_error', 'friedman_mse', or 'poisson' for regression.
        * They evaluate splits by considering scenarios where missing values go to either side.
        * During prediction, they use splits from training or default to the child with the most samples if no missing values were seen during training.
        * Ties during prediction are broken by default to the right node.
    """
    df1 = replace_with_nan_based_on_rik(df)
    x = df1.drop(columns=["יבול-  (ק\"ג/ד')"])
    y = df1["יבול-  (ק\"ג/ד')"]
    # Splitting the data into train and test sets
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # Define the parameter grid
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    # Initialize the Decision Tree Regressor
    regressor = DecisionTreeRegressor(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

    # Fit GridSearchCV
    grid_search.fit(x_train, y_train)

    # Best parameters from GridSearchCV
    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")

    # Train the best estimator on the entire training data
    best_regressor = grid_search.best_estimator_

    # Predict on the validation set
    y_pred = best_regressor.predict(x_val)

    # Calculate accuracy
    mse = mean_squared_error(y_val, y_pred)
    print(f"Mean Squared Error on Validation Set: {mse}")

    # Calculate R-squared and Adjusted R-squared
    r2 = r2_score(y_val, y_pred)
    n = x_val.shape[0]
    p = x_val.shape[1]
    r2_adj = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    print(f"R-squared on Validation Set: {r2}")
    print(f"Adjusted R-squared on Validation Set: {r2_adj}")

    # Cross-validation scores
    cv_scores = cross_val_score(best_regressor, x, y, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-Validation MSE: {-np.mean(cv_scores)} ± {np.std(cv_scores)}")

    # Visualize the tree with limited depth
    plt.figure(figsize=(20, 10))
    plot_tree(best_regressor, feature_names=df1.columns[:-1], filled=True, max_depth=4)
    plt.show()

    return best_regressor


def dt_with_rf(df):
    df1 = replace_with_nan_based_on_rik(df)
    x = df1.drop(columns=["יבול-  (ק\"ג/ד')"])
    y = df1["יבול-  (ק\"ג/ד')"]

    # Splitting the data into train and test sets
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # Define the parameter grid for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize the Random Forest Regressor
    regressor = RandomForestRegressor(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

    # Fit GridSearchCV
    grid_search.fit(x_train, y_train)

    # Best parameters from GridSearchCV
    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")

    # Train the best estimator on the entire training data
    best_regressor = grid_search.best_estimator_

    # Predict on the validation set
    y_pred = best_regressor.predict(x_val)

    # Calculate accuracy
    mse = mean_squared_error(y_val, y_pred)
    print(f"Mean Squared Error on Validation Set: {mse}")

    # Calculate R-squared and Adjusted R-squared
    r2 = r2_score(y_val, y_pred)
    n = x_val.shape[0]
    p = x_val.shape[1]
    r2_adj = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    print(f"R-squared on Validation Set: {r2}")
    print(f"Adjusted R-squared on Validation Set: {r2_adj}")

    # Cross-validation scores
    cv_scores = cross_val_score(best_regressor, x, y, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-Validation MSE: {-np.mean(cv_scores)} ± {np.std(cv_scores)}")

    return best_regressor


location_file = r'C:\Users\ofirgot\PycharmProjects\pythonProject\data\models_sets\train\reduced_שעורה_train.xlsx'
df_to_train = pd.read_excel(location_file)
best_regressors = dt(df_to_train)
