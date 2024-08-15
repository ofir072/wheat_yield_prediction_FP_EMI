import os
import re
import numpy as np
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def sanitize_filename(filename):
    # Remove invalid characters and replace spaces with underscores
    return re.sub(r'[\\/*?:"<>|]', "", filename).replace(' ', '_')


def ensure_directory_exists(filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_partial_dependence_plots(model, x_val, features, output_dir, model_name,
                                  response_method, method, batch_size=6):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(0, len(features), batch_size):
        # Separate the plots into batch for visualization propose
        batch_features = features[i:i + batch_size]
        fig, axes = plt.subplots(nrows=len(batch_features), figsize=(12, 8 * len(batch_features)))
        if len(batch_features) == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one plot
        for feature, ax in zip(batch_features, axes):
            # Creat the 1-Way PDP data
            PartialDependenceDisplay.from_estimator(model, x_val, [feature], response_method=response_method,
                                                    method=method, ax=ax)
        # Plot the PDP and the save the result
        plt.tight_layout()
        sanitized_model_name = sanitize_filename(model_name)
        filename = os.path.join(output_dir, f'pdp_one_way_{sanitized_model_name}_batch_{i // batch_size}.png')
        ensure_directory_exists(filename)
        plt.savefig(filename)
        plt.close()


def save_ice_plots(model, x_val, features, output_dir, model_name,
                   response_method, method, batch_size=6):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(0, len(features), batch_size):
        batch_features = features[i:i + batch_size]
        fig, axes = plt.subplots(nrows=len(batch_features), figsize=(12, 8 * len(batch_features)))
        if len(batch_features) == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one plot
        for feature, ax in zip(batch_features, axes):
            PartialDependenceDisplay.from_estimator(model, x_val, [feature], response_method=response_method,
                                                    method=method, kind='both', ax=ax)
        plt.tight_layout()
        sanitized_model_name = sanitize_filename(model_name)
        filename = os.path.join(output_dir, f'ice_{sanitized_model_name}_batch_{i // batch_size}.png')
        ensure_directory_exists(filename)
        plt.savefig(filename)
        plt.close()


def save_top_two_way_pdp_plots(model, x_val, features, output_dir, response_method, method):
    os.makedirs(output_dir, exist_ok=True)
    if len(features) > 1:
        # Focus only on the top 5 importance features and creation of all pairs combinations
        top_features = features[:5]
        pairs = [(top_features[i], top_features[j]) for i in range(len(top_features)) for j in range(i + 1,
                                                                              len(top_features))]
        for i, pair in enumerate(pairs):
            fig, ax = plt.subplots(figsize=(12, 8))
            try:
                # 2-Way PDP calculation and plotting
                PartialDependenceDisplay.from_estimator(model, x_val, [pair], response_method=response_method,
                                                        method=method, ax=ax)
                plt.tight_layout()
                filename = os.path.join(output_dir,
                                        f'pdp_two_way_{sanitize_filename(pair[0])}_{sanitize_filename(pair[1])}.png')
                ensure_directory_exists(filename)
                plt.savefig(filename)
            except (TypeError, ValueError) as e:
                print(f"Skipping two-way PDP plot for pair {pair} due to: {e}")
            plt.close()


def calculate_and_save_feature_importance(model, x_val, y_val, output_dir, model_name,
                                          response_method, method, batch_size=6):
    # Calculate feature importance of the feature based on the model permutation progression
    result = permutation_importance(model, x_val, y_val, n_repeats=10, random_state=0)
    perm_importances = pd.Series(result.importances_mean, index=x_val.columns).sort_values(ascending=False)
    perm_importances_df = pd.DataFrame({'Feature': perm_importances.index, 'Importance': perm_importances.values})
    sanitized_model_name = sanitize_filename(model_name)
    filename = os.path.join(output_dir, f'permutation_importance_{sanitized_model_name}.xlsx')
    ensure_directory_exists(filename)
    perm_importances_df.to_excel(filename, index=False)
    features_to_plot = perm_importances.index.tolist()
    # Creations of 1-Way & 2-Way PDP and ICE plots
    save_partial_dependence_plots(model, x_val, features_to_plot, output_dir, model_name, response_method, method,
                                  batch_size)
    save_top_two_way_pdp_plots(model, x_val, features_to_plot, output_dir, model_name, response_method, method)
    save_ice_plots(model, x_val, features_to_plot, output_dir, model_name, response_method, method, batch_size)
    if method == "predict_proba":
        ice_histogram(model, x_val, y_val, output_dir, model_name)
    print("Best model ICE, PDP and Feature Permutation results saved.")


def ice_histogram(model, x_val, y_val, output_dir, model_name, top_n_features=10):
    # Step 1: Compute Permutation Importance and Get Features to Plot
    result = permutation_importance(model, x_val, y_val, n_repeats=10, random_state=42)
    perm_importances = pd.Series(result.importances_mean, index=x_val.columns).sort_values(ascending=False)
    features_to_plot = perm_importances.index.tolist()[:top_n_features]
    num_points = 50  # Number of points to evaluate for ICE
    # Step 2: Generate ICE Data and Identify Most Probable Class
    ice_data = {feature: [] for feature in features_to_plot}
    for feature in features_to_plot:
        feature_index = x_val.columns.get_loc(feature)
        feature_values = np.linspace(x_val[feature].min(), x_val[feature].max(), num_points)
        for instance in x_val.values:
            instance_predictions = []
            for value in feature_values:
                modified_instance = instance.copy()
                modified_instance[feature_index] = value
                modified_instance_df = pd.DataFrame([modified_instance], columns=x_val.columns)
                pred_prob = model.predict_proba(modified_instance_df)[0]
                most_probable_class = np.argmax(pred_prob)
                instance_predictions.append((value, most_probable_class))
            ice_data[feature].append(instance_predictions)
    # Step 3: Separate Feature Values by Class
    class_feature_values = {feature: {i: [] for i in range(model.classes_.size)} for feature in features_to_plot}
    for feature in features_to_plot:
        for instance_predictions in ice_data[feature]:
            for value, most_probable_class in instance_predictions:
                class_feature_values[feature][most_probable_class].append(value)
    # Step 4: Create Overlapping Histograms and Distribution Lines for Each Feature
    colors = ['#77dd77', '#ff6961']  # Green and red colors
    # Calculate the actual amount of each class in the dataset
    class_counts = y_val.value_counts().to_dict()
    print(f"Class counts: {class_counts}")
    for feature in features_to_plot:
        plt.figure()
        for class_label, values in class_feature_values[feature].items():
            if class_label in class_counts:
                # Normalize the weights based on class counts
                normalized_weights = np.ones_like(values) / class_counts[class_label]
                # Convert to long-form data
                data = pd.DataFrame({
                    'values': values,
                    'weights': normalized_weights
                })
                sns.histplot(data=data, x='values', bins=50, kde=True, color=colors[class_label % len(colors)],
                             label=f'Class {class_label}', alpha=0.5, weights='weights')
        plt.xlabel(f'{feature} Value')
        plt.ylabel('Frequency (Wighted)')
        plt.legend()
        plt.title(f'ICE Histogram and Distribution for {feature}')
        plt.tight_layout()
        feature = sanitize_filename(feature)
        filename = os.path.join(output_dir, f'ice_hist_wighted_{feature}.png')
        print(f"Saved in {filename}")
        plt.savefig(filename)
        plt.close()


def ice_histogram_not_wighted(model, x_val, y_val, output_dir, model_name, top_n_features=10):
    # Step 1: Compute Permutation Importance and Get Features to Plot
    result = permutation_importance(model, x_val, y_val, n_repeats=10, random_state=42)
    perm_importances = pd.Series(result.importances_mean, index=x_val.columns).sort_values(ascending=False)
    features_to_plot = perm_importances.index.tolist()[:top_n_features]

    num_points = 50  # Number of points to evaluate for ICE

    # Step 2: Generate ICE Data and Identify Most Probable Class
    ice_data = {feature: [] for feature in features_to_plot}

    for feature in features_to_plot:
        feature_index = x_val.columns.get_loc(feature)
        feature_values = np.linspace(x_val[feature].min(), x_val[feature].max(), num_points)

        for instance in x_val.values:
            instance_predictions = []
            for value in feature_values:
                modified_instance = instance.copy()
                modified_instance[feature_index] = value
                modified_instance_df = pd.DataFrame([modified_instance], columns=x_val.columns)
                pred_prob = model.predict_proba(modified_instance_df)[0]
                most_probable_class = np.argmax(pred_prob)
                instance_predictions.append((value, most_probable_class))
            ice_data[feature].append(instance_predictions)

    # Step 3: Separate Feature Values by Class
    class_feature_values = {feature: {i: [] for i in range(model.classes_.size)} for feature in
                            features_to_plot}

    for feature in features_to_plot:
        for instance_predictions in ice_data[feature]:
            for value, most_probable_class in instance_predictions:
                class_feature_values[feature][most_probable_class].append(value)

    # Step 4: Create Overlapping Histograms and Distribution Lines for Each Feature
    colors = ['#77dd77', '#ff6961']  # Green and red colors
    for feature in features_to_plot:
        plt.figure()
        for class_label, values in class_feature_values[feature].items():
            sns.histplot(values, bins=50, kde=True, color=colors[class_label % len(colors)],
                         label=f'Class {class_label}', alpha=0.5)
        plt.xlabel(f'{feature} Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title(f'ICE Histogram and Distribution for {feature}')
        plt.tight_layout()
        feature = sanitize_filename(feature)
        filename = os.path.join(output_dir, f'ice_hist_{feature}.png')
        print(f"saved in {filename}")
        plt.savefig(filename)
        plt.close()