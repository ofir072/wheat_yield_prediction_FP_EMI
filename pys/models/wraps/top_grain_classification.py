import os
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


def run_classification_all_models():
    # Define paths
    data_dir = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\models_sets\train'
    results_base_dir = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\results'

    # Create a result directory for the current dataset
    results_dir = os.path.join(results_base_dir, "Classification")
    os.makedirs(results_dir, exist_ok=True)

    # Class splits
    splits = [0.3, 0.25, 0.2, 0.15]

    # Loop through each dataset in the data directory
    for filename in os.listdir(data_dir):
        if "reduced" in filename:
            if "שעורה" in filename or "שחת_תחמיץ" in filename or ("חיטה" in filename and "יחס" not in filename):
                dataset_path = os.path.join(data_dir, filename)
                dataset_name = "Classification_" + os.path.splitext(filename)[0]
                print(f'Running model {dataset_name}...')

                # Load dataset
                df_to_train = pd.read_excel(dataset_path)
                if "גרעינים" in filename and "יחס" not in filename:
                    df_to_train = df_to_train.drop(columns=["יבול קש (ק\"ג/ד')"])

                # Extract target variable and features
                target = df_to_train["יבול-  (ק\"ג/ד')"]
                features = df_to_train.drop(columns=["יבול-  (ק\"ג/ד')"])

                results = []

                for split in splits:
                    # Define threshold for top grain level
                    threshold = target.quantile(1 - split)
                    class_labels = (target >= threshold).astype(int)

                    # Split data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(features, class_labels, test_size=0.3,
                                                                        random_state=42)

                    # Create and train the model
                    model = HistGradientBoostingClassifier(random_state=42)
                    model.fit(X_train, y_train)

                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]

                    # Calculate evaluation scores
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    roc_auc = roc_auc_score(y_test, y_prob)

                    # Store results
                    results.append({
                        'split': split,
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'roc_auc_score': roc_auc
                    })

                # Save results to an Excel file
                results_df = pd.DataFrame(results)
                results_df.to_excel(os.path.join(results_dir, f'{dataset_name}_results.xlsx'), index=False)
                print(f'Results saved for {dataset_name}.')

            else:
                print(f'The current file: {filename} has not met the criteria for this model.')
                continue


def generate_conclusion_table(results_dir):
    conclusion_data = []

    for file in os.listdir(results_dir):
        if file.endswith("_results.xlsx"):
            file_path = os.path.join(results_dir, file)
            df = pd.read_excel(file_path)
            dataset_name = file.replace("_results.xlsx", "")

            best_accuracy_split = df.loc[df['accuracy'].idxmax()]['split']
            best_f1_split = df.loc[df['f1_score'].idxmax()]['split']
            best_roc_auc_split = df.loc[df['roc_auc_score'].idxmax()]['split']

            conclusion_data.append({
                'Dataset': dataset_name,
                'Best Accuracy Split': best_accuracy_split,
                'Best F1 Score Split': best_f1_split,
                'Best ROC AUC Split': best_roc_auc_split
            })

    conclusion_df = pd.DataFrame(conclusion_data)
    conclusion_df.to_excel(os.path.join(results_dir, 'conclusion_table.xlsx'), index=False)
    print('Conclusion table saved as conclusion_table.xlsx')


# Run the function
run_classification_all_models()

# Generate the conclusion table
results_directory = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\results\Classification'
generate_conclusion_table(results_directory)
