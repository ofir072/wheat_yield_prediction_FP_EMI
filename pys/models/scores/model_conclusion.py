import os
import pandas as pd


def find_best_model(results_folder, dataset_names):
    best_regression_models = []
    best_classification_models = []

    for dataset_name in dataset_names:
        for root, dirs, files in os.walk(results_folder):
            for dir_name in dirs:
                if dataset_name in dir_name:
                    dataset_path = os.path.join(root, dir_name)
                    for sub_root, sub_dirs, sub_files in os.walk(dataset_path):
                        for sub_file in sub_files:
                            if sub_file.endswith('.xlsx') and 'results' in sub_file:
                                file_path = os.path.join(sub_root, sub_file)
                                df = pd.read_excel(file_path, sheet_name='Results')
                                is_classification = 'אחוזון' in file_path
                                if is_classification:
                                    best_model_row = df.loc[df['ROC AUC'].idxmax()].copy()
                                    best_model_row['Dataset'] = dataset_name
                                    best_classification_models.append(best_model_row)
                                else:
                                    best_model_row = df.loc[df['R2'].idxmax()].copy()
                                    best_model_row['Dataset'] = dataset_name
                                    best_regression_models.append(best_model_row)

    return pd.DataFrame(best_regression_models), pd.DataFrame(best_classification_models)


def save_to_excel(best_regression_models_df, best_classification_models_df, output_file):
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        best_regression_models_df.to_excel(writer, sheet_name='Regression Models', index=False)
        best_classification_models_df.to_excel(writer, sheet_name='Classification Models', index=False)
    print(f"Best models summary saved to {output_file}")


def main():
    results_folder = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\results'
    dataset_names = [
        'אפונה_בעל', 'חיטה_גרעינים_בעל', 'חיטה_גרעינים_מושקה', 'חיטה_יחס_גרעינים_בעל',
        'חיטה_יחס_גרעינים_מושקה', 'חיטה_שחת_תחמיץ_בעל', 'חיטה_שחת_תחמיץ_מושקה',
        'שעורה_בעל', 'שעורה_מושקה', 'תלתן_בעל', 'חיטה_גרעינים_אחוזון_בעל', 'חיטה_גרעינים_אחוזון_מושקה',
        'חיטה_שחת_תחמיץ_אחוזון_בעל', 'חיטה_שחת_תחמיץ_אחוזון_מושקה', 'שעורה_אחוזון_בעל', 'שעורה_אחוזון_מושקה'
    ]
    best_regression_models_df, best_classification_models_df = find_best_model(results_folder, dataset_names)

    output_file = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\performance\best_models_summary.xlsx'

    save_to_excel(best_regression_models_df, best_classification_models_df, output_file)


def summarize_ml_models(data_dir, removed_features_path, output_path):
    # Initialize list to hold summary details
    summary = []

    # Load the removed features data
    removed_features = pd.read_excel(removed_features_path, header=None)

    # Extract the first row containing the model names
    model_names_row = removed_features.iloc[0, :].str.replace(".xlsx", "").str.strip()

    # Iterate through files in the specified directory
    for filename in os.listdir(data_dir):
        if "reduced" in filename and filename.endswith(".xlsx"):
            # Load the dataset
            file_path = os.path.join(data_dir, filename)
            df = pd.read_excel(file_path)

            # Extract model name, number of rows and columns
            model_name = filename.replace(".xlsx", "")
            num_rows = df.shape[0]
            num_columns = df.shape[1]

            # Extract the base model name without "reduced"
            base_model_name = model_name.replace("reduced_", "").strip()

            # Find the index of the model name in the first row after removing "reduced"
            base_model_names_row = model_names_row.str.replace("reduced", "").str.strip()
            if base_model_name in base_model_names_row.values:
                idx = base_model_names_row[base_model_names_row == base_model_name].index[0]
                num_dropped_features = removed_features.iloc[1:, idx].dropna().shape[0]
            else:
                num_dropped_features = 0

            # Append the details to the summary list
            summary.append({
                "Model Name": base_model_name,
                "Number of Samples": num_rows,
                "Number of Features": num_columns,
                "Number of Dropped Features": num_dropped_features
            })
    # Create a DataFrame from the summary list
    summary_df = pd.DataFrame(summary)

    # Save the DataFrame to an Excel file
    summary_df.to_excel(output_path, index=False)


if __name__ == "__main__":
    # Specify paths
    data_directory = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\models_sets\train'
    removed_features_file = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\models_sets\removed_features.xlsx'
    output_file = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\performance\models_summary.xlsx'

    # Run the function
    summarize_ml_models(data_directory, removed_features_file, output_file)

    # main()
