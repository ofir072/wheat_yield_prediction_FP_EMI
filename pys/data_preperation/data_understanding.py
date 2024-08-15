import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import seaborn as sns


def generate_boxplots(df_plot):
    # Loop through each column in the DataFrame
    for column in df_plot.columns:
        # Check if the column contains numeric data
        if pd.api.types.is_numeric_dtype(df_plot[column]):
            # Create a boxplot for the current column
            plt.figure(figsize=(8, 6))
            df_plot.boxplot(column=column)
            plt.title(f'Boxplot of {column}')
            plt.ylabel('Values')
            plt.show()
        else:
            print(f"Skipping column '{column}' as it's not numeric.")


def compute_statistics(df, file_type, filter_column=None, filter_value=None):
    # Filter DataFrame if filter_column and filter_value are provided
    if filter_column is not None and filter_value is not None:
        df = df[df[filter_column] == filter_value]
    # Create an empty DataFrame to store the statistics
    statistics_df = pd.DataFrame(columns=['Feature', 'Number of Values', 'Number of Non-NaN Values',
                                          'Percentage of Non-NaN Values', 'Maximum Value',
                                          'Minimum Value', 'Most Common Value', 'Shapiro-Wilk Test'])
    # Loop through each column in the DataFrame
    for column in df.columns:
        name = column
        # Compute statistics for the current column
        num_values = len(df[column])  # Total number of values
        num_non_nan = df[column].count()  # Number of non-NaN values
        percentage_non_nan = (num_non_nan / num_values) * 100 if num_values > 0 else 0  # Percentage of non-NaN values

        # For numeric columns, compute maximum and minimum values
        if pd.api.types.is_numeric_dtype(df[column]):
            max_value = df[column].max()  # Maximum value
            min_value = df[column].min()  # Minimum value
        else:
            max_value = None
            min_value = None

        # Compute the most common value (mode) and its percentage
        most_common_value = df[column].mode().iloc[0] if num_non_nan > 0 else None
        most_common_percentage = (df[column].value_counts(normalize=True).max()) * 100 if num_non_nan > 0 else 0

        # Shapiro-Wilk test
        shapiro_result = None  # Initialize shapiro_result to None
        if pd.api.types.is_numeric_dtype(df[column]):  # Perform the test only for numeric columns
            shapiro_test_statistic, shapiro_p_value = shapiro(df[column].dropna())
            shapiro_result = f"Statistic: {shapiro_test_statistic:.7f}, p-value: {shapiro_p_value:.7f}"

        # Append statistics to the DataFrame
        statistics_df.loc[column] = [name, num_values, num_non_nan, percentage_non_nan,
                                     max_value, min_value, f"{most_common_value} ({most_common_percentage:.2f}%)",
                                     shapiro_result]
    file_path = rf'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\data_understanding\statistics{file_type}.xlsx'
    statistics_df.to_excel(file_path, index=False)


def count_rows_with_no_blanks(df):
    # Count the number of non-blank rows in each column
    non_blank_counts = df.count(axis=1)
    # Count the number of rows where all values are non-blank
    total_rows_with_no_blanks = non_blank_counts[non_blank_counts == len(df.columns)].count()
    return total_rows_with_no_blanks


def corr_heatmap(df):
    # Filter numeric columns
    numeric_columns = df.select_dtypes(include=['int', 'float']).columns
    # Calculate correlation matrix
    correlation_matrix = df[numeric_columns].corr()
    # Create heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlation_matrix, cmap='coolwarm', xticklabels=correlation_matrix.columns,
                yticklabels=correlation_matrix.columns)
    plt.title('Correlation Heatmap')
    plt.show()


def get_third_of_month(day):
    if day.day <= 10:
        return 1
    elif day.day <= 20:
        return 2
    else:
        return 3


def check_third_of_month(df):
    # Extract month and third of the month
    df['month_third'] = df['מועד זריעה (תאריך)'].dt.month.astype(str) + '-' + df['מועד זריעה (תאריך)'].apply(get_third_of_month).astype(str)
    total_diff_count = sum(df.apply(lambda row: 1 if row['month_third'] != row['עשרת זריעה'] else 0, axis=1))
    print(total_diff_count)


def check_date_order(df, column1, column2):
    try:
        # Convert columns to datetime with errors set to 'coerce'
        df[column1] = pd.to_datetime(df[column1], errors='coerce')
        df[column2] = pd.to_datetime(df[column2], errors='coerce')
    except ValueError as e:
        print("Error converting to datetime:", e)
        return False
    # Remove rows with missing or invalid dates
    df = df.dropna(subset=[column1, column2])
    # Filter rows where date in column1 is before date in column2
    filtered_df = df[df[column1] < df[column2]]
    return filtered_df


def plot_histograms_by_group(df, group_column):
    # Ensure the group_column exists in the DataFrame
    if group_column not in df.columns:
        raise ValueError(f"Column '{group_column}' does not exist in the DataFrame.")
    # Select only numeric columns
    numeric_df = df.select_dtypes(include='number')
    # Add the group column to the numeric DataFrame
    numeric_df[group_column] = df[group_column]
    # Group the DataFrame by the specified column
    grouped = numeric_df.groupby(group_column)
    # Get the list of numeric columns
    numeric_columns = [col for col in numeric_df.columns if col != group_column]
    # Generate colors for each unique group
    unique_groups = numeric_df[group_column].unique()
    colors = sns.color_palette("husl", len(unique_groups))  # Generate a unique color for each group
    # Use a seaborn style for the plots
    sns.set(style='whitegrid')
    # Loop through each numeric column
    for column in numeric_columns:
        # Create a figure and axis for the column
        plt.figure(figsize=(10, 6))
        # Loop through each group to plot histograms
        for color, (group_name, group_data) in zip(colors, grouped):
            group_data[column].plot(kind='hist', bins=30, alpha=0.6, color=color, edgecolor='black', label=f'{group_column} = {group_name}')
        # Customize plot
        plt.title(f'Histogram of {column} by {group_column}', fontsize=16)
        plt.xlabel(column, fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.legend(title=group_column, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()


def plot_scatter_by_group(df, group_column, y_column):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include='number')

    # Add the group column and y_column to the numeric DataFrame
    numeric_df[group_column] = df[group_column]

    # Get the list of numeric columns excluding the y_column and group_column
    numeric_columns = [col for col in numeric_df.columns if col not in [group_column]]

    # Generate colors for each unique group
    unique_groups = numeric_df[group_column].unique()
    colors = sns.color_palette("husl", len(unique_groups))  # Generate a unique color for each group

    # Use a seaborn style for the plots
    sns.set(style='whitegrid')

    # Loop through each numeric column to create scatter plots against y_column
    for col_x in numeric_columns:
        # Create a figure for the scatter plot
        plt.figure(figsize=(10, 6))

        # Loop through each group to plot scatter points
        for color, group_name in zip(colors, unique_groups):
            group_data = numeric_df[numeric_df[group_column] == group_name]
            plt.scatter(group_data[col_x], group_data[y_column], alpha=0.6, color=color,
                        label=f'{group_column} = {group_name}')

        # Customize plot
        plt.title(f'Scatter Plot of {y_column} vs {col_x} by {group_column}', fontsize=16)
        plt.xlabel(col_x, fontsize=14)
        plt.ylabel(y_column, fontsize=14)
        plt.legend(title=group_column, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()


location_coded_file = r'/old_versions/ims_fe.xlsx'
df1 = pd.read_excel(location_coded_file)
columns_to_drop = ['name_check', 'station_ID', 'autumatic', 'notes', 'name', 'station_number']
df1 = df1.drop(columns=columns_to_drop)
df1 = df1[(df1['גידול נוכחי'] == 'חיטה') & (df1['ייעוד החלקה (גרעינים / שחת / תחמיץ)'] != 'גרעינים')]
# generate_boxplots(df1)
# compute_statistics(df1, '_extracted')
# print(count_rows_with_no_blanks(df1))
# check_third_of_month(df1)
# corr_heatmap(df1)
# print(check_date_order(df1, 'מועד דישון ראש (תאריך)', 'מועד זריעה (תאריך)'))
plot_histograms_by_group(df1, 'איזור גשם')
# plot_scatter_by_group(df1, 'סוג מחזור (שלחין / פלחה חרבה)', 'יבול')
