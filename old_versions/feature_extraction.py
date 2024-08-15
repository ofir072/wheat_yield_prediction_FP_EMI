from datetime import timedelta
import numpy as py
import pandas as pd
import warnings
import numpy as np


# Extract the meteorology data for each station in df per week - TDmin, TDmax and Rain
def extract_ims_features(filtered_df, days_bar):
    filtered_df['מועד זריעה (תאריך)'] = pd.to_datetime(filtered_df['מועד זריעה (תאריך)'], dayfirst=True).dt.date
    # Create new columns in filtered_df to store the meteorology data for each week
    num_weeks = 8  # Number of weeks including the week of the starting date
    data_type = ['Moist', 'TDmax', 'TDmin', 'Rain']
    for i in range(-2, num_weeks):
        filtered_df.loc[:, f'Moist W{i}'] = None
        filtered_df.loc[:, f'TDmax W{i}'] = None
        filtered_df.loc[:, f'TDmin W{i}'] = None
        filtered_df.loc[:, f'Rain W{i}'] = None
    for index, row in filtered_df.iterrows():
        station_number = str(int(row['station_number']))
        path = rf'C:\Users\Eli Levi\PycharmProjects\wheat_yield_prediction_FP_EMI\data\ims_data\station_complete_data\{station_number}_processed_and_aggregated_completed.xlsx'
        station_data = pd.read_excel(path)
        station_data['תאריך ושעה (שעון קיץ)'] = pd.to_datetime(station_data['תאריך ושעה (שעון קיץ)'], dayfirst=True).dt.date
        # Calculate start date (14 days before the first day of the month)
        start_date = row['מועד זריעה (תאריך)'] - timedelta(days=14)
        # Calculate end date (56 days after the first day of the month)
        end_date = row['מועד זריעה (תאריך)'] + timedelta(days=56)
        # Split date range into intervals of 7 days
        current_date = start_date
        week_index = -2
        while current_date < end_date:
            interval_end = current_date + timedelta(days=6)
            filtered_data = station_data[(station_data['תאריך ושעה (שעון קיץ)'] >= current_date) & (
                        station_data['תאריך ושעה (שעון קיץ)'] <= interval_end)]
            filtered_data = filtered_data.drop(filtered_data.columns[0], axis=1)
            for j, column in enumerate(filtered_data.columns):
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", message="Downcasting behavior in `replace` is deprecated*")
                    try:
                        filtered_data[column] = filtered_data[column].replace({'Empty': pd.NA})
                    except Warning:
                        # Handle the warning, or just ignore it
                        pass
                if filtered_data[column].count() > days_bar:
                    filtered_values = filtered_data[column][pd.notna(filtered_data[column])]
                    filtered_df.at[index, f'{data_type[j]} W{week_index}'] = py.average(filtered_values)
                else:
                    filtered_df.at[index, f'{data_type[j]} W{week_index}'] = "Empty"
                    print(f'In filed {index}, the {data_type[j]} W{week_index} is Empty')
            current_date += timedelta(days=7)
            week_index += 1
    filtered_df = min_max_scale_to_station_data(filtered_df)
    output_path = rf'C:\Users\Eli Levi\PycharmProjects\wheat_yield_prediction_FP_EMI\data\FE\ims_fe.xlsx'
    filtered_df.to_excel(output_path, index=False)
    return filtered_df


def column_to_oneHotEncode(df4, column_name):

    unique_values = df4[column_name].unique()

    for value in unique_values:
        binary_column_name = f"{column_name}-{value}"
        df4[binary_column_name] = df4[column_name].apply(lambda x: 1 if x == value else 0)
    return df4

def df_to_oneHotEncode(df3):
    df3 = column_to_oneHotEncode(df3, "סוג מחזור (שלחין / פלחה חרבה)")
    df3.drop("סוג מחזור (שלחין / פלחה חרבה)", axis=1, inplace=True)
    df3 = column_to_oneHotEncode(df3, "מגדל")
    df3.drop("מגדל", axis=1, inplace=True)
    df3 = column_to_oneHotEncode(df3, "אזור")
    df3.drop("אזור", axis=1, inplace=True)
    df3 = column_to_oneHotEncode(df3, "גידול נוכחי")
    df3.drop("גידול נוכחי", axis=1, inplace=True)
    df3 = column_to_oneHotEncode(df3, "ייעוד החלקה (גרעינים / שחת / תחמיץ)")
    df3.drop("ייעוד החלקה (גרעינים / שחת / תחמיץ)", axis=1, inplace=True)
    df3 = oneHotEncode_2(df3, "סוג זבל")
    df3.drop("סוג זבל", axis=1, inplace=True)
    df3 = oneHotEncode_2(df3, "עיבוד יסוד")
    df3.drop("עיבוד יסוד", axis=1, inplace=True)
    # df3 = oneHotEncode_2(df3, "גידול קודם") #need to sort this col
    df3 = column_to_oneHotEncode(df3, "השקייה (מושקה/בעל)")
    df3.drop("השקייה (מושקה/בעל)", axis=1, inplace=True)
    df3 = column_to_oneHotEncode(df3, "איזור גשם")
    df3.drop("איזור גשם", axis=1, inplace=True)
    df3 = column_to_oneHotEncode(df3, "סוג כרב")
    df3.drop("סוג כרב", axis=1, inplace=True)
    df3 = column_to_oneHotEncode(df3, "זן משוכלל")
    df3.drop("זן משוכלל", axis=1, inplace=True)
    return df3

def oneHotEncode_2(df_toEncode, colName): #need to fix the function
    unique_values = set()
    df_toEncode[colName].str.split(' \+ ').apply(unique_values.update)

    # Create binary columns for each unique value
    for value in unique_values:
        binary_column_name = f"{colName}-{value}"
        df_toEncode[binary_column_name] = df_toEncode[colName].apply(lambda x: 1 if value in x else 0)
    return df_toEncode

def min_max_scale_column(df5, col_name):

    min_val = df5[col_name].min()
    max_val = df5[col_name].max()
    df5[col_name] = (df5[col_name] - min_val) / (max_val - min_val)
    return df5

def cols_to_min_max_scale(df6):
    df6 = min_max_scale_column(df6, "גודל חלקה (דונם)")
    df6 = min_max_scale_column(df6, "זיבול- כמות (קוב/ד')")
    df6 = min_max_scale_column(df6, "דישון יסוד (יח' חנקן / ד')")
    df6 = min_max_scale_column(df6, "דישון ראש (יח' חנקן / ד')")
    df6 = min_max_scale_column(df6, "גשם בעונה (ממ)")
    df6 = min_max_scale_column(df6, "השקייה הנבטה (קוב/ד')")
    df6 = min_max_scale_column(df6, "השקייה במהלך הגידול (קוב/ד')")
    #df6 = min_max_scale_column(df6, "גשם בעונה (ממ)")
    #df6 = min_max_scale_column(df6, "השקייה הנבטה (קוב/ד')")

    return df6

def min_max_scale_to_station_data(df7):
    df7 = min_max_scale_column(df7, "Moist W-2")
    df7 = min_max_scale_column(df7, "TDmax W-2")
    df7 = min_max_scale_column(df7, "TDmin W-2")
    df7 = min_max_scale_column(df7, "Rain W-2")
    df7 = min_max_scale_column(df7, "Moist W-1")
    df7 = min_max_scale_column(df7, "TDmax W-1")
    df7 = min_max_scale_column(df7, "TDmin W-1")
    df7 = min_max_scale_column(df7, "Rain W-1")
    df7 = min_max_scale_column(df7, "Moist W0")
    df7 = min_max_scale_column(df7, "TDmax W0")
    df7 = min_max_scale_column(df7, "TDmin W0")
    df7 = min_max_scale_column(df7, "Rain W0")
    df7 = min_max_scale_column(df7, "Moist W1")
    df7 = min_max_scale_column(df7, "TDmax W1")
    df7 = min_max_scale_column(df7, "TDmin W1")
    df7 = min_max_scale_column(df7, "Rain W1")
    df7 = min_max_scale_column(df7, "Moist W2")
    df7 = min_max_scale_column(df7, "TDmax W2")
    df7 = min_max_scale_column(df7, "TDmin W2")
    df7 = min_max_scale_column(df7, "Rain W2")
    df7 = min_max_scale_column(df7, "Moist W3")
    df7 = min_max_scale_column(df7, "TDmax W3")
    df7 = min_max_scale_column(df7, "TDmin W3")
    df7 = min_max_scale_column(df7, "Rain W3")
    df7 = min_max_scale_column(df7, "Moist W4")
    df7 = min_max_scale_column(df7, "TDmax W4")
    df7 = min_max_scale_column(df7, "TDmin W4")
    df7 = min_max_scale_column(df7, "Rain W4")
    df7 = min_max_scale_column(df7, "Moist W5")
    df7 = min_max_scale_column(df7, "TDmax W5")
    df7 = min_max_scale_column(df7, "TDmin W5")
    df7 = min_max_scale_column(df7, "Rain W5")
    df7 = min_max_scale_column(df7, "Moist W6")
    df7 = min_max_scale_column(df7, "TDmax W6")
    df7 = min_max_scale_column(df7, "TDmin W6")
    df7 = min_max_scale_column(df7, "Rain W6")
    df7 = min_max_scale_column(df7, "Moist W7")
    df7 = min_max_scale_column(df7, "TDmax W7")
    df7 = min_max_scale_column(df7, "TDmin W7")
    df7 = min_max_scale_column(df7, "Rain W7")
    #df6 = min_max_scale_column(df6, "גשם בעונה (ממ)")
    #df6 = min_max_scale_column(df6, "השקייה הנבטה (קוב/ד')")

    return df7



location_coded_file = r'C:\Users\Eli Levi\PycharmProjects\wheat_yield_prediction_FP_EMI\data\ims_data\fields_locations.xlsx'
fields_data = r'C:\Users\Eli Levi\PycharmProjects\wheat_yield_prediction_FP_EMI\data\דאטה מעובד.xlsx'

# Read the Excel files into DataFrames
df1 = pd.read_excel(location_coded_file)
df2 = pd.read_excel(fields_data)
df2['מועד זריעה (תאריך)'] = df2['מועד זריעה (תאריך)'].replace({"נובמבר": np.nan, '': np.nan})
df2.dropna(subset=['מועד זריעה (תאריך)'], inplace=True)
df1['אזור'] = df1['אזור'].str.strip()
df2['אזור'] = df2['אזור'].str.strip()


df2 = df_to_oneHotEncode(df2)
df2 = cols_to_min_max_scale(df2)
df2.to_excel(rf'C:\Users\Eli Levi\PycharmProjects\wheat_yield_prediction_FP_EMI\data\FE\DataSet.xlsx', index=False, engine='openpyxl')  # Specify engine if necessary


# Merge the DataFrames based on the columns 'אזור' and 'field_location'
merged_df = pd.merge(df1, df2, on='אזור', how='right')
extract_ims_features(merged_df, 4)



