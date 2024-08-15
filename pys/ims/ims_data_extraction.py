import datetime
import pandas as pd
from datetime import datetime, timedelta
import os

count_time = 0
count_Empty = 0

station_350 = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\ims_data\station_processed_and_aggregated_data\350_processed_and_aggregated.xlsx'
station_82 = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\ims_data\station_processed_and_aggregated_data\82_processed_and_aggregated.xlsx'
station_236 = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\ims_data\station_processed_and_aggregated_data\236_processed_and_aggregated.xlsx'
station_208 = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\ims_data\station_processed_and_aggregated_data\208_processed_and_aggregated.xlsx'
station_79 = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\ims_data\station_processed_and_aggregated_data\79_processed_and_aggregated.xlsx'
station_59 = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\ims_data\station_processed_and_aggregated_data\59_processed_and_aggregated.xlsx'
station_58 = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\ims_data\station_processed_and_aggregated_data\58_processed_and_aggregated.xlsx'
station_381 = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\ims_data\station_processed_and_aggregated_data\381_processed_and_aggregated.xlsx'
station_349 = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\ims_data\station_processed_and_aggregated_data\349_processed_and_aggregated.xlsx'
station_98 = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\ims_data\station_processed_and_aggregated_data\98_processed_and_aggregated.xlsx'
station_112 = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\ims_data\station_processed_and_aggregated_data\112_processed_and_aggregated.xlsx'
preferences = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\ims_data\near_station_ims.xlsx'


def generate_dates_for_year(year):
    start_date = datetime(year - 1, 9, 1)  # 1st September of the previous year
    end_date = datetime(year, 3, 31)  # 31st March of the given year
    date_list = [(start_date + timedelta(days=x)).date() for x in range((end_date - start_date).days + 2)]
    return date_list


def fill_dates_for_year(data, year):
    date_list = generate_dates_for_year(year)
    missing_rows = []
    # Ensure the column is of datetime type
    date_column = data['תאריך ושעה (שעון קיץ)']
    for date in date_list:
        if not any(date_column == date):
            new_row = {column: 'Empty' for column in data.columns}
            new_row['תאריך ושעה (שעון קיץ)'] = date
            missing_rows.append(new_row)
    if missing_rows:
        missing_df = pd.DataFrame(missing_rows)
        data = pd.concat([data, missing_df], ignore_index=True)
    return data


def agg_ims_data_to_daily(raw_data_ims):
    raw_data_ims = raw_data_ims.replace({'-': pd.NA, 'Empty': pd.NA})
    daily_aggregate = raw_data_ims.groupby([raw_data_ims['תאריך ושעה (שעון קיץ)']]).agg({
        'לחות יחסית (%)': lambda x: x.mean() if (x.count() / 144) >= 0.7 else "Empty",
        'טמפרטורת מקסימום (C°)': lambda x: x.max() if (x.count() / 144) >= 0.7 else "Empty",
        'טמפרטורת מינימום (C°)': lambda x: x.min() if (x.count() / 144) >= 0.7 else "Empty",
        'כמות גשם (מ"מ)': lambda x: x.mean() if (x.count() / 144) >= 0.7 else "Empty",
    }).reset_index()
    return daily_aggregate


def process_and_aggregate_excel_file(filepath):
    data = pd.read_excel(filepath)
    data['תאריך ושעה (שעון קיץ)'] = pd.to_datetime(data['תאריך ושעה (שעון קיץ)'], dayfirst=True).dt.date
    for year in range(2021, 2025):
        data = fill_dates_for_year(data, year)
    aggregated_data = agg_ims_data_to_daily(data)
    output_dir = r'/data/ims_data/station_processed_and_aggregated_data'
    filename = os.path.basename(filepath)
    output_path = os.path.join(output_dir, filename.replace('.xlsx', '_processed_and_aggregated.xlsx'))
    aggregated_data.to_excel(output_path, index=False)
    return output_path


def pre_processing_ims_raw_data():
    directory = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\ims_data\station_raw_data'
    # Iterate over files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.xlsx'):  # Filter out Excel files
            filepath = os.path.join(directory, filename)
            process_and_aggregate_excel_file(filepath)


def find_variable_by_name(name):
    if name in globals():
        return globals()[name]
    else:
        return None


def complete_ims_data_from_files(primary_file_path, secondary_file_path, tertiary_file_path):
    global count_time
    global count_Empty
    # טעינת נתונים משלושת הקבצים
    primary_data = pd.read_excel(primary_file_path)
    secondary_data = pd.read_excel(secondary_file_path)
    tertiary_data = pd.read_excel(tertiary_file_path)
    # בדיקת חוסרים ועדכון נתונים לפי עדיפות
    for index, row in primary_data.iterrows():
        for col in primary_data.columns[1:]:
            if row[col] == 'Empty':
                # חיפוש התאריך בקובץ השני
                corresponding_row_secondary = secondary_data[
                    secondary_data['תאריך ושעה (שעון קיץ)'] == row['תאריך ושעה (שעון קיץ)']]
                if corresponding_row_secondary[col].iloc[0] != 'Empty':
                    primary_data.at[index, col] = corresponding_row_secondary[col].iloc[0]
                    count_time = count_time + 1
                else:  # חיפוש התאריך בקובץ השלישי אם לא נמצא בשני
                    corresponding_row_tertiary = tertiary_data[
                        tertiary_data['תאריך ושעה (שעון קיץ)'] == row['תאריך ושעה (שעון קיץ)']]
                    if corresponding_row_tertiary[col].iloc[0] != 'Empty':
                        primary_data.at[index, col] = corresponding_row_tertiary[col].iloc[0]
                        count_time = count_time + 1
                    else:
                        primary_data.at[index, col] = 'Empty'
                        print(
                            f"אין נתונים בשלושת הקבצים עבור התאריך: {row['תאריך ושעה (שעון קיץ)'].strftime('%d/%m/%Y')}")
                        count_Empty = count_Empty + 1
    # שמירת הנתונים המעודכנים לקובץ חדש
    output_dir = r'/data/ims_data/station_complete_data'
    filename = os.path.basename(primary_file_path)
    output_path = os.path.join(output_dir, filename.replace('.xlsx', '_completed.xlsx'))
    primary_data.to_excel(output_path, index=False)
    return output_path


def set_preferences(preferences_path):
    preferences_station = pd.read_excel(preferences_path)
    for index, row in preferences_station.iterrows():
        primary_file_path = find_variable_by_name(f'station_{row[0]}')
        secondary_file_path = find_variable_by_name(f'station_{row[1]}')
        tertiary_file_path = find_variable_by_name(f'station_{row[2]}')
        complete_ims_data_from_files(primary_file_path, secondary_file_path, tertiary_file_path)
        print(f'Updated file saved as: {primary_file_path}')


pre_processing_ims_raw_data()
set_preferences(preferences)
