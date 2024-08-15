import datetime
import json
import requests
import pandas as pd
from datetime import datetime
from openpyxl import Workbook
import my_token
import matplotlib.pyplot as plt


# Extract the meteorology data for each station in df per sample in 2020 to 2024 - TDmin, TDmax and Rain
def import_ims_station_data_per_sample(unique_stations):
    channels_to_save = ['Rain', 'TDmin', 'TDmax']
    data_df = pd.DataFrame(columns=['Field', 'Year', 'Date', 'Channel', 'Value'])
    for row in unique_stations.iterrows():
        year = 2020
        while year < 2024:
            station_number = str(int(row['station_number']))
            # Calculate start date (14 days before the first day of the month)
            start_date = datetime.date(year, 9, 1)
            end_date = datetime.date(year + 1, 3, 31)
            interval_start = start_date.strftime('%Y/%m/%d')
            interval_end = end_date.strftime('%Y/%m/%d')
            url = "https://api.ims.gov.il/v1/envista/stations/" + station_number + "/data?from=" + interval_start + "&to=" + interval_end
            headers = {'Authorization': f'ApiToken {my_token.token}'}
            response = requests.request("GET", url, headers=headers)
            try:
                data = json.loads(response.text.encode('utf8'))
                for sample in data['data']:
                    for channel in sample['channels']:
                        if channel['name'] in channels_to_save:
                            data_df.loc[len(data_df)] = {'Field': row['station_number'],
                                                         'Year': year,
                                                         'Date': sample['datetime'],
                                                         'Channel': channel['name'],
                                                         'Value': channel['value']}
            except json.decoder.JSONDecodeError as e:
                print(f"Error decoding JSON response: {e}")
            year = year + 1
    data_df.to_excel("raw_ims_data.xlsx", index=False)


# Plot the meteorology data for all the station samples
def plot_df_ims_data(df_ims):
    # Convert 'Date' column to datetime type
    df_ims['Date'] = pd.to_datetime(df_ims['Date'])
    # Get unique fields
    unique_fields = df_ims['Field'].unique()
    # Create a subplot for each field
    fig, axs = plt.subplots(len(unique_fields), 1, figsize=(10, 6 * len(unique_fields)), sharex=True)
    # Plot each channel separately for each field
    channels_to_plot = ['Rain', 'TDmin', 'TDmax']
    for i, field in enumerate(unique_fields):
        field_data = df_ims[df_ims['Field'] == field]
        for channel in channels_to_plot:
            channel_data = field_data[field_data['Channel'] == channel]
            axs[i].plot(channel_data['Date'], channel_data['Value'], label=channel)
        # Set labels and title
        axs[i].set_ylabel('Value')
        axs[i].set_title(f'Meteorological Data - Field {field}')
        axs[i].legend()
        axs[i].grid(True)
        axs[i].autoscale(axis='y')
    # Set common x-axis label
    axs[-1].set_xlabel('Date')
    # Rotate x-axis labels for better readability
    for ax in axs:
        plt.sca(ax)
        plt.xticks(rotation=45)
    # Adjust layout
    plt.tight_layout()
    # Show plot
    plt.show()


# Extract the id of all stations belongs to the IMS
def extract_station_data_ims():
    url = "https://api.ims.gov.il/v1/envista/stations"
    headers = {'Authorization': 'ApiToken 1a901e45-9028-44ff-bd2c-35e82407fb9b'}
    response = requests.request("GET", url, headers=headers)
    try:
        data = json.loads(response.text.encode('utf8'))
        # Create a new Excel workbook and select the active worksheet
        wb = Workbook()
        ws = wb.active
        # Write headers
        ws.append(["Station Name", "Station ID"])
        # Loop over the JSON and extract name and id for each station
        for station in data:
            station_name = station['name']
            station_id = station['stationId']
            ws.append([station_name, station_id])
        # Save the workbook
        wb.save("station_data.xlsx")
    except json.decoder.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
