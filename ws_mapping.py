import pandas as pd
import numpy as np
import os
from geopy.distance import geodesic
import geopandas as gpd
import matplotlib.pyplot as plt


weather_stations_df = pd.read_csv('TX_weather.csv')
# print(weather_stations_df.dtypes)
# print(weather_stations_df.columns)
print('TX weather stations: ', weather_stations_df.shape)
# TX weather stations:  (167, 6)

weather_stations_df = pd.read_csv('station_name.csv')
print('Weather stations (all): ', weather_stations_df.shape) # (6510, 6)

buses_df = pd.read_excel('bus_config.xlsx',engine='openpyxl')
# ws_index = pd.read_csv('Weatherstation_index.csv')
print('TX buses (all): ', buses_df.shape) #(6717, 24)

generator_df = pd.read_csv('generators.csv')
print('generators: ', generator_df.shape) # (731, 39)

# Get the name for our selected ws
temp = pd.read_csv('TEST_Temp_2016.csv')
station_list = temp.columns.unique()
# print(len(station_list))
ws_df_137=pd.DataFrame()
for i in range(len(station_list)):
    filtered_df = weather_stations_df[weather_stations_df['Name'].str.contains(station_list[i])]  # Filter stations
    ws_df_137 = pd.concat([ws_df_137, filtered_df], ignore_index=True)  # Concatenate with existing DataFrame
    # ws_df = ws_df.append(weather_stations_df[weather_stations_df['NAME'].str.contains(station_list[i])])
# print(len(ws_df)) # 137
# path = os.path.join(os.curdir, 'upload', 'ws_137.csv')
# pd.DataFrame.to_csv(ws_df_137, path, index=False)


with open('station_name.csv', 'r', encoding='utf-8', errors='ignore') as file:
# (6510, 6)
# with open('Weatherstation_index.csv', 'r', encoding='utf-8', errors='ignore') as file:
# (1046, 27)
    # CSV file saved with a header row, do not skip empty indices
    weather_stations_df = pd.read_csv(file,low_memory=False, sep=",",header=0)
print('weather stations (all): ', weather_stations_df.shape)
# print(weather_stations_df['Region'].value_counts())
# print(weather_stations_df.dtypes)
#create tx ws dataframe
tx_ws_df = weather_stations_df[weather_stations_df['Region']=='TX'].copy()
print('TX weather stations: ', tx_ws_df.shape) #(167,6)
# print(tx_ws_df.dtypes)

# for mapping
texas_shp = gpd.read_file('WeatherOPF/State.shp')

def bus_ws_mapping():
    with open('mapping.csv', 'r', encoding='utf-8', errors='ignore') as file:
        mapping_df = pd.read_csv(file, low_memory=False, sep=',', header=0, index_col=0)
        print(mapping_df)
        distances = mapping_df['Distance(km)']
        distance_series = distances.str.replace(' km', '', regex=True)
        distance_series = pd.to_numeric(distance_series, errors='coerce')  # Convert to numeric (handle errors)

        print(distance_series, distance_series.dtype)
        print(distance_series.describe())
        # print(mapping_df.dtypes)

        selected_stations = mapping_df['WS Name(ICAO)']
        print(selected_stations.value_counts())
        if mapping_df['Bus ID'].nunique() == buses_df.shape[0]:
            print('All unique buses are in the mapping df.')
# bus_ws_mapping()

def map_visual():
    with open('mapping.csv', 'r', encoding='utf-8', errors='ignore') as file:
        mapping_df = pd.read_csv(file, low_memory=False, sep=',', header=0, index_col=0)
    selected_loc = mapping_df['WS Name(ICAO)']
    selected_loc = selected_loc.to_frame().drop_duplicates().rename(columns={'WS Name(ICAO)':'Name'})
    selected_loc_df = selected_loc.join(tx_ws_df.set_index('Name'), on='Name', how='inner')
    selected_loc_df.to_csv('selected_ws.csv', index=False)

    # Load Texas shapefile
    # texas_shp = gpd.read_file('State.shp')
    # Convert weather stations and buses DataFrames into geopandas GeoDataFrames
    weather_gdf = gpd.GeoDataFrame(tx_ws_df, geometry=gpd.points_from_xy(tx_ws_df['Longitude'], tx_ws_df['Latitude']))
    power_grid_gdf = gpd.GeoDataFrame(buses_df, geometry=gpd.points_from_xy(buses_df['Substation Longitude'], buses_df['Substation Latitude']))
    mapping_gdf = gpd.GeoDataFrame(selected_loc_df, geometry=gpd.points_from_xy(selected_loc_df['Longitude'], selected_loc_df['Latitude']))

    fig, ax = plt.subplots(figsize=(10,10))
    texas_shp.plot(ax=ax, color='lightgray', edgecolor='black')
    power_grid_gdf.plot(ax=ax, color='red', markersize=10, label='Power Grid')
    mapping_gdf.plot(ax=ax, color='blue', markersize=10, label='Selected Weather Stations')
    weather_gdf.plot(ax=ax, color='green', markersize=10, label='All Weather Stations')
    ax.set_title('Weather Stations and Buses in 6717-bus Texas Grid')
    plt.legend()
    plt.savefig('ws_and_bus.png')
# map_visual()

def map_latent_ws():
    with open('ws_137.csv', 'r', encoding='utf-8', errors='ignore') as file:
        ws_137 = pd.read_csv(file, low_memory=False, sep=',', header=0, index_col=None)
    print(ws_137.shape)
    texas_shp_file = gpd.read_file("WeatherOPF/State.shp")

    inact_wind = pd.read_csv('index_inactive_wind.csv', encoding='utf-8', header=None, index_col=None)
    inact_wind_index = inact_wind.iloc[:,1]
    inact_ws = ws_137.iloc[inact_wind_index,:]
    print(inact_ws.shape)
    act_solar = pd.read_csv('index_active_solar.csv', encoding='utf-8', header=None, index_col=None)
    act_solar_index = act_solar.iloc[:,1]
    act_solar_ws = ws_137.iloc[act_solar_index,:]
    print(act_solar_ws.shape)
    inact_wind_gdf = gpd.GeoDataFrame(inact_ws, geometry=gpd.points_from_xy(inact_ws['Longitude'], inact_ws['Latitude']))
    act_solar_gdf = gpd.GeoDataFrame(act_solar_ws, geometry=gpd.points_from_xy(act_solar_ws['Longitude'], act_solar_ws['Latitude']))
    ws_all_gdf = gpd.GeoDataFrame(ws_137, geometry=gpd.points_from_xy(ws_137['Longitude'], ws_137['Latitude']))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    texas_shp_file.plot(ax=ax, color='lightgray', edgecolor='black')
    inact_wind_gdf.plot(ax=ax, color='red', markersize=10, label='Inactive Wind')
    act_solar_gdf.plot(ax=ax, color='blue', markersize=10, label='Active solar')
    ws_all_gdf.plot(ax=ax, color='green', markersize=2, label='All Weather Stations')
    plt.legend()
    plt.savefig('ws_latent.png')
# map_latent_ws()

def active_ws():
    print('active latent weather plotting...')
    with open('ws_137.csv', 'r', encoding='utf-8', errors='ignore') as file:
        ws_137 = pd.read_csv(file, low_memory=False, sep=',', header=0, index_col=None)
    print(ws_137.shape) # (137,6)
    texas_shp_file = gpd.read_file("WeatherOPF/State.shp")
    ws_all_gdf = gpd.GeoDataFrame(ws_137, geometry=gpd.points_from_xy(ws_137['Longitude'], ws_137['Latitude']))

    temp = pd.read_csv('temp_active.csv', encoding='utf-8', header=0, index_col=None)
    wind = pd.read_csv('wind_active.csv', encoding='utf-8', header=0, index_col=None)
    cloud = pd.read_csv('cloud_active.csv', encoding='utf-8', header=0, index_col=None)
    active_temp_idx = list(temp.columns.values)
    print(active_temp_idx)
    # fig, ax = plt.subplots(figsize=(10, 10))

    # # ax.axis('off')
    # texas_shp_file.plot(ax=ax, color='lightgray', edgecolor='black')
    # ws_all_gdf.plot(ax=ax, markersize=2, label='All Weather Stations')
    # feat_name = ['temp', 'wind', 'cloud']
    # filenames = [os.path.join(os.curdir, f"figs/fig_{j+1}.png") for j in range(3)]  
    # filename = os.path.join(os.curdir, f'figs/fig_3.png')
    # # os.makedirs(os.path.join(os.curdir, 'figs'), exist_ok=True)
    # # for z,feat in enumerate([temp, wind, cloud]):
    #     # print(f'{str(feat)} shape: ', feat.shape)
    # for col in np.arange(cloud.shape[1]):
    #     # print('col', col) 
    #     act_latent = cloud.iloc[:,col].dropna()
    #     # print(act_latent, act_latent.shape)
    #     act_ws = ws_137.iloc[act_latent,:]
    #     # print(act_ws)
    #     act_gdf = gpd.GeoDataFrame(act_ws, geometry=gpd.points_from_xy(act_ws['Longitude'], act_ws['Latitude']))
    #     act_gdf.plot(ax=ax, markersize=15, label=(f'Group {j+1}') for j in range(3))
    #     # print(cloud.columns[col])
    # plt.legend()
    # plt.savefig(filename)
    
    # fig1, ax1 = plt.subplots(figsize=(10, 10))  # Individual figure for each plot
    # fig2, ax2 = plt.subplots(figsize=(10, 10))
    # fig3, ax3 = plt.subplots(figsize=(10, 10))
    # for ax in [ax1, ax2, ax3]:

    # texas_shp_file.plot(ax=[ax1, ax2, ax3], color='lightgray', edgecolor='black')
    # ws_all_gdf = gpd.GeoDataFrame(ws_137, geometry=gpd.points_from_xy(ws_137['Longitude'], ws_137['Latitude']))

    feat_name = ['temp', 'wind', 'cloud']
    filenames = [os.path.join("figs", f"fig_{j+1}.png") for j in range(3)]

    for z, feat in enumerate(zip([temp, wind, cloud], feat_name)):
        data, name = feat
        groups = [f'Group {i+1}' for i in range(data.shape[1])]  # Create group labels
        fig, ax = plt.subplots(figsize=(10,10))
        texas_shp_file.plot(ax=ax, color='lightgray', edgecolor='black')
        ws_all_gdf.plot(ax=ax, markersize=2, label='All Weather Stations')

        for col, group in zip(np.arange(data.shape[1]), groups):
            act_latent = data.iloc[:, col].dropna()
            act_ws = ws_137.iloc[act_latent, :]
            act_gdf = gpd.GeoDataFrame(act_ws, geometry=gpd.points_from_xy(act_ws['Longitude'], act_ws['Latitude']))
            act_gdf.plot(ax=ax, markersize=15, label=group)  # Use group label
        plt.legend(loc='upper left')  # Adjust legend location if needed
        plt.savefig(filenames[z])

# You can optionally display the plots using plt.show()
# plt.show()

active_ws()


def cal_dis_ws_bus():
    near_stations = []
    for bus_index, bus_row in buses_df.iterrows():
        # print(bus_index, bus_row)
        min_dist = float('inf')
        near_station = None
        for weather_index, weather_row in tx_ws_df.iterrows():
            # print(weather_index, weather_row)
            bus_coords = (bus_row['Substation Latitude'], bus_row['Substation Longitude'])
            weather_coords = (weather_row['Latitude'], weather_row['Longitude'])

            distance = geodesic(bus_coords, weather_coords)
            if distance < min_dist:
                min_dist = distance
                near_station = weather_row['Name']

        near_stations.append((bus_row['Name'], bus_row['Number'], near_station, min_dist))
        mapping_df = pd.DataFrame(near_stations, columns=['Bus Name', 'Bus ID', 'WS Name(ICAO)', 'Distance(km)' ])
        print(mapping_df.shape)
    pd.DataFrame.to_csv(mapping_df, 'mapping.csv')
# cal_dis_ws_bus()

def fuel_type():
    # generator_df = pd.DataFrame()
    generator_df['Number of Bus'] = generator_df['Number of Bus'].astype(str)
    buses_df['Number'] = buses_df['Number'].astype(str)

    # Merge the two DataFrames on the respective columns
    gen_merged_df = pd.merge(generator_df, buses_df, left_on='Number of Bus', right_on='Number', how='left')
    # print(gen_merged_df.dtypes, gen_merged_df.shape) # (731,63)

    name_list = gen_merged_df.columns.values
    # print(name_list)

    selected_columns = ['Number of Bus', 'Name of Bus', 'Substation Latitude', 'Substation Longitude', 'Fuel Type Integer (Generic)', 'Fuel Type (Generic)', 'Gen MW_x','City', 'State']
    gen_short_df = gen_merged_df[selected_columns]
    # gen_short_df

    # Find the rows where the 'Fuel Type (Generic)' column is wind or solar
    renewable_gen_df = gen_short_df[(gen_short_df['Fuel Type (Generic)'] == 'Wind') | (gen_short_df['Fuel Type (Generic)'] == 'Solar')]
    solar_gen_df = gen_short_df[(gen_short_df['Fuel Type (Generic)'] == 'Solar')]
    wind_gen_df = gen_short_df[(gen_short_df['Fuel Type (Generic)'] == 'Wind')]
    # print(renewable_gen_df) # (189.9)

    # Convert weather stations and buses DataFrames into geopandas GeoDataFrames
    weather_gdf = gpd.GeoDataFrame(tx_ws_df, geometry=gpd.points_from_xy(tx_ws_df['Longitude'], tx_ws_df['Latitude']))
    power_grid_gdf = gpd.GeoDataFrame(buses_df, geometry=gpd.points_from_xy(buses_df['Substation Longitude'], buses_df['Substation Latitude']))
    renewable_gdf = gpd.GeoDataFrame(renewable_gen_df, geometry=gpd.points_from_xy(renewable_gen_df['Substation Longitude'], renewable_gen_df['Substation Latitude']))
    solar_gdf = gpd.GeoDataFrame(solar_gen_df, geometry=gpd.points_from_xy(solar_gen_df['Substation Longitude'], solar_gen_df['Substation Latitude']))
    wind_gdf = gpd.GeoDataFrame(wind_gen_df, geometry=gpd.points_from_xy(wind_gen_df['Substation Longitude'], wind_gen_df['Substation Latitude']))

    fig, ax = plt.subplots(figsize=(10, 10))
    texas_shp.plot(ax=ax, color='lightgray', edgecolor='black')
    power_grid_gdf.plot(ax=ax, color='red', markersize=10, label='Grid Buses')
    weather_gdf.plot(ax=ax, color='blue', markersize=10, label='Weather Stations')
    wind_gdf.plot(ax=ax, color='green', markersize=15, label='Wind farms')
    solar_gdf.plot(ax=ax, color='yellow', markersize=15, label='Solar farms')
    ax.set_title("Weather Stations and Power Grid in Texas")
    plt.legend()
    plt.savefig('ws_pg.png')

# fuel_type()
