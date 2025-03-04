import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load and preprocess data
coords_df = pd.read_csv('county.csv')
coords_df['lat'] = pd.to_numeric(coords_df['lat'], errors='coerce')
coords_df['lng'] = pd.to_numeric(coords_df['lng'], errors='coerce')
coords_df.dropna(subset=['lat', 'lng'], inplace=True)
coords_df['county_state'] = coords_df.apply(lambda row: f"{row['county_full']}, {row['state_id']}", axis=1)
coords_df.set_index('county_state', inplace=True)

# Load and preprocess aging data
aging_df = pd.read_csv('aging.csv')
aging_df['Crude Rate'] = aging_df['Crude Rate'].str.replace(r'\s*\(Unreliable\)', '', regex=True)
aging_df['Crude Rate'] = pd.to_numeric(aging_df['Crude Rate'], errors='coerce')

# Exclude unreliable entries
reliable_aging_df = aging_df[~aging_df['Crude Rate'].isna()]

# Exclude rows with racial or gender data and retain only the county-level data
overall_aging_df = reliable_aging_df[reliable_aging_df['Race'].isna() & reliable_aging_df['Gender'].isna()].copy()
overall_aging_df.set_index('County', inplace=True)
overall_aging_df['Crude Rate'] *= 0.1

def get_crude_rate(county_state):
    return overall_aging_df.loc[county_state, 'Crude Rate'] if county_state in overall_aging_df.index else np.nan

coords_df['Crude Rate'] = coords_df.index.map(get_crude_rate)
coords_df.dropna(subset=['Crude Rate'], inplace=True)

# Fit NearestNeighbors model
X = coords_df[['lat', 'lng']].values
nbrs = NearestNeighbors(n_neighbors=8)
nbrs.fit(X)

# Set the number of test runs and intensity levels
num_runs = 50
intensity_levels = [0.1, 0.2, 0.3, 0.4, 0.5]  # Intensity levels for the complex average weights

# Initialize results dictionary for MAE and intensity levels
mae_results_per_intensity = {intensity: [] for intensity in intensity_levels}

# Loop through each intensity level
for intensity in intensity_levels:
    print(f"\nTesting with complex average weight intensity: {intensity}")

    # Loop through multiple test runs
    for i in range(num_runs):
        # Randomly select a county
        random_county = coords_df.sample(n=1).iloc[0]
        county_name = random_county.name
        lat = random_county['lat']
        lng = random_county['lng']

        print(f"\nTest Run {i + 1}")
        print(f"Randomly selected county: {county_name}")
        print(f"Latitude: {lat}")
        print(f"Longitude: {lng}")

        # Find crude rate in aging_df
        if county_name in overall_aging_df.index:
            actual_crude_rate = overall_aging_df.loc[county_name, 'Crude Rate']
        else:
            actual_crude_rate = np.nan
        print(f"Actual Crude Rate: {actual_crude_rate:.2f}" if not np.isnan(actual_crude_rate) else "Actual Crude Rate: Not Available")

        # Find nearest neighbors
        distances, indices = nbrs.kneighbors([[lat, lng]])
        neighbor_indices = indices[0][1:]  # Exclude the county itself
        neighbor_counties = coords_df.iloc[neighbor_indices]

        # Calculate average crude rate for the 7 nearest neighbors
        neighbor_crude_rates = []
        for index in neighbor_counties.index:
            if index in overall_aging_df.index:
                neighbor_crude_rate = overall_aging_df.loc[index, 'Crude Rate']
                neighbor_crude_rates.append(neighbor_crude_rate)
        
        avg_crude_rate_neighbors = np.mean(neighbor_crude_rates) if neighbor_crude_rates else np.nan

        print("\n7 Nearest Neighbors:")
        for index, row in neighbor_counties.iterrows():
            neighbor_name = row.name
            neighbor_lat = row['lat']
            neighbor_lng = row['lng']

            # Find crude rate in aging_df
            if neighbor_name in overall_aging_df.index:
                neighbor_crude_rate = overall_aging_df.loc[neighbor_name, 'Crude Rate']
            else:
                neighbor_crude_rate = np.nan

            print(f"County: {neighbor_name}, Latitude: {neighbor_lat}, Longitude: {neighbor_lng}, Crude Rate: {neighbor_crude_rate:.2f}" if not np.isnan(neighbor_crude_rate) else f"County: {neighbor_name}, Latitude: {neighbor_lat}, Longitude: {neighbor_lng}, Crude Rate: Not Available")
        
        print(f"Average Crude Rate of 7 Nearest Neighbors: {avg_crude_rate_neighbors:.2f}" if not np.isnan(avg_crude_rate_neighbors) else "Average Crude Rate of 7 Nearest Neighbors: Not Available")

        # Calculate weighted average using the complex average
        weighted_crude_rates = []
        for index in neighbor_counties.index:
            if index in overall_aging_df.index:
                neighbor_crude_rate = overall_aging_df.loc[index, 'Crude Rate']
                simple_avg = avg_crude_rate_neighbors
                weight = (neighbor_crude_rate - simple_avg) / simple_avg
                weighted_crude_rate = neighbor_crude_rate + (intensity * simple_avg * weight)
                weighted_crude_rates.append(weighted_crude_rate)
        
        complex_avg_crude_rate = np.mean(weighted_crude_rates) if weighted_crude_rates else np.nan
        
        # Use the complex average for prediction
        predicted_crude_rate = complex_avg_crude_rate
        
        print(f"Complex Average Crude Rate (Weighted): {complex_avg_crude_rate:.2f}" if not np.isnan(complex_avg_crude_rate) else "Complex Average Crude Rate: Not Available")

        # Calculate and store MAE
        if not np.isnan(actual_crude_rate) and not np.isnan(predicted_crude_rate):
            mae = mean_absolute_error([actual_crude_rate], [predicted_crude_rate])
            mae_results_per_intensity[intensity].append(mae)
        else:
            mae_results_per_intensity[intensity].append(np.nan)

# Calculate average MAE for each intensity level
average_mae_per_intensity = {intensity: np.nanmean(mae_results_per_intensity[intensity]) for intensity in intensity_levels}

# Plot average MAE for each intensity level
plt.figure(figsize=(12, 6))
plt.plot(list(average_mae_per_intensity.keys()), list(average_mae_per_intensity.values()), marker='o', linestyle='-', color='b')
plt.title('Average MAE for Each Intensity Level of Complex Average Weights')
plt.xlabel('Complex Average Weight Intensity')
plt.ylabel('Average Mean Absolute Error')
plt.grid(True)
plt.show()

# Print average MAE results
print("\nAverage MAE for each intensity level:")
for intensity, avg_mae in average_mae_per_intensity.items():
    print(f"Intensity {intensity}: {avg_mae:.2f}")
