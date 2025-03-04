import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def clean_county_name(full_name):
    if isinstance(full_name, str):
        return full_name.split(',')[0].strip()
    return None

cdc_wonder_map = gpd.read_file(r'C:\Users\oleg2\Downloads\Shapefiles\Counties\co1980p020.shp')

aging_df = pd.read_csv('aging.csv')

aging_df['County Name'] = aging_df['County'].apply(clean_county_name)


from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

coords_df = pd.read_csv('county.csv')
coords_df['lat'] = pd.to_numeric(coords_df['lat'], errors='coerce')
coords_df['lng'] = pd.to_numeric(coords_df['lng'], errors='coerce')
coords_df.dropna(subset=['lat', 'lng'], inplace=True)
coords_df['county_state'] = coords_df.apply(lambda row: f"{row['county_full']}, {row['state_id']}", axis=1)
coords_df.set_index('county_state', inplace=True)

aging_df['Crude Rate'] = aging_df['Crude Rate'].str.replace(r'\s*\(Unreliable\)', '', regex=True)
aging_df['Crude Rate'] = pd.to_numeric(aging_df['Crude Rate'], errors='coerce')
aging_df = aging_df[~aging_df['Crude Rate'].isna()]
overall_aging_df = aging_df[aging_df['Race'].isna() & aging_df['Gender'].isna()].copy()
overall_aging_df.set_index('County', inplace=True)
overall_aging_df['Crude Rate'] *= 0.1

def get_crude_rate(county_state):
    return overall_aging_df.loc[county_state, 'Crude Rate'] if county_state in overall_aging_df.index else np.nan

coords_df['Crude Rate'] = coords_df.index.map(get_crude_rate)
coords_df.dropna(subset=['Crude Rate'], inplace=True)

X = coords_df[['lat', 'lng']].values
nbrs = NearestNeighbors(n_neighbors=8)
nbrs.fit(X)

num_runs = 1000
intensity_levels = np.arange(0.40, 1.62, 0.02)
mae_results_per_intensity = {intensity: [] for intensity in intensity_levels}

def get_reliable_random_county():
    while True:
        random_county = coords_df.sample(n=1).iloc[0]
        county_name = random_county.name
        if county_name in overall_aging_df.index:
            return random_county

def calculate_weighted_crude_rate(demographic_data_df):
    demographic_data_df = demographic_data_df.dropna(subset=['Crude Rate', 'Population'])
    total_population = demographic_data_df['Population'].sum()
    demographic_data_df['Population Proportion'] = demographic_data_df['Population'] / total_population
    demographic_data_df['Weighted Crude Rate'] = demographic_data_df['Crude Rate'] * demographic_data_df['Population Proportion']
    adjusted_crude_rate = demographic_data_df['Weighted Crude Rate'].sum()
    return adjusted_crude_rate

def prepare_data_for_regression(intensity, sample=True):
    features = []
    targets = []

    if sample:
        for _ in range(num_runs):
            random_county = get_reliable_random_county()
            county_name = random_county.name
            lat = random_county['lat']
            lng = random_county['lng']

            actual_crude_rate = overall_aging_df.loc[county_name, 'Crude Rate']

            distances, indices = nbrs.kneighbors([[lat, lng]])
            neighbor_indices = indices[0][1:]
            neighbor_counties = coords_df.iloc[neighbor_indices]

            neighbor_crude_rates = []
            for index in neighbor_counties.index:
                if index in overall_aging_df.index:
                    neighbor_crude_rate = overall_aging_df.loc[index, 'Crude Rate']
                    neighbor_crude_rates.append(neighbor_crude_rate)
            
            avg_crude_rate_neighbors = np.mean(neighbor_crude_rates) if neighbor_crude_rates else np.nan

            weighted_crude_rates = []
            for index in neighbor_counties.index:
                if index in overall_aging_df.index:
                    neighbor_crude_rate = overall_aging_df.loc[index, 'Crude Rate']
                    simple_avg = avg_crude_rate_neighbors
                    weight = (neighbor_crude_rate - simple_avg) / simple_avg
                    weighted_crude_rate = neighbor_crude_rate + (intensity * simple_avg * weight)
                    weighted_crude_rates.append(weighted_crude_rate)
            
            complex_avg_crude_rate = np.mean(weighted_crude_rates) if weighted_crude_rates else np.nan

            demographic_data_df = aging_df[~(aging_df['Race'].isna() & aging_df['Gender'].isna()) & (aging_df['County'] == county_name)]
            adjusted_crude_rate = calculate_weighted_crude_rate(demographic_data_df) * 0.1
            
            features.append([lat, lng, avg_crude_rate_neighbors, complex_avg_crude_rate, adjusted_crude_rate])
            targets.append(actual_crude_rate)
    else:
        for county_name in coords_df.index:
            if county_name in overall_aging_df.index:
                county_data = coords_df.loc[county_name]
                lat = county_data['lat']
                lng = county_data['lng']
                actual_crude_rate = overall_aging_df.loc[county_name, 'Crude Rate']

                distances, indices = nbrs.kneighbors([[lat, lng]])
                neighbor_indices = indices[0][1:]
                neighbor_counties = coords_df.iloc[neighbor_indices]

                neighbor_crude_rates = []
                for index in neighbor_counties.index:
                    if index in overall_aging_df.index:
                        neighbor_crude_rate = overall_aging_df.loc[index, 'Crude Rate']
                        neighbor_crude_rates.append(neighbor_crude_rate)
                
                avg_crude_rate_neighbors = np.mean(neighbor_crude_rates) if neighbor_crude_rates else np.nan

                weighted_crude_rates = []
                for index in neighbor_counties.index:
                    if index in overall_aging_df.index:
                        neighbor_crude_rate = overall_aging_df.loc[index, 'Crude Rate']
                        simple_avg = avg_crude_rate_neighbors
                        weight = (neighbor_crude_rate - simple_avg) / simple_avg
                        weighted_crude_rate = neighbor_crude_rate + (best_intensity * simple_avg * weight)
                        weighted_crude_rates.append(weighted_crude_rate)
                
                complex_avg_crude_rate = np.mean(weighted_crude_rates) if weighted_crude_rates else np.nan

                demographic_data_df = aging_df[~(aging_df['Race'].isna() & aging_df['Gender'].isna()) & (aging_df['County'] == county_name)]
                adjusted_crude_rate = calculate_weighted_crude_rate(demographic_data_df) * 0.1
                
                features.append([lat, lng, avg_crude_rate_neighbors, complex_avg_crude_rate, adjusted_crude_rate])
                targets.append(actual_crude_rate)

    return np.array(features), np.array(targets)

for intensity in intensity_levels:
    print(f"\nTesting with complex average weight intensity: {intensity:.2f}")

    X_train, y_train = prepare_data_for_regression(intensity, sample=True)
    
    if len(X_train) > 0:
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_train)
        mae = mean_absolute_error(y_train, predictions)
        mae_results_per_intensity[intensity].append(mae)
    else:
        mae_results_per_intensity[intensity].append(np.nan)

average_mae_per_intensity = {intensity: np.nanmean(mae_results_per_intensity[intensity]) for intensity in intensity_levels}
best_intensity = min(average_mae_per_intensity, key=average_mae_per_intensity.get)
best_mae = average_mae_per_intensity[best_intensity]

print(f"\nBest Intensity Level: {best_intensity:.2f} with Average MAE: {best_mae:.2f}")
plt.figure(figsize=(12, 6))
plt.plot(list(average_mae_per_intensity.keys()), list(average_mae_per_intensity.values()), marker='o', linestyle='-', color='b')
plt.title('Average MAE for Each Intensity Level of Complex Average Weights')
plt.xlabel('Complex Average Weight Intensity')
plt.ylabel('Average Mean Absolute Error')
plt.grid(True)
plt.show()

print("\nAverage MAE for each intensity level:")
for intensity, avg_mae in average_mae_per_intensity.items():
    print(f"Intensity {intensity:.2f}: {avg_mae:.2f}")

print(f"\nRunning final test with the best intensity level: {best_intensity:.2f}")

final_results = []

X_final, y_final = prepare_data_for_regression(best_intensity, sample=False)

if len(X_final) > 0:
    final_model = LinearRegression()
    final_model.fit(X_final, y_final)
    final_predictions = final_model.predict(X_final)

    for i in range(len(X_final)):
        county_name = coords_df.iloc[i].name
        
        geographic_predicted_crude_rate = X_final[i][2]
        demographic_predicted_crude_rate = X_final[i][4]
        
        final_results.append({
            'County': county_name,
            'Actual Crude Rate': y_final[i],
            'Predicted Crude Rate': final_predictions[i],
            'Geographic Predicted Crude Rate': geographic_predicted_crude_rate,
            'Demographic Predicted Crude Rate': demographic_predicted_crude_rate,
            'MAE': abs(y_final[i] - final_predictions[i])
        })

    final_results_df = pd.DataFrame(final_results)

    print("\nFinal Results for Each County:")
    print(final_results_df[['County', 'Actual Crude Rate', 'Predicted Crude Rate', 
                            'Geographic Predicted Crude Rate', 'Demographic Predicted Crude Rate', 'MAE']])

    print("\nAverage Values for Final Results:")
    print(f"Average Actual Crude Rate: {final_results_df['Actual Crude Rate'].mean():.2f}")
    print(f"Average Predicted Crude Rate: {final_results_df['Predicted Crude Rate'].mean():.2f}")
    print(f"Average Geographic Predicted Crude Rate: {final_results_df['Geographic Predicted Crude Rate'].mean():.2f}")
    print(f"Average Demographic Predicted Crude Rate: {final_results_df['Demographic Predicted Crude Rate'].mean():.2f}")
    print(f"Average MAE: {final_results_df['MAE'].mean():.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(final_results_df.index + 1, final_results_df['MAE'], marker='o', linestyle='-', color='g')
    plt.title('Mean Absolute Error for Each County in Final Test')
    plt.xlabel('County Index')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.scatter(final_results_df['Actual Crude Rate'], final_results_df['Predicted Crude Rate'], alpha=0.5)
    plt.plot([min(final_results_df['Actual Crude Rate']), max(final_results_df['Actual Crude Rate'])],
             [min(final_results_df['Actual Crude Rate']), max(final_results_df['Actual Crude Rate'])],
             color='r', linestyle='--', linewidth=2)
    plt.title('Final Predictions vs Actual Crude Rates')
    plt.xlabel('Actual Crude Rate')
    plt.ylabel('Predicted Crude Rate')
    plt.grid(True)
    plt.show()

    final_results_df['Geographic MAE'] = abs(final_results_df['Actual Crude Rate'] - final_results_df['Geographic Predicted Crude Rate'])
    final_results_df['Demographic MAE'] = abs(final_results_df['Actual Crude Rate'] - final_results_df['Demographic Predicted Crude Rate'])

    plt.figure(figsize=(10, 6))
    plt.scatter(final_results_df['Geographic MAE'], final_results_df['Demographic MAE'], alpha=0.6, color='purple')
    plt.title('Demographic MAE vs. Geographic MAE')
    plt.xlabel('Geographic MAE')
    plt.ylabel('Demographic MAE')
    plt.grid(True)

    plt.show()


else:
    print("not enough data for testing")
