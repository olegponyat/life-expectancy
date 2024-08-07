import requests

# Define the URL and parameters for the CDC WONDER query
url = "https://wonder.cdc.gov/controller/datarequest/D77"
params = {
    "B_1": "D77",  # Geographic level (e.g., state, county)
    "B_2": "*None*",         # Age groups
    "B_3": "*None*",         # Gender
    "B_4": "*None*",         # Race
    "B_5": "D77.V5-level1",  # Year
    "B_6": "*None*",         # Multiple Cause of Death
    "B_7": "*None*",         # Injury Intent and Mechanism
    "B_8": "D77.V2-level1",  # Underlying Cause of Death
    "F_D77.V5": ["1980"],    # Year of interest
    "F_D77.V2": ["I00-I99"], # ICD-10 codes for heart diseases
    "B_9": "*None*",         # Place of death
    "B_10": "*None*",        # Autopsy
}

# Make the request to CDC WONDER
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    # Process and analyze the data
    # (the exact processing steps will depend on the format of the returned data)
else:
    print("Failed to retrieve data:", response.status_code)
