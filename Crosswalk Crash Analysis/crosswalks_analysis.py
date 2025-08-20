import geopandas as gpd
import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import Point
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from shapely.geometry import Point
import folium
from folium.plugins import HeatMap

# -------------------------------
# load, clean, prepare data; creating data of crash distance to crosswalks

#load datasets
cwgdf = gpd.read_file("Crosswalks.geojson")
crashgdf = gpd.read_file("crashes_params.geojson")

# Define the CRS
la_crs_feet = "EPSG:2229"

# Data format in LA CRS (feet)
crashgdf_proj = crashgdf.to_crs(la_crs_feet)
cwgdf_proj = cwgdf.to_crs(la_crs_feet)
crashgdf = crashgdf.to_crs(epsg=4326)  

# Get center of the map
center = [crashgdf.geometry.y.mean(), crashgdf.geometry.x.mean()]

#check geometry types
print(crashgdf.geometry.iloc[0])
print(type(crashgdf.geometry.iloc[0]))

print(cwgdf.geometry.iloc[0])    
print(type(cwgdf.geometry.iloc[0]))

# Finding nearest crosswalk for each crash

crash_missing = crashgdf[crashgdf.geometry.isna() | crashgdf.geometry.is_empty].copy()
crashgdf = crashgdf[~(crashgdf.geometry.isna() | crashgdf.geometry.is_empty)].copy()

crosswalk_missing = cwgdf[cwgdf.geometry.isna() | cwgdf.geometry.is_empty].copy()
cwgdf = cwgdf[~(cwgdf.geometry.isna() | cwgdf.geometry.is_empty)].copy()

print(f"Crashes with missing geometries: {len(crash_missing)}")
print(f"Crosswalks with missing geometries: {len(crosswalk_missing)}")

crash_missing.to_file("crashes_missing_geom.geojson", driver='GeoJSON')
crosswalk_missing.to_file("crosswalks_missing_geom.geojson", driver='GeoJSON')

print('crashes with misssing geometries found')

## CRS matching
crashgdf_proj = crashgdf.to_crs("EPSG:2229")
cwgdf_proj = cwgdf.to_crs("EPSG:2229")

# algorithm to find nearest crosswalk for each crash

# get coordinates
crash_coords = np.array([[pt.x, pt.y] for pt in crashgdf_proj.geometry])
crosswalk_coords = np.array([[pt.x, pt.y] for pt in cwgdf_proj.geometry])

# KDTree for crosswalks
tree = KDTree(crosswalk_coords)

# KDTree for nearest crosswalk to each crash
distances, indices = tree.query(crash_coords)

# Add results to crashes gdf
crashgdf_proj['closest_crosswalk_index'] = indices
crashgdf_proj['distance_to_crosswalk_ft'] = distances  # because2229 units are feet

# combining attributes from matched crosswalk
crosswalk_reset = cwgdf_proj.reset_index().rename(columns={'index': 'closest_crosswalk_index'})
crashgdf_proj = crashgdf_proj.merge(
    crosswalk_reset,
    on='closest_crosswalk_index',
    how='left',
    suffixes=('_crash', '_crosswalk')
)

# Checking output
# print(crashgdf_proj[['geometry_crash', 'geometry_crosswalk', 'distance_to_crosswalk_ft']].head())


# ---------------------------------------------
# Exploring and visualizing single variable and new crosswalk distance results (maps, histograms, etc.)

print(crashgdf_proj.columns)
print(crashgdf_proj.head())
# Filter for pedestrian-vehicle collisions and print value counts
ped_crashes = crashgdf_proj[crashgdf_proj['TYPE_OF_COLLISION'] == 'Vehicle/Pedestrian']['TYPE_OF_COLLISION']
print(ped_crashes.value_counts())

# Filter df for pedestrian-vehicle collisions
ped_crashes_df = crashgdf_proj[crashgdf_proj['TYPE_OF_COLLISION'] == 'Vehicle/Pedestrian'].copy()

print(ped_crashes_df.head())

# Save ped_crashes_df to a CSV file
output_csv_path = "pedestrian_crashes_data.csv"
ped_crashes_df.to_csv(output_csv_path, index=False) # index=False prevents pandas from writing the DataFrame index as a column

print(f"Data saved to {output_csv_path}")

# Create a new df for single variable visualization of crash distance from crosswalks 
df = pd.DataFrame({
    'Distance_From_Crosswalk': ped_crashes_df['distance_to_crosswalk_ft'],
})
'''
sns.histplot(
    data=ped_crashes_df,
    x='distance_to_crosswalk_ft',
    binwidth=50,        
    color='skyblue',
    edgecolor='black'
)

# histogram
plt.title("Histogram of Pedestrian Crashes by Distance from Crosswalk")
plt.xlabel("Distance from Nearest Crosswalk (ft)")
plt.ylabel("Number of Crashes")  
plt.xlim(0, 1000)  # zoom in 
plt.tight_layout()
plt.show()

ped_crashes_df['distance_bin'] = pd.cut(
    ped_crashes_df['distance_to_crosswalk_ft'],
    bins=[0, 50, 100, 200, 400, 600, 1000, np.inf],
    labels=["0–50", "51–100", "101–200", "201–400", "401–600", "601–1000", "1000+"]
)

ped_crashes_df = ped_crashes_df.set_geometry("geometry_crash")

# MAP 1: Heatmap of total pedestrian crashes 
# Make sure geometry is  compatible with folium
ped_crashes_latlon = ped_crashes_df.to_crs(epsg=4326)

# Extract [lat, lon] coordinates
heat_data = [
    [point.y, point.x] for point in ped_crashes_latlon.geometry
    if point.is_valid and not point.is_empty
]

# folium map centering
center = [ped_crashes_latlon.geometry.y.mean(), ped_crashes_latlon.geometry.x.mean()]
m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")

# Add heatmap layer
HeatMap(heat_data, radius=10, blur=15, max_zoom=1).add_to(m)

# save and view
m.save("pedestrian_crash_heatmap.html")
print("Heatmap saved as pedestrian_crash_heatmap.html!")

#MAP 2: heatmap for crashes far from crosswalks (150 ft+)

# 1. Filter crashes >150 ft from a crosswalk
far_from_crosswalk = ped_crashes_latlon[ped_crashes_latlon['distance_to_crosswalk_ft'] > 150].copy()

# 2. map centering
m = folium.Map(location=[34.05, -118.25], zoom_start=11, tiles='cartodbpositron')

# 3. Prepare data for HeatMap
heat_data = [[point.y, point.x] for point in far_from_crosswalk.geometry]

# 4. Add heat layer to map
HeatMap(heat_data, radius=10, blur=15, max_zoom=13).add_to(m)

# 5. Display the map 
m.save("far_from_crosswalk_heatmap.html")
print("Heatmap saved as far_from_crosswalk_heatmap.html!")

# MAP 3 Multiple crash hotspot 150 ft+ from crosswalks
from sklearn.cluster import DBSCAN
import numpy as np

#  lat/lon as numpy array
coords = np.array([[point.y, point.x] for point in far_from_crosswalk.geometry])

# Run DBSCAN to group points within ~100m (0.001 degrees ~ 100m)
kms_per_radian = 6371.0088
epsilon = 0.1 / kms_per_radian  # 100 meters
db = DBSCAN(eps=epsilon, min_samples=2, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))

# 3. Add cluster labels to DataFrame
far_from_crosswalk['cluster'] = db.labels_

# 4. Filter for only clusters with more than 1 crash
valid_clusters = far_from_crosswalk['cluster'].value_counts()
multi_crash_clusters = valid_clusters[valid_clusters > 1].index
filtered = far_from_crosswalk[far_from_crosswalk['cluster'].isin(multi_crash_clusters)]

# 5. map those
heat_data_filtered = [[point.y, point.x] for point in filtered.geometry]
# Create folium map centered on average location of filtered crashes        
# 6. Add HeatMap layer using filtered data
HeatMap(
    heat_data_filtered,
    radius=10,
    blur=15,
    max_zoom=13
).add_to(m)

# 7. Save map
m.save("far_from_crosswalk_multiple_heatmap.html")
print("Heatmap saved as far_from_crosswalk_multiple_heatmap.html!") 
'''
# -------------------------------------
# further analysis: regressions, at fault, severity, time of day, etc.

print(ped_crashes_df['at_fault_type'].unique())
driver_at_fault = ped_crashes_df[ped_crashes_df['at_fault_type'] == 'Driver'].copy()
pedestrian_at_fault = ped_crashes_df[ped_crashes_df['at_fault_type'] == 'Pedestrian'].copy()

# at fault type and distance to crosswalk regression analysis
#identify variables of interest
'''base_vars = [
    'driver_fault',
    'distance_to_crosswalk_ft',
    'LIGHTING',
    'speed_bucket'
    'n_lanes',
    'ControlTypeDesc'
]'''
#regression prep
##create a dummy variable for at fault type : 1 if driver at fault, 0 otherwise
ped_crashes_df['driver_fault'] = (ped_crashes_df['at_fault_type'] == 'Driver').astype(int)

# control variables and maniupulation for regression
control_dummies = pd.get_dummies(ped_crashes_df['ControlTypeDesc'], drop_first=True)
lighting_dummies = pd.get_dummies(ped_crashes_df['LIGHTING'], drop_first=True)
speed_map = {'<30': 1, '30-35': 2, '>35': 3}
ped_crashes_df['speed_bucket_num'] = ped_crashes_df['speed_bucket'].map(speed_map)

# Create regression DataFrame
at_fault_regression_df = ped_crashes_df[['distance_to_crosswalk_ft', 'driver_fault','n_lanes','speed_bucket_num']].copy()
at_fault_regression_df = pd.concat([
    at_fault_regression_df,
    control_dummies,
    lighting_dummies], axis=1)
bool_cols = at_fault_regression_df.select_dtypes(include='bool').columns
at_fault_regression_df[bool_cols] = at_fault_regression_df[bool_cols].astype(int)
print(at_fault_regression_df.iloc[0])
print(at_fault_regression_df.dtypes)
# Check for missing values in the regression dataframe
missing_mask = at_fault_regression_df.isnull()

if missing_mask.values.any():
    print("Missing values detected!\n")
    
    # Get row and column indices of missing values
    rows_with_missing = missing_mask.any(axis=1)
    missing_rows = at_fault_regression_df[rows_with_missing]
    
    print("🔎 Rows with missing values:")
    print(missing_rows)
else:
    print("✅ No missing values detected in the regression DataFrame.")

# 1. driverfault_model_logit_continuous 
# Split into y and X
y = at_fault_regression_df['driver_fault']
X = at_fault_regression_df.drop(columns='driver_fault')

# Add intercept
X = sm.add_constant(X)

# Fit logit model
driverfault_model_logit_continuous  = sm.Logit(y, X)
result = driverfault_model_logit_continuous.fit()

# Print summary
print(result.summary())

# 2. pedfault_model_logit_continuous 
ped_crashes_df['ped_fault'] = (ped_crashes_df['at_fault_type'] == 'Pedestrian').astype(int)
ped_fault_regression_df_2 = ped_crashes_df[[
    'distance_to_crosswalk_ft',
    'ped_fault',
    'n_lanes',
    'speed_bucket_num'
]].copy()

ped_fault_regression_df_2 = pd.concat([
    ped_fault_regression_df_2,
    control_dummies,
    lighting_dummies
], axis=1)

# Convert bools to int
bool_cols = ped_fault_regression_df_2.select_dtypes(include='bool').columns
ped_fault_regression_df_2[bool_cols] = ped_fault_regression_df_2[bool_cols].astype(int)
X = ped_fault_regression_df_2.drop(columns='ped_fault')
y = ped_fault_regression_df_2['ped_fault']

X = sm.add_constant(X)
pedfault_model_logit_continuous = sm.Logit(y, X).fit()
print(pedfault_model_logit_continuous.summary())

# 3. binned logit regression pedfault_model_logit_binned
bins = [0, 50, 100, 150, 250, 400, 600, np.inf]
labels = [
    "0-49 ft", "50-99 ft", "100-149 ft", "150-249 ft",
    "250-399 ft", "400-599 ft", "600+ ft"
]
ped_crashes_df['dist_bin'] = pd.cut(ped_crashes_df['distance_to_crosswalk_ft'], bins=bins, labels=labels)
distance_bin_dummmies = pd.get_dummies(ped_crashes_df['dist_bin'], drop_first=True)  # drops '0–49' as reference
ped_crashes_df['ped_fault'] = (ped_crashes_df['at_fault_type'] == 'Pedestrian').astype(int)
#numerical variables
ped_fault_regression_df_3 = ped_crashes_df[[
    'ped_fault',
    'n_lanes',
    'speed_bucket_num'
]].copy()
#adding categorical variables
ped_fault_regression_df_3 = pd.concat([
    ped_fault_regression_df_3,
    distance_bin_dummmies,
    control_dummies,
    lighting_dummies
], axis=1)

# Convert bools to int
bool_cols = ped_fault_regression_df_3.select_dtypes(include='bool').columns
ped_fault_regression_df_3[bool_cols] = ped_fault_regression_df_3[bool_cols].astype(int)
X = ped_fault_regression_df_3.drop(columns='ped_fault')
y = ped_fault_regression_df_3['ped_fault']

X = sm.add_constant(X)
pedfault_model_logit_binned = sm.Logit(y, X).fit()
print(pedfault_model_logit_binned.summary())

# 3.1. forest plot to visualize odds ratios
# get regression output and create dataframe for CI and coefficients
model_3_reg_output_df = pedfault_model_logit_binned.summary2().tables[1].copy()

# check to see if correct dataframe was created
print(model_3_reg_output_df.head())

# Create a df for the odds ratios and confidence intervals
print(model_3_reg_output_df.iloc[3:9,[0,4,5]])
print(np.exp(model_3_reg_output_df.iloc[3:9,[0,4,5]]))

forest_df = np.exp(model_3_reg_output_df.iloc[3:9,[0,4,5]])
forest_df.columns = ['Odds Ratio', 'Lower CI', 'Upper CI']

# reverse  rows 
forest_df = forest_df.iloc[::-1]

# https://matplotlib.org/stable/api/axes_api.html
# https://www.henrylau.co.uk/2020/05/10/visualising-odds-ratios/
# set up
fig, ax = plt.subplots(figsize=(8, 5))

# error bars
ax.errorbar(
    x=forest_df['Odds Ratio'],
    y=forest_df.index,
    xerr=[forest_df['Odds Ratio'] - forest_df['Lower CI'], forest_df['Upper CI'] - forest_df['Odds Ratio']],
    fmt='o',
    color='red',
    ecolor='gray',
    capsize=4,
    label='95% Confidence Interval')

# plot formatting
ax.axvline(x=1, color='black', linestyle='-', linewidth=2)
ax.set_xticks([1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
ax.set_xticklabels(['Just as likely', '1.5x as likely', '2x as likely', '2.5x as likely', 
                    '3x as likely', '3.5x as likely', '4x as likely', '4.5x as likely', '5x as likely'])
ax.set_xlabel("Likelihood of pedestrian being assigned at fault compared to 0-49 ft away from crosswalk")
ax.set_ylabel("") 
ax.set_title('Controlled for speed limit, lighting, number of lanes, and traffic control', fontsize=9) 
fig.suptitle('Odds of Pedestrian Being Assigned at Fault for a Pedestrian-Vehicle Crash by Distance from Crosswalk')
plt.show()

# 3.2. predicted probabilities bins plot (logit) 

## 0-49ft group exploration 

#need to correct datasets from crashgdf_proj to ped_crashes_df for the following analyses:
'''
# 4. binned probit regression pedfault_model_probit_binned
bins = [0, 50, 100, 150, 250, 400, 600, np.inf]
labels = [
    "0-49 ft", "50-99 ft", "100-149 ft", "150-249 ft",
    "250-399 ft", "400-599 ft", "600+ ft"
]
crashgdf_proj['dist_bin'] = pd.cut(crashgdf_proj['distance_to_crosswalk_ft'], bins=bins, labels=labels)
distance_bin_dummmies = pd.get_dummies(crashgdf_proj['dist_bin'], drop_first=True)  # drops '0–49' for multicollinearity
crashgdf_proj['ped_fault'] = (crashgdf_proj['at_fault_type'] == 'Pedestrian').astype(int)
#numerical variables
ped_fault_regression_df_4 = crashgdf_proj[[
    'ped_fault',
    'n_lanes',
    'speed_bucket_num'
]].copy()
#adding categorical variables
ped_fault_regression_df = pd.concat([
    ped_fault_regression_df_4,
    distance_bin_dummmies,
    control_dummies,
    lighting_dummies
], axis=1)

# Convert bools to int
bool_cols = ped_fault_regression_df_4.select_dtypes(include='bool').columns
ped_fault_regression_df_4[bool_cols] = ped_fault_regression_df_4[bool_cols].astype(int)
X = ped_fault_regression_df_4.drop(columns='ped_fault')
y = ped_fault_regression_df_4['ped_fault']

X = sm.add_constant(X)
pedfault_model_probit_binned = sm.Probit(y, X).fit()
print(pedfault_model_probit_binned.summary())

# 5. pedestrian at fault probit regression continuous pedfault_model_probit_continuou
crashgdf_proj['ped_fault'] = (crashgdf_proj['at_fault_type'] == 'Pedestrian').astype(int)
ped_fault_regression_df_5 = crashgdf_proj[[
    'distance_to_crosswalk_ft',
    'ped_fault',
    'n_lanes',
    'speed_bucket_num'
]].copy()

ped_fault_regression_df_5 = pd.concat([
    ped_fault_regression_df_5,
    control_dummies,
    lighting_dummies
], axis=1)

# Convert bools to int
bool_cols = ped_fault_regression_df_5.select_dtypes(include='bool').columns
ped_fault_regression_df_5[bool_cols] = ped_fault_regression_df_5[bool_cols].astype(int)
X = ped_fault_regression_df_5.drop(columns='ped_fault')
y = ped_fault_regression_df_5['ped_fault']

X = sm.add_constant(X)
pedfault_model_probit_continuous = sm.Probit(y, X).fit()
print(pedfault_model_probit_continuous.summary())

#exploration of further variables for analysis
# ped_action
# crosswalk design
print(crashgdf_proj['CrosswalkColorCodeDesc'].unique())
print(crashgdf_proj['MarkingTypeDesc'].unique())
print(crashgdf_proj['CrossType'].unique())

# MarkingTypeDesc dummy variables
marking_dummies = pd.get_dummies(crashgdf_proj['MarkingTypeDesc'], drop_first=True)

# regression of analysis of crash distance to crosswalk and at fault type
print(crashgdf_proj['PED_ACTION'].unique())
'''