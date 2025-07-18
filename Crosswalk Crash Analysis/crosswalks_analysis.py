import geopandas as gpd
import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import Point
import seaborn as sns
import matplotlib.pyplot as plt

cwgdf = gpd.read_file("Crosswalks.geojson")
'''
print("✅ Loaded GeoDataFrame:")
print(cwgdf.info())
print("\n🧠 First rows:\n", cwgdf.head())
print("\n🌍 Geometry types:\n", cwgdf.geom_type.value_counts())
print("\n📐 CRS:\n", cwgdf.crs)'''

crashgdf = gpd.read_file("crashes_params.geojson")
'''
print("✅ Loaded GeoDataFrame:")
print(crashgdf.info())
print("\n🧠 First rows:\n", crashgdf.head())
print("\n🌍 Geometry types:\n", crashgdf.geom_type.value_counts())
print("\n📐 CRS:\n", crashgdf.crs)'''

# Define the CRS
la_crs_feet = "EPSG:2229"

# Project to LA State Plane in feet
crashgdf_proj = crashgdf.to_crs(la_crs_feet)
cwgdf_proj = cwgdf.to_crs(la_crs_feet)

import geopandas as gpd
from shapely.strtree import STRtree
from shapely.geometry import Point
import folium

# Check if geometry is valid and in the right format
crashgdf = crashgdf[crashgdf.geometry.notnull()]  # drop rows with missing geometry
crashgdf = crashgdf.to_crs(epsg=4326)  # ensure it's in WGS84 (lat/lon)

# Get center of the map
center = [crashgdf.geometry.y.mean(), crashgdf.geometry.x.mean()]
'''
# Create map
m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

# Add crash points
for _, row in crashgdf.iterrows():
    if row.geometry.geom_type == "Point":
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=2,
            color='red',
            fill=True,
            fill_opacity=0.7,
            popup=str(row.get("CASE_ID", ""))  # show ID or other field
        ).add_to(m)

# Save to HTML
m.save("crash_map.html")
print("🌐 Map saved as crash_map.html — open it in your browser!")


# Check if geometry is valid and in the right format
cwgdf = cwgdf[cwgdf.geometry.notnull()]  # drop rows with missing geometry
cwgdf = cwgdf.to_crs(epsg=4326)  # ensure it's in WGS84 (lat/lon)

# Get center of the map
center = [cwgdf.geometry.y.mean(), cwgdf.geometry.x.mean()]

# Create map
m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

# Add crosswalk points
for _, row in cwgdf.iterrows():
    if row.geometry.geom_type == "Point":
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=2,
            color='blue',  # use blue for contrast with crash points
            fill=True,
            fill_opacity=0.7,
            popup=str(row.get("ID", ""))  # replace with any relevant attribute
        ).add_to(m)

# Save to HTML
m.save("crosswalk_map.html")
print("🌐 Map saved as crosswalk_map.html — open it in your browser!") 

from shapely.ops import nearest_points

# Make sure both are in the same projected CRS (already EPSG:2229)
# Build a spatial index for efficient querying
crosswalk_geoms = cwgdf_proj.geometry.tolist()
crosswalk_tree = STRtree(crosswalk_geoms)

# Find nearest crosswalk for each crash
nearest_crosswalk_geom = []
distance_to_crosswalk = []


print(crashgdf.columns)
print(cwgdf.columns)

'''
#check geometry types
print(crashgdf.geometry.iloc[0])    
print(type(crashgdf.geometry.iloc[0]))

print(cwgdf.geometry.iloc[0])    
print(type(cwgdf.geometry.iloc[0]))

# Find nearest crosswalk for each crash

crash_missing = crashgdf[crashgdf.geometry.isna() | crashgdf.geometry.is_empty].copy()
crashgdf = crashgdf[~(crashgdf.geometry.isna() | crashgdf.geometry.is_empty)].copy()

crosswalk_missing = cwgdf[cwgdf.geometry.isna() | cwgdf.geometry.is_empty].copy()
cwgdf = cwgdf[~(cwgdf.geometry.isna() | cwgdf.geometry.is_empty)].copy()

print(f"🚫 Crashes with missing geometries: {len(crash_missing)}")
print(f"🚫 Crosswalks with missing geometries: {len(crosswalk_missing)}")

crash_missing.to_file("crashes_missing_geom.geojson", driver='GeoJSON')
crosswalk_missing.to_file("crosswalks_missing_geom.geojson", driver='GeoJSON')

print('crashes with misssing geometries found')

## CRS matching
crashgdf_proj = crashgdf.to_crs("EPSG:2229")
cwgdf_proj = cwgdf.to_crs("EPSG:2229")

# Build a spatial index for efficient querying


# Assuming you already have:
# crashgdf_proj  = projected crashes GeoDataFrame (EPSG:2229)
# cwgdf_proj     = projected crosswalks GeoDataFrame (EPSG:2229)

# Step 1: Extract coordinates
crash_coords = np.array([[pt.x, pt.y] for pt in crashgdf_proj.geometry])
crosswalk_coords = np.array([[pt.x, pt.y] for pt in cwgdf_proj.geometry])

# Step 2: Build KDTree for crosswalks
tree = KDTree(crosswalk_coords)

# Step 3: Query KDTree for nearest crosswalk to each crash
distances, indices = tree.query(crash_coords)

# Step 4: Add results to crashes GeoDataFrame
crashgdf_proj['closest_crosswalk_index'] = indices
crashgdf_proj['distance_to_crosswalk_ft'] = distances  # because EPSG:2229 units are feet

# Step 5: Merge attributes from the matched crosswalk
crosswalk_reset = cwgdf_proj.reset_index().rename(columns={'index': 'closest_crosswalk_index'})
crashgdf_proj = crashgdf_proj.merge(
    crosswalk_reset,
    on='closest_crosswalk_index',
    how='left',
    suffixes=('_crash', '_crosswalk')
)

# Step 6: Done! Check output
# print(crashgdf_proj[['geometry_crash', 'geometry_crosswalk', 'distance_to_crosswalk_ft']].head())

# analyze results
## regresion analysis
print(crashgdf_proj.columns)
## Filter for pedestrian-vehicle collisions and print value counts
ped_crashes = crashgdf_proj[crashgdf_proj['TYPE_OF_COLLISION'] == 'Vehicle/Pedestrian']['TYPE_OF_COLLISION']

# Create a DataFrame for regression analysis

# Filter the whole DataFrame for pedestrian-vehicle collisions
ped_crashes_df = crashgdf_proj[crashgdf_proj['TYPE_OF_COLLISION'] == 'Vehicle/Pedestrian'].copy()

# Create a new DataFrame for regression (optional — or you can just use ped_crashes_df directly)
df = pd.DataFrame({
    'Distance_From_Crosswalk': ped_crashes_df['distance_to_crosswalk_ft'],
})

sns.kdeplot(data=ped_crashes_df, x='distance_to_crosswalk_ft', fill=True)
plt.title("KDE of Pedestrian Crashes by Distance from Crosswalk")
plt.xlabel("Distance from Nearest Crosswalk (ft)")
plt.ylabel("Distribution Density")
plt.xlim(0, 1000) 
plt.show()

sns.histplot(
    data=ped_crashes_df,
    x='distance_to_crosswalk_ft',
    binwidth=50,          # you can adjust bin width to taste
    color='skyblue',
    edgecolor='black'
)

# histogram
plt.title("Histogram of Pedestrian Crashes by Distance from Crosswalk")
plt.xlabel("Distance from Nearest Crosswalk (ft)")
plt.ylabel("Number of Crashes")  # ← now it shows counts
plt.xlim(0, 1000)  # zoom in on the range you care about
plt.tight_layout()
plt.show()

'''ped_crashes_df['distance_bin'] = pd.cut(
    ped_crashes_df['distance_to_crosswalk_ft'],
    bins=[0, 50, 100, 200, 400, 600, 1000, np.inf],
    labels=["0–50", "51–100", "101–200", "201–400", "401–600", "601–1000", "1000+"]
)'''

import folium
from folium.plugins import HeatMap

ped_crashes_df = ped_crashes_df.set_geometry("geometry_crash")
ped_crashes_latlon = ped_crashes_df.to_crs(epsg=4326)

# MAP 1: Heatmap of total pedestrian crashes 
# Make sure geometry is in WGS84 (lat/lon) for folium
ped_crashes_latlon = ped_crashes_df.to_crs(epsg=4326)

# Extract [lat, lon] coordinates
heat_data = [
    [point.y, point.x] for point in ped_crashes_latlon.geometry
    if point.is_valid and not point.is_empty
]

# Optionally weight heatmap by distance (inverse = hotter near crosswalks)
# heat_data = [
#     [point.y, point.x, 1 / (row['distance_to_crosswalk_ft'] + 1)] 
#     for point, (_, row) in zip(ped_crashes_latlon.geometry, ped_crashes_latlon.iterrows())
#     if point.is_valid and not point.is_empty
# ]

# Create folium map centered on average location
center = [ped_crashes_latlon.geometry.y.mean(), ped_crashes_latlon.geometry.x.mean()]
m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")

# Add heatmap layer
HeatMap(heat_data, radius=10, blur=15, max_zoom=1).add_to(m)

# Save and view
m.save("pedestrian_crash_heatmap.html")
print("🌐 Heatmap saved as pedestrian_crash_heatmap.html — open it in your browser!")

#MAP 2heatmap for crashes far from crosswalks

# 1. Filter crashes >150 ft from a crosswalk
far_from_crosswalk = ped_crashes_latlon[ped_crashes_latlon['distance_to_crosswalk_ft'] > 150].copy()

# 2. Create a folium map centered on LA (you can adjust zoom or center if needed)
m = folium.Map(location=[34.05, -118.25], zoom_start=11, tiles='cartodbpositron')

# 3. Prepare data for HeatMap
heat_data = [[point.y, point.x] for point in far_from_crosswalk.geometry]

# 4. Add heat layer to map
HeatMap(heat_data, radius=10, blur=15, max_zoom=13).add_to(m)

# 5. Display the map (Jupyter/Notebook) or save to HTML
m.save("far_from_crosswalk_heatmap.html")
print("🌐 Heatmap saved as far_from_crosswalk_heatmap.html — open it in your browser!")

# MAP 3 Multiple crash hotspot > 150 ft from crosswalks! 
from sklearn.cluster import DBSCAN
import numpy as np

# 1. Extract lat/lon as numpy array
coords = np.array([[point.y, point.x] for point in far_from_crosswalk.geometry])

# 2. Run DBSCAN to group points within ~100m (0.001 degrees ~ 100m)
kms_per_radian = 6371.0088
epsilon = 0.1 / kms_per_radian  # 100 meters
db = DBSCAN(eps=epsilon, min_samples=2, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))

# 3. Add cluster labels to DataFrame
far_from_crosswalk['cluster'] = db.labels_

# 4. Filter for only clusters with more than 1 crash
valid_clusters = far_from_crosswalk['cluster'].value_counts()
multi_crash_clusters = valid_clusters[valid_clusters > 1].index
filtered = far_from_crosswalk[far_from_crosswalk['cluster'].isin(multi_crash_clusters)]

# 5. Now map those
heat_data_filtered = [[point.y, point.x] for point in filtered.geometry]
# Create folium map centered on average location of filtered crashes        
# 3. Add HeatMap layer using filtered data
HeatMap(
    heat_data_filtered,
    radius=10,
    blur=15,
    max_zoom=13
).add_to(m)

# 4. Save map
m.save("far_from_crosswalk_multiple_heatmap.html")
print("🌐 Heatmap saved as far_from_crosswalk_multiple_heatmap.html — open it in your browser!")

# further analysis: at fault, severity, time of day, etc.

# regression of analysis of crash distance to crosswalk and at fault type