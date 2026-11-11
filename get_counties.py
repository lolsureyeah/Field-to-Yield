import geopandas as gpd

url = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_500k.zip"
gdf = gpd.read_file(url)
ia = gdf[gdf["STATEFP"] == "19"][["GEOID", "NAME", "geometry"]]
ia.to_file("counties_iowa.geojson", driver="GeoJSON")
print(len(ia), "counties saved")
print(ia.crs)