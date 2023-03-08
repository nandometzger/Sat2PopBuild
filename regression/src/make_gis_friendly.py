"""
This snippet demonstrates how to access and convert the buildings
data from .csv.gz to geojson for use in common GIS tools. You will
need to install pandas, geopandas, and shapely.
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from tqdm import tqdm
import os

def main():
    # this is the name of the geography you want to retrieve. update to meet your needs
    location = 'France'
    outfolder = os.path.join("/scratch2/metzgern/HAC/code/So2SatPOP/data/MSBuildingsRaw/GISfriendly", location)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    dataset_links = pd.read_csv("https://minedbuildings.blob.core.windows.net/global-buildings/dataset-links.csv")
    roi_links = dataset_links[dataset_links.Location == location]
    # for _, row in tqdm(roi_links.iterrows()):
    for _, row in tqdm(roi_links[150:].iterrows()):
        df = pd.read_json(row.Url, lines=True)
        df['geometry'] = df['geometry'].apply(shape)
        gdf = gpd.GeoDataFrame(df, crs=4326)
        outfile = os.path.join(outfolder, f"{row.QuadKey}.geojson")
        gdf.to_file(outfile, driver="GeoJSON")

if __name__ == "__main__":
    main()