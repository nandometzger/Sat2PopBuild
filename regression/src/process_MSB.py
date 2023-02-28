

import glob
import numpy as np
from tqdm import tqdm
from os import listdir, sep, makedirs
from os.path import isfile, join, exists 
import pandas as pd

import rasterio 
import geopandas as gpd
from rasterio.warp import transform_geom
from rasterio.features import rasterize

import ee
try:
    ee.Initialize()
except:
    print("couldn't init EE")
    ee.Authenticate(auth_mode="localhost")
    ee.Initialize()

from rasterio.crs import CRS
EE_CRS = CRS.from_epsg(4326)


MSB_no_data_countries = ["Germany", "United Kingdom", "Netherlands", "France", "Switzerland", "Ireland"]
medium = ["Italy", "Spain"] 
MSmanually_blocked_cities = ["zaragoza", "bologna", "murcia", "alicante", "palma", "valencia",
                             "sevilla", "cordoba", "madrid", "catania", "malaga", "florence", "barcelona"]
MSmanually_checked_cities = {"riga": "Latvia", 
                             "bari": "Italy", "palermo": "Italy", "rome": "Italy", "milan": "Italy", "naples": "Italy", "turin": "Italy",
                             "bilbao": "Spain", "valladolid": "Spain","lisbon": "Portugal"}

rename_countries = {"Czechia": "Czech_Republic"}

# MSdone_cities = ["zagreb", "budapest", "riga", "bari"]
MSdone_cities = ["zagreb", "budapest", "riga", "bari", "plovdiv", "bucharest", "gdansk",
                #  ""
                 ]
S1done_cities = ["leipzig","zagreb", "budapest", "riga", "edinburgh", "london", "bucharest"
                 
                "dresden", "london", "bari", "plovdiv", "gdansk", "zaragoza", "glasgow", "marseille", "stuttgart", "glasgow", "zurich", "marseille", "muenster",
                
                 ]

def write_raster_like(output, output_path, guide_tiff):
    with rasterio.Env():
        # read profile info from first file 
        with rasterio.open(guide_tiff) as src:
            meta = src.meta.copy()
        # guide_tiff["reader"].close()

        meta.update({"driver": "GTiff", "count": output.shape[0]})

        with rasterio.open(output_path, 'w', **meta) as dst:
            for ch in range(output.shape[0]):
                # iterate over channels and write bands
                img_channel = output[ch, :, :]
                dst.write(img_channel,ch + 1)  # rasterio bands are 1-indexed 

    return None
    


def rounder(values):
    def f(x):
        idx = np.argmin(np.abs(values - x))
        return values[idx]
    return np.frompyfunc(f, 1, 1)

def get_country_name(file_name): 

    with rasterio.open(file_name) as src:
        # band1 = src.read(1)
        height = src.height
        width = src.width
        this_crs = src.crs

    # get the coordinates in their local coodinate system
    # height = band1.shape[0]
    # width = band1.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src.transform, rows, cols)
    lons = np.array(xs)
    lats = np.array(ys)
    x_min, y_min, x_max, y_max = np.min(lons), np.min(lats), np.max(lons), np.max(lats)
    inpoly = { "type": "Polygon",  "coordinates": [[[x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min], [x_max, y_min]]]  }

    outpoly = transform_geom( this_crs,  EE_CRS, inpoly)
    polyshape = ee.Geometry.Polygon(outpoly["coordinates"]) 
    # Load the countries feature collection
    countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
    country = countries.filterBounds(polyshape).first().get('country_na')
    try:
        country_name = country.getInfo()
    except:
        print("couldn't find country")
        country_name = None

    if country_name in rename_countries.keys():
        country_name = rename_countries[country_name]

    return country_name, polyshape


def process_data(target_data_dir,MS_downlaod_dir, force_reprocess=False):
    city_folders = glob.glob(join(target_data_dir, "*"))
    f_names_all = np.array([])
    labs_all = np.array([])
    task_id = 0
    tasks = []
    for each_city in tqdm(city_folders):
        if each_city.endswith(".txt"):
            continue
        data_path = join(each_city, "sen2spring")
        data_pathMSB = join(each_city, "MSB")
        data_pathMSBc = join(each_city, "MSBc")
        data_pathMSBs = join(each_city, "MSBs")
        data_pathS1 = join(each_city, "S1")
        csv_path = join(each_city, each_city.split(sep)[-1:][0] + '.csv')
        city_df = pd.read_csv(csv_path)
        ids = city_df['GRD_ID']
        pop = city_df['POP']
        classes = city_df['Class']
        classes_str = [str(x) for x in classes]
        classes_paths = [data_path + '/Class_' + x + '/' for x in classes_str]
        classes_pathsMSB = [data_pathMSB + '/Class_' + x + '/' for x in classes_str]
        classes_pathsMSBc = [data_pathMSBc + '/Class_' + x + '/' for x in classes_str]
        classes_pathsMSBs = [data_pathMSBs + '/Class_' + x + '/' for x in classes_str]
        classes_pathsS1 = [data_pathS1 + '/Class_' + x + '/' for x in classes_str]
        city = each_city.split("_")[-1] 
        country_name,_ = get_country_name(classes_paths[0] + str(ids[0]) + '_sen2spring.tif')

        # MSB Footprints
        # if country_name in MSB_no_data_countries:
        #     # print("There is no MSB data for:", country_name)
        #     continue

        if country_name is None:
            # there are some cities where countryname== is still oke, because they are by the sea
            if city in MSmanually_checked_cities.keys(): 
                country_name = MSmanually_checked_cities[city]
            # else:
            #     continue

        print(country_name, ",", city)

        for index in tqdm(range(0, len(classes_paths))):
            # if classes_paths[index][-2]=='0':
            #     continue
            file_name = classes_paths[index] + str(ids[index]) + '_sen2spring.tif'
            name_MSB = str(ids[index]) + '_MSB'
            name_S1 = str(ids[index]) + '_S1'
            descriptionMSB = name_MSB + "_" + city
            descriptionS1 = name_S1 + "_" + city
            file_name_MSB = classes_pathsMSB[index] + str(ids[index]) + '_MSB.tif'
            file_name_MSBc = classes_pathsMSBc[index] + str(ids[index]) + '_MSBc.tif'
            file_name_MSBs = classes_pathsMSBs[index] + str(ids[index]) + '_MSBs.tif'
            file_name_S1 = classes_pathsS1[index] + str(ids[index]) + '_S1.tif'
            makedirs(classes_pathsMSBc[index], exist_ok=True)
            makedirs(classes_pathsMSBs[index], exist_ok=True)
            makedirs(classes_pathsS1[index], exist_ok=True)

            folderMSB = "So2Satdata/" + classes_pathsMSB[index]
            folderMSB = folderMSB.replace("/", "／")
            geojsonfile = join(join(MS_downlaod_dir,folderMSB), descriptionMSB+".geojson")

            folderS1 = "So2Satdata/" + classes_pathsS1[index]
            folderS1 = folderS1.replace("/", "／")
            S1file = join(join(MS_downlaod_dir,folderS1), descriptionS1 + ".tif")


            #check if the MSB Footprints
            process_MSB = True
            if country_name in MSB_no_data_countries:
                process_MSB = False

            if city in MSdone_cities:
                process_MSB = False

            if country_name is None:
                # there are some cities where countryname== is still oke, because they are by the sea
                if city in MSmanually_checked_cities.keys():
                    pass
                    country_name = MSmanually_checked_cities[city]
                else:
                    process_MSB = False
            
            if city in MSmanually_blocked_cities:
                process_MSB = False

            if isfile(file_name_MSBs) and (not force_reprocess):
                process_MSB = False


            # Start processing the MSBuildings
            if process_MSB:

                with rasterio.open(file_name) as src: 
                    height = src.height
                    width = src.width
                    this_crs = src.crs
                    this_meta = src.meta.copy()

                # get the coordinates in their local coodinate system
                cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                xs, ys = rasterio.transform.xy(src.transform, rows, cols)
                lons = np.array(xs)
                lats = np.array(ys) 
                # Transform the Buildings into the this_crs (3035) coordinate system, much better
                x84_min = np.min(lons)
                x84_max =  np.max(lons)
                y84_min = np.min(lats)
                y84_max = np.max(lats)

                long_pixel = (x84_max-x84_min)/width
                lat_pixel = (y84_max-y84_min)/height

                long_start = x84_min
                lat_start = y84_max
            
                long_end = long_start + long_pixel * width
                lat_end = lat_start - lat_pixel * height # negative because the latitude dimension is flipped

                long_range = np.linspace(long_start, long_end, width) #+ long_pixel/2
                lat_range = np.linspace(lat_start, lat_end, height) #- lat_pixel/2 # negative because the latitude dimension is flipped
                
                # Count Buildings
                with open(geojsonfile) as f:
                    first_line = f.readline()
                if first_line==']}':
                    #no buildings for that patch
                    output = np.zeros((1,height, width)).astype(np.float32)
                    no_shapes = True
                else:
                    no_shapes = False
                    buildingsdf = gpd.read_file(geojsonfile)

                    # Calculate central pixel and build geometry element
                    buildingsdf["centroids"] = buildingsdf["geometry"].apply(lambda x: x.centroid) 
                    buildingsdf["c_x"] = buildingsdf["centroids"].apply(lambda x: x.x)
                    buildingsdf["c_y"] = buildingsdf["centroids"].apply(lambda x: x.y)
                    buildingsdf["coords_in"] = buildingsdf.apply(lambda row: {"type": "Point",  "coordinates": [row["c_x"], row["c_y"]]} , axis=1)

                    #  e4326 to e3035
                    buildingsdf["coords_proj"] = buildingsdf["coords_in"].apply(lambda x: np.array(transform_geom(EE_CRS, this_crs, x)["coordinates"]) )
                    buildingsdf["long"] = buildingsdf["coords_proj"].apply(lambda x: x[0])
                    buildingsdf["lat"] = buildingsdf["coords_proj"].apply(lambda x: x[1])

                    # Filter Buildings that are out of bounds
                    this_buildings = buildingsdf[np.logical_and( 
                                                    np.logical_and(buildingsdf["long"]>long_start,  buildingsdf["long"]<long_end) ,
                                                    np.logical_and(buildingsdf["lat"]<lat_start, buildingsdf["lat"]>lat_end))].copy()
                    
                    # Round to the pixel values
                    this_buildings["long_range_round"] = rounder(long_range)(buildingsdf["long"]) 
                    this_buildings["lat_range_round"] = rounder(lat_range)(buildingsdf["lat"]) 

                    # get the indices of the buildings in the pixel coorinate space of the target system
                    with rasterio.open(file_name) as src: 
                        this_buildings["index_coords"] =  this_buildings.apply(lambda row: src.index(row["long_range_round"],row["lat_range_round"]), axis=1)
                        index_coords = src.index(this_buildings["long_range_round"],this_buildings["lat_range_round"])
                    
                    # count the buildings
                    this_buildings["x"] =  index_coords[0]
                    this_buildings["y"] =  index_coords[1]
                    this_buildings["x"] =  this_buildings["index_coords"].apply(lambda x: x[0])
                    this_buildings["y"] =  this_buildings["index_coords"].apply(lambda x: x[1])
                    count_df = this_buildings.groupby(["x","y"]).size().reset_index(name='building_counts')

                    output = np.zeros((1,height, width)).astype(np.float32)
                    output[0,count_df["x"],count_df["y"]] = count_df["building_counts"]

                    # to build the segmentation masks, we do:
                    buildingsdf["valid"] = buildingsdf["geometry"].type=="Polygon"
                    buildingsdf_valid = buildingsdf[buildingsdf["valid"]]
                    shapes = buildingsdf_valid["geometry"].apply(lambda inpoly: transform_geom(EE_CRS, this_crs, inpoly))

                # write the Building counts
                write_raster_like(output, output_path=file_name_MSBc, guide_tiff=file_name)
                
                # write the segmentation masks
                this_meta.update({"count": 1, "dtype": "float32"})
                with rasterio.open(file_name_MSBs, 'w+', **this_meta) as out:
                    if no_shapes:
                        out.write_band(1, output[0])
                    else:
                        out_arr = out.read(1)
                        burned = rasterize(shapes=shapes, fill=1, out=out_arr, transform=out.transform)
                        out.write_band(1, burned)


            # Also put the Sentinel-1 tile with the correct projection in the correct folder
            # Check if downloading of S1 is needed
            process_S1 = True 
            if (city in S1done_cities) and (not force_reprocess):
                process_S1 = False

            if isfile(file_name_S1):
                process_S1 = False


            # Start processing the Sentinel-1
            if process_S1:
                with rasterio.open(S1file) as src: 
                    s1raster = src.read((1,2))
                write_raster_like(s1raster, output_path=file_name_S1, guide_tiff=file_name)


            # /scratch2/metzgern/HAC/code/So2SatPOP/data/GEEexport/So2Satdata／／scratch2／metzgern／HAC／code／So2SatPOP／data／So2Sat_POP_Part1／train／00745_20865_zagreb／MSB／Class_0／

    return None


if __name__=="__main__":

    target_data_dir = '/scratch2/metzgern/HAC/code/So2SatPOP/data/So2Sat_POP_Part1/train'
    MS_downlaod_dir = '/scratch2/metzgern/HAC/code/So2SatPOP/data/GEEexport/'

    process_data(target_data_dir, MS_downlaod_dir)

    print("Done")