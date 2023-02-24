import glob
from os import listdir, sep
from os.path import isfile, join, exists 

from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from sklearn import model_selection

from tqdm import tqdm
import rasterio
from rasterio.warp import transform_geom

from rasterio.crs import CRS

from utils.utils import get_fnames_labs_reg
from utils.transform import RandomRotationTransform, RandomBrightness, RandomGamma
from utils.dataset import PopulationDataset_Reg

rot_angle = np.arctan(16/97)*(180/np.pi)
img_rows = 100  # patch height
img_cols = 100  # patch width

import ee
# ee.Authenticate()
try:
    ee.Initialize()
except:
    ee.Authenticate(auth_mode="localhost")
    ee.Initialize()

# EE_CRS = CRS.from_epsg(3857)
EE_CRS = CRS.from_epsg(4326)

MSB_no_data_countries = ["Germany", "United Kingdom", "Netherlands", "France", "Switzerland", "Ireland"]
medium = ["Italy", "Spain"]
MSmanually_blocked_cities = ["zaragoza"]
good = ["Croatia", "Slovakia", "Bulgaria", "Czechia", "Romania", "Sweden", "Greece", "Austria", "Finland", "Denmark" ]

MSmanually_checked_cities = {"riga": "Latvia", "bari": "Italy", "palermo": "Italy", "lisbon": "Portugal"}

rename_countries = {"Czechia": "Czech_Republic"}


# done_cities = ["zagreb", "budapest", "riga", "bari"]
MSdone_cities = ["zagreb", "budapest", "riga", "bari"]
S1done_cities = []
# gcloud auth login --remote-bootstrap="https://accounts.google.com/o/oauth2/auth?response_type=code&client_id=517222506229-vsmmajv00ul0bs7p89v5m89qs8eb9359.apps.googleusercontent.com&redirect_uri=https%3A%2F%2Fsdk.cloud.google.com%2Fapplicationdefaultauthcode.html&scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fearthengine+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdevstorage.full_control+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Faccounts.reauth&state=JxOD4NAMO70eVCK3e5VK5Ga53M2CGF&prompt=consent&access_type=offline&code_challenge=JqIR_YDSAVUNu5pkOruqBeB84xNrT0mXAbTGvWmdlxU&code_challenge_method=S256"

Sentinel1_start_date = '2017-07-03'
Sentinel1_finish_date = '2017-07-30'
orbit = 'ASCENDING'

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
        # print("couldn't find country")
        country_name = None

    if country_name in rename_countries.keys():
        country_name = rename_countries[country_name]

    return country_name, polyshape


def extend_dataset(path, data_to_download=["MSBuildings"]):
    """
    :param path: path to patch folder (sen2spring)
    :return: gives the paths of all the tifs and its corresponding class labels
    """
    city_folders = glob.glob(join(path, "*"))
    f_names_all = np.array([])
    labs_all = np.array([])
    task_id = 0
    tasks = []
    for each_city in tqdm(city_folders):
        if each_city.endswith(".txt"):
            continue
        data_path = join(each_city, "sen2spring")
        data_pathMSB = join(each_city, "MSB")
        csv_path = join(each_city, each_city.split(sep)[-1:][0] + '.csv')
        city_df = pd.read_csv(csv_path)
        ids = city_df['GRD_ID']
        pop = city_df['POP']
        classes = city_df['Class']
        classes_str = [str(x) for x in classes]
        classes_paths = [data_path + '/Class_' + x + '/' for x in classes_str]
        classes_pathsMSB = [data_pathMSB + '/Class_' + x + '/' for x in classes_str]
        city = each_city.split("_")[-1] 
        
        country_name,_ = get_country_name(classes_paths[0] + str(ids[0]) + '_sen2spring.tif')

        if country_name in medium:
            print("please check if there is really data available")
        # MSB Footprints
        # if country_name in MSB_no_data_countries:
        #     # print("There is no MSB data for:", country_name)
        #     continue

        # if country_name is None:
        #     # there are some cities where countryname== is still oke, because they are by the sea
        #     if city in manually_checked_cities.keys():
        #         pass
        #         country_name = manually_checked_cities[city]
        #     else:
        #         continue
        print(city, ",", country_name)

        # if city in done_cities:
        #     continue

        for index in tqdm(range(0, len(classes_paths))):
            file_name = classes_paths[index] + str(ids[index]) + '_sen2spring.tif'

            name_MSB = str(ids[index]) + '_MSB'
            file_name_MSB = classes_pathsMSB[index] + str(ids[index]) + '_MSB'


            #MSB Footprints
            download_MSB = True
            if country_name in MSB_no_data_countries:
                download_MSB = False

            if city in MSdone_cities:
                download_MSB = False

            if country_name is None:
                # there are some cities where countryname== is still oke, because they are by the sea
                if city in MSmanually_checked_cities.keys():
                    pass
                    country_name = MSmanually_checked_cities[city]
                else:
                    download_MSB = False
            
            if city in MSmanually_blocked_cities:
                download_MSB = False

            
            download_S1 = True # TODO: change later
            if download_MSB or download_S1: 
                this_country, exportarea = get_country_name(file_name)
                this_country = country_name if this_country is None else this_country


            if download_MSB:
                Bfeature = ee.FeatureCollection('projects/sat-io/open-datasets/MSBuildings/'+this_country)
                B_filtered = Bfeature.filterBounds(exportarea)
                task = ee.batch.Export.table.toDrive(
                    collection = B_filtered,
                    description = name_MSB+"_"+city,
                    fileFormat = 'GeoJSON',
                    folder = "So2Satdata/"+classes_pathsMSB[index], 
                )
                tasks.append(task)
                task.start()
                task.status()

                task_id +=1

            if download_S1:
                collectionS1 = ee.ImageCollection('COPERNICUS/S1_GRD')
                collectionS1 = collectionS1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                collectionS1 = collectionS1.filter(ee.Filter.eq('instrumentMode', 'IW'))
                collectionS1 = collectionS1.filterDate(Sentinel1_start_date, Sentinel1_finish_date)
                collectionS1 = collectionS1.filterBounds(exportarea)
                collectionS1 = collectionS1.first()
                # collectionS1 = collectionS1.clip(exportarea)

                task_config = {
                    'description': 'imageToDriveExample',
                    'scale': 30,  
                    'region': exportarea
                }

                task = ee.batch.Export.image(collectionS1, 'exportExample', task_config)


            # TODO: Download Sentinel 1 as well!! should be quick!

            if (task_id + 1) % 3000 ==0:
                print("take a break to give google EE some time to process ;)")

    return None


def process():
    data_dir = '/scratch2/metzgern/HAC/code/So2SatPOP/data/So2Sat_POP_Part1/train'
    extend_dataset(data_dir, data_to_download=["MSBuildings"])
    return


if __name__=="__main__":



    process()
    print("Done")

    # cd /scratch2/metzgern/misc/ethzdrivemount
    # rsync -a So2Satdata* /scratch2/metzgern/HAC/code/So2SatPOP/data/GEEexport/
    # rsync -r So2Satdata* /scratch2/metzgern/HAC/code/So2SatPOP/data/GEEexport/ --ignore-existing
    # rsync -r So2Satdata* /scratch2/metzgern/HAC/code/So2SatPOP/data/GEEexport/ --ignore-existing --info=progress2 --info=name0

