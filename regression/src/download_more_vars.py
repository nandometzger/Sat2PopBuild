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
    # gcloud auth application-default login --no-browser

import time
# time.sleep(0.5)

# EE_CRS = CRS.from_epsg(3857)
EE_CRS = CRS.from_epsg(4326)

MSB_no_data_countries = ["Germany", "United Kingdom", "Netherlands", "France", "Switzerland", "Ireland", "Belgium"]
medium = ["Italy", "Spain"]
MSmanually_blocked_cities = ["zaragoza", "bologna", "murcia", "alicante", "palma", "valencia",
                             "sevilla", "cordoba", "madrid", "catania", "malaga", "florence", "barcelona", "tallinn"]
good = ["Croatia", "Slovakia", "Bulgaria", "Czechia", "Romania", "Sweden", "Greece", "Austria", "Finland",
        "Denmark", "Latvia", "Bulgaria", "Lithuania", "Norway"]

MSmanually_checked_cities = {"riga": "Latvia", 
                             "bari": "Italy", "palermo": "Italy", "rome": "Italy", "milan": "Italy", "naples": "Italy", "turin": "Italy", "genoa": "Italy",
                             "bilbao": "Spain", "valladolid": "Spain","lisbon": "Portugal"}

rename_countries = {"Czechia": "Czech_Republic"}


# done_cities = ["zagreb", "budapest", "riga", "bari"]
MSdone_cities = ["zagreb", "budapest", "riga", "bari"] + ["plovdiv", "gdansk", "bucharest"] + ["murcia", "rome", "sofia",
                "warsaw", "wroclaw", "poznan", "oslo", "brno", "milan", "timisoara", "genoa", "vilnius", "lubin", "bilbao", "copenhagen", "cluj-napoca",
                "prague", "milan", "lodz", "naples"
                ]

S1done_cities = ["leipzig", "zagreb", "budapest" , "riga" , "edinburgh" , "dresden"] + [ #zagreb
                "london", "bari", "plovdiv", "gdansk", "zaragoza", "glasgow", "bucharest", "zurich", "marseille", "muenster", "stuttgart", "bielefeld"] +[
                "amsterdam", "stoke-on-trent", "sunderland", "bologna", "leicester", "murcia", "rome", "sofia", "barcelona",
                "warsaw", "nottingham", "wroclaw", "poznan", "oslo", "munich", "brno", "bremen", "timisoara", "florence", "genoa", "toulouse",
                "vilnius", "catania", "lubin", "hannover", "berlin", "alicante", "liverpool", "madrid", "reading", "rotterdam", "bilbao", "cardiff",
                "copenhagen", "nates", "karlsruhe", "cluj-napoca", "frankfurtammain", "prague", "dublin", "preston",  "tallinn",  "milan",  "lodz", "naples",
                "thessaloniki"
                 ]

# gcloud auth login --remote-bootstrap="https://accounts.google.com/o/oauth2/auth?response_type=code&client_id=517222506229-vsmmajv00ul0bs7p89v5m89qs8eb9359.apps.googleusercontent.com&redirect_uri=https%3A%2F%2Fsdk.cloud.google.com%2Fapplicationdefaultauthcode.html&scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fearthengine+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdevstorage.full_control+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Faccounts.reauth&state=JxOD4NAMO70eVCK3e5VK5Ga53M2CGF&prompt=consent&access_type=offline&code_challenge=JqIR_YDSAVUNu5pkOruqBeB84xNrT0mXAbTGvWmdlxU&code_challenge_method=S256"
Sentinel1_start_date = '2017-07-03'
Sentinel1_finish_date = '2017-08-30'
orbit = 'DESCENDING'

def start(task):
    task.start()

def get_country_name(file_name): 

    with rasterio.open(file_name) as src: 
        height = src.height
        width = src.width
        this_crs = src.crs

    # get the coordinates in their local coodinate system 
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
    MS_downlaod_dir = '/scratch2/metzgern/HAC/code/So2SatPOP/data/GEEexport/'

    city_folders = glob.glob(join(path, "*"))
    inv = True
    if inv:
        city_folders = city_folders[::-1]

    f_names_all = np.array([])
    labs_all = np.array([])
    task_id = 0
    tasks = []
    for each_city in tqdm(city_folders):
        if each_city.endswith(".txt"):
            continue
        data_path = join(each_city, "sen2spring")
        data_pathMSB = join(each_city, "MSB")
        data_pathS1 = join(each_city, "S1")
        csv_path = join(each_city, each_city.split(sep)[-1:][0] + '.csv')
        city_df = pd.read_csv(csv_path)
        ids = city_df['GRD_ID']
        pop = city_df['POP']
        classes = city_df['Class']
        classes_str = [str(x) for x in classes]
        classes_paths = [data_path + '/Class_' + x + '/' for x in classes_str]
        classes_pathsMSB = [data_pathMSB + '/Class_' + x + '/' for x in classes_str]
        classes_pathsS1 = [data_pathS1 + '/Class_' + x + '/' for x in classes_str]
        city = each_city.split("_")[-1] 

        
        country_name,_ = get_country_name(classes_paths[0] + str(ids[0]) + '_sen2spring.tif')
        print(city, ",", country_name)
        if country_name in medium:
            print("Disclaimer: please check if there is really MS data available")  
            # raise Exception("Still need to check the availablitiy of MSB footprints for cities here")
        
        # continue
        city_idx = 0
        class_city_idx = 0

        Classlist = []


        for index in tqdm(range(0, len(classes_paths))):
            file_name = classes_paths[index] + str(ids[index]) + '_sen2spring.tif'


            name_MSB = str(ids[index]) + '_MSB'
            name_S1 = str(ids[index]) + '_S1'
            file_name_MSB = classes_pathsMSB[index] + str(ids[index]) + '_MSB'
            file_name_S1 = classes_pathsS1[index] + str(ids[index]) + '_S1'


            descriptionMSB = name_MSB + "_" + city
            descriptionS1 = name_S1 + "_" + city

            folderMSB = "So2Satdata/" + classes_pathsMSB[index]
            folderMSB = folderMSB.replace("/", "／")
            geojsonfile = join(join(MS_downlaod_dir,folderMSB), descriptionMSB+".geojson")

            folderS1 = "So2Satdata/" + classes_pathsS1[index]
            folderS1 = folderS1.replace("/", "／")
            S1file = join(join(MS_downlaod_dir,folderS1), descriptionS1 + ".tif")

            #check if the MSB Footprints is needed
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
            if isfile(geojsonfile):
                download_MSB = False

            # Check if downloading of S1 is needed
            download_S1 = True 
            if city in S1done_cities:
                download_S1 = False
            if isfile(S1file):
                download_S1 = False


            if download_MSB or download_S1: 
                this_country, exportarea = get_country_name(file_name)
                if this_country in MSB_no_data_countries:
                    download_MSB = False
                this_country = country_name if this_country is None else this_country
                if classes_paths[index][-3:-1] not in Classlist:
                    Classlist.append(classes_paths[index][-3:-1])
                    class_city_idx = 0

            if download_MSB:
                Bfeature = ee.FeatureCollection('projects/sat-io/open-datasets/MSBuildings/'+this_country)
                B_filtered = Bfeature.filterBounds(exportarea)
                task = ee.batch.Export.table.toDrive(
                    collection = B_filtered,
                    description = name_MSB + "_" + city,
                    fileFormat = 'GeoJSON',
                    folder = "So2Satdata/" + classes_pathsMSB[index], 
                )
                try:
                    start(task)
                except ee.ee_exception.EEException:
                    for i in range(32):
                        print("Congrats. too-many jobs. EE is at it's limit. taking a 10s pause...") 
                        time.sleep(10)
                        try:
                            start(task)
                        except:
                            pass
                        else:
                            break
                        if i>30:
                            raise Exception("Could not submit EE job")
                tasks.append(task)
                task.status() 
                task_id +=1

            if download_S1:

                collectionS1 = ee.ImageCollection('COPERNICUS/S1_GRD')
                collectionS1 = collectionS1.filter(ee.Filter.eq('instrumentMode', 'IW'))
                collectionS1 = collectionS1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                collectionS1 = collectionS1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                collectionS1 = collectionS1.filter(ee.Filter.eq('orbitProperties_pass', orbit))
                collectionS1 = collectionS1.filterBounds(exportarea)
                collectionS1 = collectionS1.filter(ee.Filter.contains('.geo', exportarea));
                collectionS1 = collectionS1.filterDate(Sentinel1_start_date, Sentinel1_finish_date)
                collectionS1 = collectionS1.select(['VV', 'VH'])
                # collectionS1_first = ee.Image(collectionS1.sort("", False).first())
                # collectionS1_first = ee.Image(collectionS1.sort("system:time_start", False).first())
                # collectionS1_first = ee.Image(ee.List(collectionS1.toList(999)).get(0))
                collectionS1_first = collectionS1.median()
                # collectionS1_first = collectionS1.first()
                # collectionS1_first = collectionS1.mosaic()

                task = ee.batch.Export.image.toDrive(
                    image = collectionS1_first,
                    scale = 10,  
                    description = name_S1 + "_" + city,
                    fileFormat="GEOTIFF", 
                    folder = "So2Satdata/" + classes_pathsS1[index], 
                    region = exportarea,
                    crs='EPSG:4326'
                )
                try:
                    start(task)
                except ee.ee_exception.EEException:
                    for i in range(32):
                        print("Congrats. too-many jobs. EE is at it's limit. taking a 10s pause...")
                        time.sleep(10)
                        try:
                            start(task)
                        except:
                            pass
                        else:
                            break 
                        if i>30:
                            raise Exception("Could not submit EE job")
                task.status() 
                # tasks.append(task)
                task_id +=1
                task.status() 
                task_id +=1

            if city_idx==0 or class_city_idx==0:
                #preventing that the first two jobs finish at exacly the same time and create duplicate folders...
                if download_S1 or download_MSB:
                    time.sleep(15)

            if (task_id + 1) % 3000 ==0:
                print("take a break to give google EE some time to process ;)")
                time.sleep(30)

            city_idx += 1
            class_city_idx += 1

    return None


def process():
    data_dir = '/scratch2/metzgern/HAC/code/So2SatPOP/data/So2Sat_POP_Part1/train'
    # data_dir = '/scratch2/metzgern/HAC/code/So2SatPOP/data/So2Sat_POP_Part1/train'
    extend_dataset(data_dir, data_to_download=["MSBuildings"])
    return


if __name__=="__main__":



    process()
    print("Done")

    # cd /scratch2/metzgern/misc/ethzdrivemount
    # rsync -a So2Satdata* /scratch2/metzgern/HAC/code/So2SatPOP/data/GEEexport/
    # rsync -r So2Satdata* /scratch2/metzgern/HAC/code/So2SatPOP/data/GEEexport/ --ignore-existing
    # rsync -r So2Satdata* /scratch2/metzgern/HAC/code/So2SatPOP/data/GEEexport/ --ignore-existing --info=progress2 --info=name0

