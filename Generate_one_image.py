import requests
import matplotlib.pyplot as plt
import rasterio
import numpy as np
import os
from glob import glob
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
from matplotlib.colors import ListedColormap
from calendar import monthrange
from pathlib import Path
import shutil
import os
from pathlib import Path


class Generate_one_image:
    def __init__(self) -> None:
        BASE_DIR = Path(__file__).resolve().parent
        folder = "image"
        os.mkdir(folder)
        self.dir_path = os.path.join(BASE_DIR, folder) + "/"
    def get_oath_token(self):
            
        token_url="https://services.sentinel-hub.com/oauth/token"
        request_data = {
            "client_id": "8e00344a-44f9-4cd2-832e-a021d4be3375",
            "client_secret": "I[OYjKP*]t#Ff2eY&Rt7QTr/D0NP3104-]M.7Ma?",
            "scope": "",
            "grant_type": "client_credentials",
        
        }

        # post to token url with token credentials
        # the request object stores the token response cookies
        r1 = requests.post(token_url, data=request_data)
        response = r1.json()
        token_access = response["access_token"]
        return token_access

    def generate_images(self,month,year,token,dir_path):
        num_days = monthrange(year, month)[1]
        print(str(year) + "-0" if month < 10 else "-"  + str(month) + "-"+str(num_days)+"T00:00:00Z")
        
        response_ndvi_images = requests.post('https://services.sentinel-hub.com/api/v1/process',
        headers={"Authorization" : "Bearer " + token},
        json={
            "input": {
                "bounds": {
            "bbox":  [ 9.004027,
        36.545022,
        9.022554,
        36.552331]
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                            "timeRange": {
                                "from": str(year) + "-0" + str(month) + "-01T00:00:00Z" if month < 10 else  str(year) + "-" + str(month) + "-01T00:00:00Z" ,
                                "to": str(year) + "-0" + str(month) + "-"+str(num_days)+"T00:00:00Z" if month < 10 else  str(year) + "-"  + str(month) + "-"+str(num_days)+"T00:00:00Z"
                            },
                                "maxCloudCoverage": 20,
                "mosaickingOrder": "leastCC"
                        }
                }]
            },
                "output": {
        "width": 1065,
            "height": 523,
            "responses": [
                    {
                        "identifier": "default",
                        "format": {
                            "type": "image/tiff"
                        }
                    }
                ]
        },
            "evalscript": """function setup() {
            
        return{
            input: [{
            bands: ["B04", "B08"],
            units: "REFLECTANCE"
            }],
            output: {
            id: "default",
            bands: 1,
            sampleType: SampleType.FLOAT32
            }
        }
            }

            function evaluatePixel(sample) {
                let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04)
        return [ ndvi ] }"""
        })
        print(response_ndvi_images,"year",year,"month",month)
        if response_ndvi_images.status_code == 200:
            with open( dir_path+ str(year)+"/"+str(month) + "_" +str(year) + ".tiff", 'wb') as f:
                    for chunk in response_ndvi_images.iter_content():
                        f.write(chunk)



    def display_images(self,dir_path):
        
        absolute_path = self.dir_path
        image_file_name = "4_2022.tiff"

        imgone = rasterio.open(absolute_path + image_file_name, driver="Gtiff")
        print(imgone.read())
        evaluation = imgone.read()

        ndvi_class_bins = [-np.inf, 0, 0.07, 0.1, 0.43, 0.8,  np.inf]
        # ndvi_class_bins = [-np.inf,0,0.07 ,0.239,0.687 ,0.9,  np.inf]  // for wheat in 6 and 7

        ndvi_landsat_class = np.digitize(evaluation, ndvi_class_bins)
        ndvi_landsat_class = np.ma.masked_where(
            np.ma.getmask(evaluation), ndvi_landsat_class)
        np.unique(ndvi_landsat_class)
        nbr_colors = ["black", "blue", "gray", "yellowgreen", "darkgreen"]
        nbr_cmap = ListedColormap(nbr_colors)
        ndvi_cat_names = [

            "Bare Area",
            "water",
            "uncultivated era",
            "wheat",

            "High Vegetation",
        ]
        classes_l8 = np.unique(ndvi_landsat_class)
        classes_l8 = classes_l8.tolist()
        classes_l8 = classes_l8[0:5], "yellow",
        ndvi_landsat_class = np.squeeze(ndvi_landsat_class)
        print(ndvi_landsat_class, ndvi_landsat_class.shape)
        print(np.unique(ndvi_landsat_class))
        fig, ax = plt.subplots(figsize=(18, 18))
        im = ax.imshow(ndvi_landsat_class, cmap=nbr_cmap)
        ep.draw_legend(im_ax=im, classes=classes_l8, titles=ndvi_cat_names)
        ax.set_title(
            "Landsat 8 - Normalized Difference Vegetation Index (NDVI) Classes", fontsize=14)
        ax.set_axis_off()
        plt.tight_layout()


    def image_preprocessing(self,image_path):
        absolute_path = self.dir_path
        image_file_name = "4_2022.tiff"

        one_img = rasterio.open(
            "/content/data_ndvi_images/2022/2_2022.tiff").read()
        ep.plot_bands(one_img, title="one",
                    cmap='RdYlGn',
                    )
        plt.show()

        one_img[one_img < 0.5] = np.nan
        one_img[np.isnan(one_img)] = np.mean(
            one_img[~np.isnan(one_img)])
        print(one_img, one_img.shape, "test")
        ep.plot_bands(one_img, title="one",
                    cmap='RdYlGn',
                    )
        plt.show()
        kwargs = rasterio.open("/content/data_ndvi_images/2022/2_2022.tiff").meta
        kwargs.update(
            dtype=rasterio.float32,
            count=1)

        with rasterio.open("/content/data_ndvi_images/2022/2_2022.tiff", 'w', **kwargs) as dst:
                dst.write_band(1, one_img[0].astype(rasterio.float32))

        final_img = rasterio.open("/content/data_ndvi_images/2022/2_2022.tiff").read()
        ep.plot_bands(final_img, title="one",
                      cmap='RdYlGn',)
        plt.show()



