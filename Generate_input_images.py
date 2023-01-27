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

class Generate_input_images:
    def __init__(self) -> None:
        BASE_DIR = Path(__file__).resolve().parent
        os.mkdir("data_ndvi_images")
        self.dir_path = os.path.join(BASE_DIR, 'data_ndvi_images') + "/"

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
        
         for year in range(2021, 2023):

            full_path = []
            paths = os.listdir(dir_path+str(year))
            for path in paths:
                if path != '.ipynb_checkpoints' and Path(dir_path+str(year)+"/"+path).is_file():
                    full_path_image = dir_path+str(year)+"/"+path
                    full_path.append(full_path_image)
            array_stack, meta_data = es.stack(full_path, nodata=-9999)
            ep.plot_bands(array_stack, title=paths,
                        cmap='RdYlGn',
                        )
            plt.show()


    def clean_data(self,image_path):
        print(image_path, "outside of try")
        try:  
                img_raster = rasterio.open(image_path,driver="Gtiff")
                img = img_raster.read()
                count = img.flatten()
                

                if np.count_nonzero(~np.isnan(img)) < (len(count)/3) :
                    print("count 2",len(count))
                    print(np.count_nonzero(~np.isnan(img)),"nonero" , image_path  )
                    os.remove(image_path)

        except:
           
                os.remove(image_path)
          




    def execute_generate(self):
        token = self.get_oath_token()

        for year in range(2021, 2023):
            os.mkdir(self.dir_path+str(year))
            for month in (1,2,3,4,5,6,12):
                self.generate_images(month, year, token, self.dir_path)
        
        for year in range(2021, 2023):
            for image_path in os.listdir(self.dir_path+str(year)):
                path = Path(self.dir_path+str(year)+"/"+image_path)
                if image_path != '.ipynb_checkpoints' and path.is_file():
                    self.clean_data(path)
                    
        
        self.display_images(self.dir_path)
