import rasterio
import numpy as np
import os

path = "C:/Users/lenovo/Desktop/Assets/NDVI/Wheat_Mapping/data_ndvi_images/data_ndvi_images/"

years = [str(i) for i in range(2017, 2023)]
months = [str(i) for i in (1, 3, 4, 5, 6, 7, 8, 9)]

def get_E_ndvi():
    E_ndvi = np.array([])
    for month in months:
        src = rasterio.open(path+'2017/'+month+'_2017.tiff')
        new = src.read(1)
        for year in years:
            src = rasterio.open(path+year+'/'+month+'_'+year+'.tiff')
            arr = src.read(1)
            if np.nan in arr:
                print("COULD NOT RESOLVE FOR : ", month+'_'+year+'.tiff')
                continue
            else:
                new += arr
        yield np.append(E_ndvi, new/len(years))

june, july, august, september = [x for x in get_E_ndvi()][4:]
    
observations = np.array([np.array([june[i], july[i], august[i], september[i]]) for i in range(262144)])