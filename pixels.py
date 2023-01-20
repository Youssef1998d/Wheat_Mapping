import rasterio
from PIL import Image
from earthpy.plot import plot_bands
import matplotlib.pyplot as plt
import numpy as np
import os

path = "C:/Users/lenovo/Desktop/Assets/NDVI/Wheat_Mapping/data_ndvi_images/data_ndvi_images/"

years = ["2017", "2018", "2021", "2022"]
months = [str(i) for i in (4, 5, 6, 7, 8, 9)]
colors = [(255, 0, 0), (240, 50, 0), (200, 100, 0), (255, 200, 0), (0, 255, 0), (0, 55, 200), (0, 0, 255), (0,0,100), (0,0,0)]
def get_E_ndvi( moment=1):

    values = np.array([]) # le tableau qui va contenir les distances
    result = np.array([]) # le tableau qui va contenir le moment de tout les pixels 



    # parcourir chaque mois pour en obtenir lemoment associé

    for month in months:        
        if moment ==2:
            ob = np.array([])
        # création du matrice qui va contenir les moments de chaques mois                  
        new = np.array([])         

        # parcourir pour chaque mois toutes les années
        for year in years:                       
            src = rasterio.open(path+year+'/'+month+'_'+year+'.tiff')

            # arr va contenir la matrice spécifique a un mois pour plusieurs années lors des itérations
            arr = src.read(1)
            if moment==2:

                # Si il s'agit du calcul du 2ème moment, on enregistre les observations de l'année 
                # de chaque mois dans ob
                arr_v = arr.flatten()
                if len(ob)==0:
                    ob = arr_v
                else:
                    # ob = [ob_1, ob_2, ..., ob_n] avec n est le nombre d'années et len(ob_i)=nombre d'observations
                    ob = np.vstack((ob, arr_v))   

            if len(new)==0:

                # si new est vide, on va lui associé la 1ère matrice
                new = arr     

            else:
                # sinon on additionne a new arr par une simple addition matricielle
                new += arr
        
        # après avoir itérer toutes les années d'un mois donnée, on calcul le moment   
        
        # new va contenir la moyenne de chaque observation d'un mois itéré sur 4 années    
        new = np.array(new/len(years))
        
        if moment==1:
            if len(result)==0:
                result = new.flatten()
            else:
                result = np.vstack((result, new.flatten()))
        
        elif moment==2:

            #s'il s'agit d'un caclul du 2ème moment
            new = new.flatten()
            ob = ob.transpose()
            mean = np.repeat(new[np.newaxis, :],len(years), axis=0 )
            mean = mean.transpose()
            s = ob-mean
            s = s**2/4
            if len(result)==0:
                result=np.array(np.sum(s, axis=1))
            else:
                result = np.vstack((result, np.sum(s, axis=1)))
    
    values = result.transpose()
    return values

d = lambda moment, ref:np.linalg.norm(moment - ref, axis=1)

def generate_image(rgb_data, width, height, output_file):
    # Create a new image with the given width and height
    img = Image.new('RGB', (width, height))

    # Get the image's pixel data as a 2D array
    pixels = img.load()

    # Iterate over the array of RGB vectors and set each pixel's color
    for x in range(width):
        for y in range(height):
            pixels[x, y] = tuple(rgb_data[y * width + x])

    img.save(output_file)


def generate_colors(colors_range, momentum, ref=np.array([0,0,0,0,0,0])):
    # max of colors range must be 9, otherwise, you must scale colors more
    # change the ref vector of the momentum, otherwise, it will calculate distances between null momentum distributed evenly
    distances = d(get_E_ndvi(momentum), ref)
    color_points = np.linspace(max(distances), min(distances), colors_range)
    colors_dictionary = {color_points[i]:colors[i] for i in range(colors_range)}
    result = []
    min_, diff = 999999, 0
    for j in range(len(distances)):
        for x in colors_dictionary:
            diff = (x-distances[j])**2
            print(colors_dictionary[x], x, diff, min_, distances[j])
            if diff<min_:
                min_=diff
                c = x
        pass
        result.append(colors_dictionary[c])
    return result


arr = d(get_E_ndvi(2), np.array([0,0,0,0,0,0])).reshape(856,1089)
fig, ax = plt.subplots()
plot_bands(arr, cmap='rainbow', ax=ax)
plt.show()


