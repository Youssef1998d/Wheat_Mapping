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
from PIL import Image
from scipy.stats import skew, kurtosis
from earthpy.plot import plot_bands

class Detect_Wheat:

    def __init__(self,array_of_years,tuple_of_months):

        BASE_DIR = Path(__file__).resolve().parent
        folder = "data_ndvi_images"
        self.path = os.path.join(BASE_DIR, folder) + "/"
        self.years = array_of_years
        self.months = [str(i) for i in tuple_of_months]
        self.colors = [(255, 0, 0), (240, 50, 0), (200, 100, 0), (255, 200, 0),
                (0, 255, 0), (0, 55, 200), (0, 0, 255), (0, 0, 100), (0, 0, 0)]
        
    def get_E_ndvi(self,all=False, moment=1):

        values = np.array([])  # le tableau qui va contenir les distances
        # le tableau qui va contenir le moment de tout les pixels
        result = np.array([])
        mean = np.array([])

        # parcourir chaque mois pour en obtenir lemoment associé

        for month in self.months:
            if moment == 2 or moment == 3 or moment == 4:
                ob = np.array([])
            # création du matrice qui va contenir les moments de chaques mois
            new = np.array([])

            # parcourir pour chaque mois toutes les années
            for year in self.years:
                src = rasterio.open(self.path+year+'/'+month+'_'+year+'.tiff')

                # arr va contenir la matrice spécifique a un mois pour plusieurs années lors des itérations
                arr = src.read(1)

                if (moment == 2 or moment == 3 or moment == 4):

                    # Si il s'agit du calcul du 2ème moment, on enregistre les observations de l'année
                    # de chaque mois dans ob
                    arr_v = arr.flatten()

                    if len(ob) == 0:
                        ob = arr_v
                    else:
                        # ob = [ob_1, ob_2, ..., ob_n] avec n est le nombre d'années et len(ob_i)=nombre d'observations
                        ob = np.vstack((ob, arr_v))

                if len(new) == 0:

                    # si new est vide, on va lui associé la 1ère matrice
                    new = arr

                else:
                    # sinon on additionne a new arr par une simple addition matricielle
                    new += arr

            # après avoir itérer toutes les années d'un mois donnée, on calcul le moment

            # new va contenir la moyenne de chaque observation d'un mois parcouru sur les années
            new = np.array(new/len(self.years))
            print("before flatten new shape", new.shape)

            if len(result) == 0:
                    result = new.flatten()
            else:
                    result = np.vstack((result, new.flatten()))
            print("after flatten new shape", new.flatten().shape)

        values = result.transpose()
        if moment == 1 and all:
            print(values.shape, "shapee of result values")

            values = np.sum(values, axis=1)/len(self.months)
            print(values.shape, "shapee of result values")
        elif moment == 2 and all:
            print(result.T, result.T.shape, "result transpose")
            print(np.var(result.T, axis=1), np.var(
                result.T, axis=1).shape, "result after variance")
            values = np.var(values, axis=1)
            print(values, values.shape, "test2 var")

        elif moment == 3 and all:
            print(values, values.shape, "result values and shape")
            print(skew(values, axis=1), skew(
                values, axis=1).shape, "skewness final")
            # print(sk(values,axis=1),"skweness---------")
            # print(values.shape)
            values = skew(values, axis=1)

        elif moment == 4 and all:
            print(values, values.shape, "result values and shape")
            print(kurtosis(values, axis=1), kurtosis(
                values, axis=1).shape, "kurtosis final")
            values = kurtosis(values, axis=1)
        elif moment == 5 and all:
            result_final = np.array([])
            print(values.shape, "values shapê")
            skeweness = skew(values, axis=1).flatten()
            result_final = skeweness
            print("skeweness shape", skeweness.shape)
            print("skeweness values shape", result_final.shape)
            kurtosiss = kurtosis(values, axis=1).flatten()
            result_final = np.vstack((result_final, kurtosiss))
            print("kurtosiss shape", kurtosiss.shape)
            print("kurtosis values shape", result_final.shape)
            variance = np.var(values, axis=1).flatten()
            print("variance shape", variance.shape)
            print("mean shape", (np.sum(values, axis=1)/len(self.months)).shape)
            mean = (np.sum(values, axis=1)/len(self.months)).flatten()
            print("mean shape", mean.shape)

            final_result = np.vstack((skeweness, kurtosiss, variance, mean))
            print(final_result.shape, "values shapê")
            final_result = final_result.T
            values = final_result

        return values, values.max(), values.min(), len(values)

    def d(self,moment, ref): return np.linalg.norm(moment - ref, axis=1)

    def eucludian_distance(self,a, b):
        substruction = a - b
        power_2 = substruction ** 2
        sqrt = np.sqrt(power_2)
        distance = sqrt
        return distance


    def eucludian_distance_multiple(self,a, b):
        subtract = np.subtract(a, b)
        power_2 = subtract ** 2
        sum = np.sum(power_2, axis=1)
        sqrt_arr = np.sqrt(sum)
        return sqrt_arr


    def display_result(self,moment, ref, color, width, height,scale):
        if moment == 1:
            print(self.get_E_ndvi(True, moment)[0], "ndvi get shape")
            print(np.mean(np.array(ref)), "get ref shape")
            print(self.eucludian_distance(self.get_E_ndvi(True, moment)[
                0], np.mean(np.array(ref))), "eucludian distance")
            arr = self.eucludian_distance(self.get_E_ndvi(True, moment)[
                0], np.mean(np.array(ref))), "eucludian distance"
            arr = np.array(arr[0])
            print(arr.shape, "arr shape")
            # print(get_E_ndvi(True, 2))
            fig, ax = plt.subplots(figsize=(18, 18))
            plot_bands(arr.reshape(height, width), cmap=color, ax=ax)
            plt.show()
        elif moment == 2:
            arr = self.eucludian_distance(self.get_E_ndvi(True, moment)[
                0], np.var(np.array(ref))), "eucludian distance"
            arr = np.array(arr[0])
            print(arr.shape, "arr shape")
            # print(get_E_ndvi(True, 2))
            fig, ax = plt.subplots(figsize=(18, 18))
            plot_bands(arr.reshape(height, width), cmap=color, ax=ax)
            plt.show()
        elif moment == 3:
            print(skew(np.array(ref)), "skew")
            arr = self.eucludian_distance(self.get_E_ndvi(True, moment)[
                0], skew(np.array(ref)))
            print(arr, "arr shape")

            # print(get_E_ndvi(True, 2))
            fig, ax = plt.subplots(figsize=(18, 18))
            plot_bands(arr.reshape(height, width), cmap=color, ax=ax)
            plt.show()
        elif moment == 4:
            print(kurtosis(np.array(ref)), "kurtosis")
            arr = self.eucludian_distance(self.get_E_ndvi(True, moment)[
                0], kurtosis(np.array(ref)))
            print(arr.shape, "arr shape of kurtosis")

            # print(get_E_ndvi(True, 2))
            fig, ax = plt.subplots(figsize=(18, 18))
            plot_bands(arr.reshape(height, width), cmap=color, ax=ax)
            plt.show()
        elif moment == 5:
            kurtosis_ref = kurtosis(np.array(ref))
            skweness_ref = skew(np.array(ref))
            variance_ref = np.var(np.array(ref))
            mean_ref = np.mean(np.array(ref))

            print(self.get_E_ndvi(True, moment)[0].shape, "shape get e ndvi")
            arr = self.eucludian_distance_multiple(self.get_E_ndvi(True, moment)[0], [
                                                skweness_ref, kurtosis_ref, variance_ref, mean_ref])
            print(arr.shape, arr, 'get arr shape')
            fig, ax = plt.subplots(figsize=(24, 24))
            plot_bands((arr).reshape(height, width),
                        cmap=color, ax=ax, vmax=scale, vmin=0)
            plt.show()


# display_result(5, [0.2004178, 0.6260978,  0.81834114,
#                    0.83477926, 0.23763539, 0.1645725], "gist_stern", 1065, 523)
