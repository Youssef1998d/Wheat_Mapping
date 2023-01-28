import rasterio
from PIL import Image
from earthpy.plot import plot_bands
import matplotlib.pyplot as plt
import numpy as np
import os

path = "C:/Users/lenovo/Desktop/Assets/NDVI/Wheat_Mapping/data_ndvi_images/data_ndvi_images/bouselem/"

years = ["2017", "2018", "2022"]
months = [str(i) for i in (5, 6, 7, 8, 9)]
colors = [(255, 0, 0), (240, 50, 0), (200, 100, 0), (255, 200, 0), (0, 255, 0), (0, 55, 200), (0, 0, 255), (0,0,100), (0,0,0)]
def get_E_ndvi(all=False, moment=1):
    """
    Cette fonction lit des images tiff contenant les valeurs ndvi de chaque mois sur plusieurs années 
    Elle calculera pour chaque mois donnée la moyenne de ce mois pour finir avec une matrice moyenne de chaque mois, noté new
    On va alors disposer 6 matrices moyennes, chaque matrice represente la tendance moyenne d'un mois donnée
    
    1) Si on veut les moments sur chaque mois par rapport aux années, le 1er paramètre all=False et:
        
        1 Si le moment souhaité est le 1ér (Moyenne):
            > La fonction retournera une matrice de m colonnes (nombre de mois) et l lignes (nb pixels) 
            contanant la moyenne de chaque pixels sur chaque mois pour les nombre des années.
        
        2 Si le moment souhaité est le 2ème (variance), la fonction prendra les m matrices moyennes et calculera somme((Xi-E(X))^2)/N 
            > La fonction retournera une matrice de m colonnes (m nombre de mois) et l lignes (l nombre de pixels) avec pour chaque pixels, 
                on l'accorde la variance associé a chaque mois sur les années (observations)
        
        3 Si le moment souhaité est le 3ème (Skewness), la fonction calculera la matrice Somme((Xi-E(X))^3)/sd^3*(N-1) (N est le nombre des années, Xi sont les observations d'un mois sur des années, X est la moyenne des observations)
            > La fonction retournera une matrice qui contient le skewness associé da chaque pixels de chaque mois sur les observations des années
        
        4 Si le moment souhaité est le 4ème (Kustosis), la fonction calculera la matrice N*Somme((Xi-E(X))^4)/Somme((Xi-E(X)^2)^2) (N est le nombre des années, Xi sont les observations d'un mois sur des années, X est la moyenne des observations)
            > La fonction retournera une matrice qui contient le kurtosis associé da chaque pixels de chaque mois sur les observations des années itérés.
    
    2) Si on veut les moments sur les observations moyennes de chaque années; le 1er paramètre all=True : 
        > 1 Si on veut une matrice contenant la moyenne des mois, auquel chaque mois on en calcul sa moyenne.
        > 2 Si on veut une matrice contenant la variance des mois par rapport a la moyenne.
        > 3 Si on veut une matrice contenant la skewness.
        > 4 Si on veut la kurtosis.
    """
    values = np.array([]) # le tableau qui va contenir les distances
    result = np.array([]) # le tableau qui va contenir le moment de tout les pixels 
    mean = np.array([])


    # parcourir chaque mois pour en obtenir lemoment associé

    for month in months:        
        if moment ==2 or moment==3 or moment==4:
            ob = np.array([])
        # création du matrice qui va contenir les moments de chaques mois                  
        new = np.array([])         

        # parcourir pour chaque mois toutes les années
        for year in years:                       
            src = rasterio.open(path+year+'/'+month+'_'+year+'.tiff')

            # arr va contenir la matrice spécifique a un mois pour plusieurs années lors des itérations
            arr = src.read(1)
            if (moment==2 or moment==3 or moment ==4):

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
                # sinon, il s'agit d'un calcul d'espérance on additionne a new arr par une simple addition matricielle
                new += arr
        
        # après avoir itérer toutes les années d'un mois donnée, on calcul le moment   
        
        # new va contenir la moyenne de chaque observation d'un mois parcouru sur les années    
        new = np.array(new/len(years))
        
        if moment==1:
            if len(result)==0:
                result = new.flatten()
            else:
                result = np.vstack((result, new.flatten()))
        
        elif moment==2 and not all:

            # on ajoute a result la variance
            if len(result)==0:
                result=np.array(np.sum((ob.transpose()-np.repeat(new.flatten()[np.newaxis, :],len(years), axis=0 ).transpose())**2/4, axis=1))
            else:
                result = np.vstack((result, np.sum((ob.transpose()-np.repeat(new.flatten()[np.newaxis, :],len(years), axis=0 ).transpose())**2/4, axis=1)))
        
        elif moment==3 and not all:

            # on calcul Skewness
            skew = np.sum((ob.transpose()-np.repeat(new.flatten()[np.newaxis, :],len(years), axis=0 ).transpose())**3, axis=1)/(len(years)-1)*np.sqrt(np.sum((ob.transpose()-np.repeat(new.flatten()[np.newaxis, :],len(years), axis=0 ).transpose())**2/len(years), axis=1))**3
            if len(result)==0:
                result=np.array(skew)
            else:
                result = np.vstack((result, skew))
        elif moment==4 and not all:
    
            # on calcul kurtosis
            kurtosis = (np.sum((ob.transpose()-np.repeat(new.flatten()[np.newaxis, :],len(years), axis=0 ).transpose())**4, axis=1)/np.sum((ob.transpose()-np.repeat(new.flatten()[np.newaxis, :],len(years), axis=0 ).transpose()**2)**2, axis=1))*len(years)
            if len(result)==0:
                result=np.array(kurtosis)
            else:
                result = np.vstack((result, kurtosis))

    
    values = result.transpose()
    if moment==1 and all:
        values = np.sum(values, axis=1)/len(months)
    elif moment==2 and all: 
        values = np.sum((get_E_ndvi(False, 1)[0]-np.repeat(get_E_ndvi(True, 1)[0][np.newaxis, :],len(months), axis=0).transpose())**2, axis=1)/len(months)
    elif moment==3 and all:
        values = np.sum((get_E_ndvi(False, 1)[0]-np.repeat(get_E_ndvi(True, 1)[0][np.newaxis, :],len(months), axis=0).transpose())**3, axis=1)/(len(months)-1)*(np.sqrt(get_E_ndvi(True, 2)[0])**3) #np.sqrt(get_E_ndvi(True, 2))
    elif moment==4 and all:
        values = len(months)*2*(get_E_ndvi(True, 2)[0]**2)/(np.sum((get_E_ndvi(False, 1)[0]-np.repeat(get_E_ndvi(True, 1)[0][np.newaxis, :],len(months), axis=0).transpose()**2)**2, axis=1))

    return values, values.max(), values.min(), len(values)

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

skew = lambda arr: np.sum((np.array(arr-[np.mean(arr) for x in arr]))**3)/((len(arr)-1)*np.sqrt(np.var(arr))**3)
kurtosis = lambda arr: (np.sum((np.array([np.mean(arr) for x in arr])-arr)**4)/np.sum(arr-(np.array([np.mean(arr) for x in arr])**2)**2))/len(arr)
ref = np.array([0.2, 0.35, 0.8, 0.66, 0.25])
var = np.var(ref)
skew = skew(ref)
kurtosis = kurtosis(ref)
print(kurtosis)
print(get_E_ndvi(True, 4)[0])
moment_4 = ((get_E_ndvi(True, 4)[0]-kurtosis)**2).reshape(746,1578)
moment_3 = ((get_E_ndvi(True, 3)[0]-skew)**2).reshape(746,1578)
moment_2 = ((get_E_ndvi(True, 2)[0]-var)**2).reshape(746,1578)
moment_1 = d(get_E_ndvi(False,1)[0],ref).reshape(746,1578)

def get_index(momentum, values, tolerance):
    for i in range(len(momentum)):
        yield np.where(np.isclose(momentum[i], values[i], tolerance[i]))
momentum = [moment_1, moment_2, moment_3, moment_4]
values =  [0.4, 0.002, 0.2399, 0.00004]
tolerance = [0.05, 0.0005, 0.0005, 0.000005] 
# par exemple pour le 1er moment la distance doit etre entre 0.35 et 0.45
for x in get_index(momentum, values, tolerance):
    print(x)
fig, ax = plt.subplots()



fig = plt.figure(1)
fig.subplots_adjust(hspace=0.4, top=0.85)
fig.suptitle("Les 4 indicateurs")
moment = ("Variance", "Skewness", "Kurtosis", "Mean")
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title(moment[0])
plot_bands(moment_2, cmap='rainbow', ax=ax1, vmin=0.0015)
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title(moment[1])
plot_bands(moment_3, cmap='rainbow', ax=ax2, vmin=0.2399)
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title(moment[2])
plot_bands(moment_4, cmap='rainbow', ax=ax3, vmax=0.00005)
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title(moment[3])
plot_bands(moment_1, cmap='rainbow', ax=ax4, vmax=1)
plt.show()

#plot_bands(arr, cmap='Reds', ax=ax)
#plt.show()


