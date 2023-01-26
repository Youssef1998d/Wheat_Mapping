import rasterio 

src = rasterio.open(r"Wheat_Mapping\data_ndvi_images\data_ndvi_images\2017\7_2017.tiff")
arr = src.read(1)
print(arr.shape)