import rasterio
from rasterio.io import MemoryFile
import numpy as np
from osgeo import gdal
import os

with rasterio.open('test1.tif', 'w', driver='GTiff', width=10, height=10, count=2, dtype='uint8') as dst:
    dst.write(np.zeros((2, 10, 10), dtype='uint8'))
    dst.set_band_description(1, 'SR_B4')
    dst.set_band_description(2, 'SR_B5')

gdal.BuildVRT('test.vrt', ['test1.tif']).FlushCache()

with rasterio.open('test.vrt') as src:
    print('VRT descriptions:', src.descriptions)
    
with rasterio.open('test1.tif') as src:
    print('TIF descriptions:', src.descriptions)

os.remove('test1.tif')
os.remove('test.vrt')
