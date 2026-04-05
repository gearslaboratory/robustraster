import rasterio
import numpy as np
from osgeo import gdal
import os

with rasterio.open('test1.tif', 'w', driver='GTiff', width=10, height=10, count=2, dtype='uint8') as dst:
    dst.write(np.zeros((2, 10, 10), dtype='uint8'))
    dst.set_band_description(1, 'SR_B4')
    dst.set_band_description(2, 'SR_B5')

print('Building VRT...')
vrt_dataset = gdal.BuildVRT('test.vrt', ['test1.tif'])

# Try to set descriptions directly on build
first_dataset = gdal.Open('test1.tif')
try:
    for i in range(1, first_dataset.RasterCount + 1):
        desc = first_dataset.GetRasterBand(i).GetDescription()
        print('Band desc:', desc)
        if desc:
            vrt_dataset.GetRasterBand(i).SetDescription(desc)
except Exception as e:
    print('Failed to set:', e)

first_dataset = None
vrt_dataset.FlushCache()
vrt_dataset = None
print('First approach complete')

with rasterio.open('test.vrt') as src:
    print('VRT approach 1 descriptions:', src.descriptions)

# Second approach: reopen in update mode
print('Reopening in Update mode...')
vrt_dataset = gdal.Open('test.vrt', gdal.GA_Update)
first_dataset = gdal.Open('test1.tif')
try:
    for i in range(1, first_dataset.RasterCount + 1):
        desc = first_dataset.GetRasterBand(i).GetDescription()
        if desc:
            vrt_dataset.GetRasterBand(i).SetDescription(desc)
except Exception as e:
    print('Failed to set in update mode:', e)

first_dataset = None
vrt_dataset.FlushCache()
vrt_dataset = None

with rasterio.open('test.vrt') as src:
    print('VRT approach 2 descriptions:', src.descriptions)

os.remove('test1.tif')
os.remove('test.vrt')
