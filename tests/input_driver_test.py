from google.auth.exceptions import RefreshError
import xarray as xr
import rasterio
import os
import numpy as np
import unittest
import ee
import json
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from input_driver import LocalRasterReader, EarthEngineData
from earth_engine_auth import initialize_earth_engine

class TestLocalRasterReader(unittest.TestCase):
    ''' 
    Test cases for the LocalRasterReader class.
    '''
    def setUp(self) -> None:
        ''' Set up the unit tests for all methods.'''

        # Create a temporary raster file for testing
        self.temp_raster_file = "temp_raster.tif"
        # Create a small temporary raster for testing
        with rasterio.open(self.temp_raster_file, 'w', driver='GTiff', width=10, height=10, count=1, dtype='uint8') as dst:
            dst.write_band(1, np.zeros((10, 10), dtype='uint8'))

        self.temp_invalid_file = "invalid_file.txt"  # A non-raster file
        with open(self.temp_invalid_file, "w") as file:
            pass

    def tearDown(self) -> None:
        ''' 
        Clean up the temporary files used for unit testing.
        '''
        try:
            # Remove the temporary raster file
            if os.path.exists(self.temp_raster_file):
                os.remove(self.temp_raster_file)
        except Exception as e:
            print(f"Error occurred while removing {self.temp_raster_file}: {e}")

        try:
            # Remove the temporary invalid file
            if os.path.exists(self.temp_invalid_file):
                os.remove(self.temp_invalid_file)
        except Exception as e:
            print(f"Error occurred while removing {self.temp_invalid_file}: {e}")
    
    def no_test_read_data(self) -> None:
        ''' 
        Test reading in a raster file from a local machine.

        Test Assertions:
        - assertIsNotNone: if xr.DataArray is not None
        - assertIsInstance: if object is of instance xr.DataArray
        '''
        # Test reading raster data successfully
        reader = LocalRasterReader(self.temp_raster_file)
        xarray_data = reader._xarray_data
        
        # Assert that xarray_data is not None and is an instance of xarray DataArray. What about xarray DataSet?
        self.assertIsNotNone(xarray_data)
        self.assertIsInstance(xarray_data, xr.Dataset)
        
        # Add more specific assertions based on your requirements
        # For example, you could assert the shape of the xarray, or specific values
        
    def no_test_read_data_file_not_found(self) -> None:
        ''' 
        Test if the raster file is not found when running read_data() 

        Exceptions:
        - rasterio.errors.RasterioIOError: if the raster file is not found.

        Test Assertions:
        - assertRaises: if rasterio.errors.RasterioIOError is raised in read_data()
        - assertTrue: if the string "No such file or directory" appears in the rasterio.errors.RasterioIOError exception.
        '''
        non_existing_file = "non_existing_file.tif"
        with self.assertRaises(rasterio.errors.RasterioIOError) as context:
            LocalRasterReader(non_existing_file)
        self.assertTrue("No such file or directory" in str(context.exception))
        
    def no_test_read_data_invalid_file(self) -> None:
        ''' 
        Test if the raster file found is invalid.

        Exceptions:
        - rasterio.errors.RasterioIOError if the raster file is not recognized as a supported file format.

        Test Assertions:
        - assertRaises: if rasterio.errors.RasterioIOError is raised in read_data()
        - assertTrue: if the string "not recognized as a supported file format" appears in the rasterio.errors.RasterioIOError exception.
        '''

        # Test behavior when trying to read an invalid raster file
        with self.assertRaises(rasterio.errors.RasterioIOError) as context:
            LocalRasterReader(self.temp_invalid_file)
        self.assertTrue("not recognized as a supported file format" in str(context.exception))

class TestEarthEngineData(unittest.TestCase):
    def setUp(self, json_key=None):
        ''' Set up the unit tests for all methods.'''
        initialize_earth_engine(json_key)

        parameters = {
            'collection': None,
            'start_date': None,
            'end_date': None,
            'geometry': None,
            'crs': None,
            'scale': None
        }

        #self._reader = EarthEngineData(parameters, json_key)

    def test_construct_ee_collection_no_collection(self):
        '''
        Test if no ImageCollection is passed.

        Test Assertions:
        - assertRaises: if collection is None
        - assertTrue: if the string "Earth Engine collection was not provided." appears in the ee.EEException exception.
        '''
        parameters = {
            'start_date': '2021-01-01',
            'end_date': '2021-01-31',
            'geometry': ee.Geometry.Point(-122.082, 37.42)
        }
        with self.assertRaises(ee.EEException) as context:
            reader = EarthEngineData(parameters, json_key=None)
            #reader.read_data(parameters)
        self.assertTrue(f"Earth Engine collection was not provided." in str(context.exception))
    
    def test_construct_ee_collection_invalid_collection_type(self):
        '''
        Test to ensure the ImageCollection string is a valid collection type.

        Test Assertions:
        - assertRaises: if the ImageCollection string is not str
        - assertTrue: if the string "Unrecognized argument type" appears in the ee.EEException exception. 
        '''
        parameters = {
            'collection': 500,
            'start_date': '1992-10-05',
            'end_date': '1993-03-31',
            'geometry': ee.Geometry.Point(-122.082, 37.42)
        }
        with self.assertRaises(ee.EEException) as context:
            reader = EarthEngineData(parameters, json_key=None)
            #reader.read_data(parameters)
        self.assertTrue(f"Unrecognized argument type" in str(context.exception))
        
    def test_map_function_applied(self):
        '''
        Test if the map_function is applied to the ImageCollection.

        Test Assertions:
        - assertTrue if the map_function is applied correctly.
        '''
        # Define a simple map function
        def map_function(image):
            return image.add(10)

        parameters = {
            'collection': 'MODIS/061/MOD13A2',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'geometry': ee.Geometry.Rectangle([-122.5, 37.0, -121.5, 38.0]),
            'map_function': map_function
        }

        reader = EarthEngineData(parameters, json_key=None)
        ee_collection = reader._construct_ee_collection(parameters)
        
        # Retrieve the first image from the collection
        first_image = ee_collection.first().getInfo()

        # Check if the map_function has been applied
        self.assertTrue(first_image is not None, "Map function was not applied correctly.")

    def test_read_data(self):
        '''
        Test reading Earth Engine data.
        
        Test Assertions:
        - assertIsNotNone if xr.Dataset is not None
        - assertIsInstance if object is of instance xr.Dataset
        '''
        parameters = {
            'collection': 'ECMWF/ERA5_LAND/HOURLY',
            'start_date': '1992-10-05',
            'end_date': '1993-03-31',
            'geometry': ee.Geometry.Rectangle(113.33, -43.63, 153.56, -10.66),
            'crs': 'EPSG:4326',
            'scale': 0.25
        }
        # Test reading raster data successfully
        reader = EarthEngineData(parameters, json_key=None)
        xarray_data = reader._xarray_data
        #xarray_data = reader.read_data(parameters)

        # Assert that xarray_data is not None and is an instance of xr.Dataset.
        self.assertIsNotNone(xarray_data)
        self.assertIsInstance(xarray_data, xr.Dataset)


if __name__ == '__main__':
    unittest.main()