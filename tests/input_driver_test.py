from abc import ABC, abstractmethod
import xarray as xr
import rasterio
import rioxarray
import os
import numpy as np
import unittest
import traceback
import ee
import xee

class RasterDataReaderInterface(ABC):
    @abstractmethod
    def _read_data(self):
        """
        Abstract method to read raster data from the source.
        This method should be implemented in the derived classes.
        """
        def test_read_data_not_implemented_error(self):
            # Test that instantiating RasterDataReaderInterface directly raises a TypeError
            with self.assertRaises(TypeError):
                reader = RasterDataReaderInterface()  # This should raise a TypeError

class EarthEngineInterface(ABC):
    @abstractmethod
    def _construct_ee_collection(self, parameters: dict) -> ee.ImageCollection:
        """
        Construct an Earth Engine image collection query based on parameters.

        Parameters:
        - parameters (dict): A dictionary containing parameters for the Earth Engine data.

        Returns:
        - ee.ImageCollection: Earth Engine image collection object.
        """
        
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
    
    def test_read_data(self) -> None:
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

class LocalRasterReader(RasterDataReaderInterface):
    def __init__(self, file_path: str) -> None:
        '''
        Initialize a LocalRasterReader instance.

        Parameters:
        - file_path (str): The absolute path to the raster file.

        Attributes:
        - file_path (str): The absolute path to the raster file being read.
        '''
        self._file_path = file_path
        self._xarray_data = self._read_data()
    
    def _read_data(self) -> xr.Dataset:
        try:
            with rioxarray.open_rasterio(self._file_path, band_as_variable=True) as xarray_data:
                return xarray_data
        except rasterio.errors.RasterioIOError as e:
            print(f"Error reading raster data from {e}:")
            raise

class TestEarthEngineRasterReader(unittest.TestCase):
    def setUp(self, auth_key=None):
        ''' Set up the unit tests for all methods.'''

        if auth_key:
            ee.Initialize(auth_key)
        else:
            try:
                ee.Initialize()
            except ee.EEException:
                ee.Authenticate()
                ee.Initialize()

        parameters = {
            'collection': None,
            'start_date': None,
            'end_date': None,
            'geometry': None,
            'crs': None,
            'scale': None
        }

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
            reader = EarthEngineRasterReader(parameters, auth_key=None)
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
            reader = EarthEngineRasterReader(parameters, auth_key=None)
        self.assertTrue(f"Unrecognized argument type" in str(context.exception))

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
        reader = EarthEngineRasterReader(parameters, auth_key=None)
        xarray_data = reader._xarray_data
        
        # Assert that xarray_data is not None and is an instance of xarray DataArray. What about xarray DataSet?
        self.assertIsNotNone(xarray_data)
        self.assertIsInstance(xarray_data, xr.Dataset)

class EarthEngineRasterReader(RasterDataReaderInterface, EarthEngineInterface):
    def __init__(self, parameters: dict, auth_key: str = None) -> None:
        """
        Initialize the EarthEngineRasterReader class.

        Parameters:
        - auth_key (str): Earth Engine authentication key. If None, it assumes the user is already authenticated.
        """

        if auth_key:
            ee.Initialize(auth_key)
        else:
            try:
                ee.Initialize()
            except ee.EEException:
                ee.Authenticate()
                ee.Initialize()
        
        self._xarray_data = self._read_data(parameters)
    
    def _construct_ee_collection(self, parameters: dict) -> ee.ImageCollection:
        """
        Construct an Earth Engine image collection query based on parameters.

        Parameters:
        - parameters (dict): A dictionary containing parameters for the Earth Engine data.

        Returns:
        - ee.ImageCollection: Earth Engine image collection object.
        """
        # Extract parameters with defaults
        collection = parameters.get('collection', None)
        start_date = parameters.get('start_date', None)
        end_date = parameters.get('end_date', None)
        geometry = parameters.get('geometry', None)
        # Cloud masking?

        if collection is None:
            raise ee.EEException("Earth Engine collection was not provided.")
        
        try:
            ee_collection = ee.ImageCollection(collection)

            # Optional filters
            if start_date:
                ee_collection = ee_collection.filterDate(start_date, end_date)
            if geometry:
                ee_collection = ee_collection.filterBounds(geometry)
            
            return ee_collection
        except ee.EEException:
            raise ee.EEException(f"Unrecognized argument type {type(collection)} to convert to an ImageCollection.")
    
    def _read_data(self, parameters) -> xr.Dataset:
        """
        Read Earth Engine data and convert it to xarray format.

        Parameters:
        - parameters (dict): A dictionary containing parameters for the Earth Engine data to be pulled.

        Returns:
        - xarray.Dataset: The dataset containing the Earth Engine data.
        """

        # Construct Earth Engine image collection query based on parameters
        ee_collection = self._construct_ee_collection(parameters)
        scale = parameters.get('scale', None)
        crs = parameters.get('crs', None)

        # Fetch data from Earth Engine
        xarray_data = xr.open_dataset(
            ee_collection, 
            engine='ee', 
            crs=crs, 
            scale=scale)
        
        return xarray_data


if __name__ == '__main__':
    unittest.main()