from abc import ABC, abstractmethod
from earth_engine_auth import initialize_earth_engine
import xarray as xr
import rasterio
import rioxarray
import ee

class DataReaderInterface(ABC):
    @abstractmethod
    def read_data(self):
        """
        Abstract method to read raster data from the source.
        This method should be implemented in the derived classes.
        """
        def test_read_data_not_implemented_error(self):
            # Test that instantiating DataReaderInterface directly raises a TypeError
            with self.assertRaises(TypeError):
                reader = DataReaderInterface()  # This should raise a TypeError

class LocalRasterReader(DataReaderInterface):
    def __init__(self, file_path: str) -> None:
        '''
        Initialize a LocalRasterReader instance.

        Parameters:
        - file_path (str): The absolute path to the raster file.
        '''
        self._file_path = file_path
        self._xarray_data = self.read_data()
    
    def read_data(self) -> xr.Dataset:
        try:
            with rioxarray.open_rasterio(self._file_path, band_as_variable=True) as xarray_data:
                return xarray_data
        except rasterio.errors.RasterioIOError as e:
            print(f"Error reading raster data from {e}:")
            raise

class EarthEngineReader(DataReaderInterface):
    def __init__(self, json_key: str = None) -> None:
        """
        Initialize the EarthEngineReader class. Reads in a service account credentials file (JSON format) that has permission to use the 
        Earth Engine API. If no file is passed, it will first try to initialize Earth Engine using credentials stored on the machine. If 
        it can't find the credentials stored on the machine, it will run ee.Authenticate() to create a credentials file to initialize the 
        Earth Engine API with.

        Parameters:
        - json_key (str): Service account JSON credentials file. If None, it assumes the user is already authenticated.
        """
        initialize_earth_engine(json_key)
        
        #self._xarray_data = self._read_data(parameters)
    
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
    
    def read_data(self, parameters) -> xr.Dataset:
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
        geometry = parameters.get('geometry', None)
        crs = parameters.get('crs', None)

        # Fetch data from Earth Engine
        xarray_data = xr.open_dataset(
            ee_collection, 
            engine='ee', 
            crs=crs, 
            scale=scale,
            geometry=geometry)
        
        return xarray_data