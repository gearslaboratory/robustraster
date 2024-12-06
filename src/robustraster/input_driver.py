from abc import ABC, abstractmethod
from .earth_engine_auth import initialize_earth_engine
import xarray as xr
import rasterio
import rioxarray
import ee
import numpy as np

class DataReaderInterface(ABC):
    @abstractmethod
    def _read_data(self):
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
    
    def _read_data(self) -> xr.Dataset:
        try:
            with rioxarray.open_rasterio(self._file_path, band_as_variable=True) as xarray_data:
                return xarray_data
        except rasterio.errors.RasterioIOError as e:
            print(f"Error reading raster data from {e}:")
            raise

class EarthEngineData(DataReaderInterface):
    def __init__(self, parameters: dict, json_key: str = None) -> None:
        """
        Initialize the EarthEngineData class. Reads in a service account credentials file (JSON format) that has permission to use the 
        Earth Engine API. If no file is passed, it will first try to initialize Earth Engine using credentials stored on the machine. If 
        it can't find the credentials stored on the machine, it will run ee.Authenticate() to create a credentials file to initialize the 
        Earth Engine API with.

        Once Earth Engine is initialized, it will use 'parameters' to query Earth Engine and store the results as an xarray Dataset. 
        This xarray Dataset is then chunked based on the Earth Engine's request payload size
        Documentation on payload size here - https://developers.google.com/earth-engine/guides/usage#request_payload_size

        Parameters:
        - parameters (dict): A dictionary containing user parameters to query Earth Engine.
        - json_key (str): Service account JSON credentials file. If None, it assumes the user is already authenticated.
        """

        #initialize_earth_engine(json_key)
        self._xarray_data = self._read_data(parameters)
        self._max_chunks_limit = self._auto_compute_max_chunks()
    
    @property
    def dataset(self):
        return self._xarray_data
    
    @property
    def get_max_chunks_limit(self):
        return self._max_chunks_limit
    
    def _get_data_type_in_bytes(self):
        '''
        Using an xarray Dataset object derived from Google Earth Engine, obtain the data type of a single 
        data variable. Because an ee.Image object must have the same data type for all bands when exporting, 
        it does not matter which data variable we extract the data type from (I arbitrarily choose the first 
        data variable for no particular reason).
        '''

        first_data_var = list(self._xarray_data.data_vars)[0]
        return self._xarray_data[first_data_var].dtype.itemsize
 
    def _auto_compute_max_chunks(self, request_byte_limit=2**20 * 48):
        """
        Computes the appropriate chunk sizes for all three dimension given Earth Engine's request 
        payload size limit. Ensures the chunk size gets as close to Earth Engine's request payload 
        size without exceeding it.
        
        Parameters:
        dim1 (int): The size of the first dimension.
        target_size_mb (float): The target chunk size in megabytes. Default is 50.331648 MB.
        
        Returns:
        dict: A dictionary containing the sizes for the first, second, and third dimensions.
        """

        # Get the name of the first dimension
        first_dim_name = list(self._xarray_data.dims)[0]

        # Get the size of the first dimension
        index = self._xarray_data.sizes[first_dim_name]

        # Given the data type size, a fixed index size, and request limit, calculate optimal chunks.
        dtype_bytes = self._get_data_type_in_bytes()
         
        # Calculate the byte size used by the given index
        index_byte_size = index * dtype_bytes
        
        # Check if the index size alone exceeds the request_byte_limit
        if index_byte_size >= request_byte_limit:
            raise ValueError("The given index size exceeds or nearly exhausts the request byte limit.")

        # Calculate the remaining bytes available for width and height dimensions
        remaining_bytes = request_byte_limit - index_byte_size
        
        # Logarithmic splitting of remaining bytes into width and height, adjusted for dtype size
        log_remaining = np.log2(remaining_bytes / dtype_bytes)  # Directly account for dtype_bytes

        # Divide log_remaining between width and height
        d = log_remaining / 2
        wd, ht = np.ceil(d), np.floor(d)

        # Convert width and height from log space to actual values
        width = int(2 ** wd)
        height = int(2 ** ht)

        # Recheck if the final size exceeds the request_byte_limit and adjust
        total_bytes = index * width * height * dtype_bytes
        while total_bytes > request_byte_limit:
            # If the total size exceeds, scale down width and height by reducing one of them
            if width > height:
                width //= 2
            else:
                height //= 2
            total_bytes = index * width * height * dtype_bytes

        actual_bytes = index * width * height * dtype_bytes
        if actual_bytes > request_byte_limit:
            raise ValueError(
                f'`chunks="auto"` failed! Actual bytes {actual_bytes!r} exceeds limit'
                f' {request_byte_limit!r}.  Please choose another value for `chunks` (and file a'
                ' bug).'
            )
    
        return {'time': index, 'X': width, 'Y': height}


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
        bands = parameters.get('bands', None)
        start_date = parameters.get('start_date', None)
        end_date = parameters.get('end_date', None)
        geometry = parameters.get('geometry', None)
        map_function = parameters.get('map_function', None)

        if collection is None:
            raise ee.EEException("Earth Engine collection was not provided.")
        
        try:
            ee_collection = ee.ImageCollection(collection)

            # Optional filters
            if start_date:
                ee_collection = ee_collection.filterDate(start_date, end_date)
            if geometry:
                ee_collection = ee_collection.filterBounds(geometry)
            if map_function and callable(map_function):
                ee_collection = ee_collection.map(map_function)
            if bands:
                ee_collection = ee_collection.select(bands)
            
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
        geometry = parameters.get('geometry', None)
        crs = parameters.get('crs', None)


        # So the payload size in Earth Engine says its 10MB, but xee found through trial and error 48 MBs.
        # When using ee.data.computePixels (which xee using in the backend), it sends a request object. 
        # This object will also contain the chunk size. To compute the size of the chunk, you can multiple 
        # each dimension and then multiply by the dtype size (if the pixels are float64, then 8 bytes). 
        # This, including the other aspects of the request object (filtering by date, cloud mask, etc.) 
        # would add up to your total payload size. To compute the bytes say filter by date takes up, you 
        # add up the characters, including white space, and multiply it by 1 byte (assuming the characters
        # are UTF-8 encoded).

        # This will be the initialize chunk size that will be used to determine an optimal 
        # chunk size for the user.
        '''test_chunk_size = {
            'time': ee_collection.size().getInfo(),
            'X': 512,
            'Y': 256
        }'''

        
        # Fetch data from Earth Engine
        xarray_data = xr.open_dataset(
            ee_collection, 
            engine='ee', 
            crs=crs, 
            scale=scale,
            geometry=geometry)
        
        xarray_data = xarray_data.sortby('time')
        return xarray_data
    '''
        # Extract the sizes of each dimension dynamically
        dims_sizes = {dim: size for dim, size in xarray_data.sizes.items()}

        # Example chunk sizes - in this case, chunk size for each dimension is set to its full size
        # You can modify the chunking size as needed for each dimension
        chunking = {dim: size for dim, size in dims_sizes.items()}

        # Re-chunk the dataset with the new chunk sizes
        chunked_dataset = xarray_data.chunk(chunking)
        
        # Chunking after loading the data bypasses a UserWarning where the chunk shape doesn't match for your
        # machine's storage array.
        return chunked_dataset
    '''
    def chunk_dataset(self, chunk_size):
        self._xarray_data = self._xarray_data.chunk(chunk_size)