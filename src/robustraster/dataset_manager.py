from abc import ABC, abstractmethod
import xarray as xr
import rasterio
import rioxarray
import ee
import numpy as np
import pandas as pd

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

class RasterDataset(DataReaderInterface):
    """
    A reader for local raster files.

    This class provides functionality to read raster data from a specified file path
    using xarray and rioxarray. It is intended for handling raster data stored locally
    and converts it into an xarray.Dataset for further analysis.
    
    Any private attributes or methods (indicated with an underscore at the start of its name) 
    are not intended for use by the user. Documentation is provided should the user want to 
    delve deeper into how the class works, but it is not a requirement.

    Public Methods (these are functions that are openly available for users to use):
    - dataset: A property that obtains the dataset's metadata.

    Private Methods (these are functions that the user will NOT use that are called behind the scenes):
    - _read_data: A private method that reads the user's dataset into an xarray Dataset once the user 
                  instantiates the class.

    To instantiate an object of type RasterDataset:
    >>> reader = RasterDataset("/path/to/raster.tif")

    If you would like to pass in multiple raster files:
    >>> from robustraster import input_driver
    >>> raster_path_list = ['./raster1.tif', './raster2.tif']
    >>> local_raster = input_driver.RasterDataset(raster_path_list)
    """
    def __init__(self, file_path: str) -> None:
        """
        Initialize a RasterDataset instance.

        Parameters:
        - file_path (str): The absolute path to the raster file.

        Example:
        >>> reader = RasterDataset("/path/to/raster.tif")
        """
        self._file_path = file_path
        self._xarray_data = self._read_data()
    
    @property
    def dataset(self):
        """
        A property meant to retrieve the xarray Dataset stored in _xarray_data.

        Example:
        >>> local_raster = dataset_manager.RasterDataset('./raster.tif')
        >>> dataset = local_raster.dataset
        >>> print(dataset)
        """
        return self._xarray_data
    
    def _read_data(self) -> xr.Dataset:
        """
        Open the raster data and store it into an xarray object.

        Returns:
        - xr.Dataset: An xarray Dataset of the raster object.
        """
        # Ensure raster_paths is a list, even if a single string is passed
        if isinstance(self._file_path, str):
            raster_paths = [self._file_path]
        else:
            raster_paths = self._file_path
        datasets = []
    
        for i, raster_path in enumerate(raster_paths):
            try:
                with rioxarray.open_rasterio(raster_path, band_as_variable=True) as xarray_data:
                    # Add a new "index" dimension and assign the current index
                    xarray_data = xarray_data.expand_dims(time=[i + 1])  # Use i+1 for 1-based indexing

                    # Dynamically get the dimensions (first one should be 'time')
                    dim_names = list(xarray_data.dims)

                    # Initialize chunk sizes with 'time' set to 48
                    chunk_sizes = {dim_names[0]: 48}  # Assuming 'time' is the first dimension

                    # Set chunk sizes for the other dimensions (second and third)
                    if len(dim_names) >= 2:
                        chunk_sizes[dim_names[1]] = 512  # Set chunk size for the second dimension
                    if len(dim_names) >= 3:
                        chunk_sizes[dim_names[2]] = 256  # Set chunk size for the third dimension
                    
                    # Apply chunking with the defined chunk sizes
                    chunked_data = xarray_data.chunk(chunk_sizes)

                    datasets.append(chunked_data)
            except rasterio.errors.RasterioIOError as e:
                print(f"Error reading raster data from {raster_path}: {e}")
                raise
        
        # Combine all datasets along the "index" dimension
        combined_dataset = xr.concat(datasets, dim="time")

        return combined_dataset

class EarthEngineDataset(DataReaderInterface):
    """
    A reader for Google Earth Engine data.

    This class is an extension of xee (link to the package: https://github.com/google/Xee) that reads data
    from Google Earth Engine into an xarray object. It is intended to make reading data from Google Earth
    Engine to your machine a bit easier, without necessarily having to learn the xarray data structure.

    Any private attributes or methods (indicated with an underscore at the start of its name) 
    are not intended for use by the user. Documentation is provided should the user want to 
    delve deeper into how the class works, but it is not a requirement.

    Attributes:
    - _xarray_data: A private attribute of the user's queried Earth Engine data stored into an xarray object.

    - _max_chunks_limit: A private attribute that requires no user interference. This is meant to determine the
                         maximum amount of data we can pull from Google Earth Engine per request.

    Public Methods (these are functions that are openly available for users to use):

    - dataset: A property meant to retrieve the xarray Dataset stored in _xarray_data.

    - get_max_chunks_limit: A property that requires no user interference. This is called if the user wants to use
                            the tuning functionality of this package. However, the user does not need to understand
                            how to use this function (unless they choose to set a chunk size themselves).

    Private Methods (these are functions that the user will NOT use that are called behind the scenes):

    - _get_data_type_in_bytes: A private method that obtains the data type of the Earth Engine bands (stored as "data variables" in
                               the xarray object).
    
    - _auto_compute_max_chunks: A private method that stores the maximum amount of data we can pull from Google Earth Engine per
                                request into "_max_chunks_limit".
    
    - _construct_ee_collection: A private method that constructs an Earth Engine ee.ImageCollection based on the user's specified
                                parameters (see the Example below and the docstring for __init__ on how to specify parameters).
    
    - _read_data: A private method that uses xee to read the data query from Earth Engine into an xarray object.
    
    To instantiate an EarthEngineDataset object, the user must pass in a dictionary object of parameters. Below is an example
    `parameters` variable. 

    >>> parameters = {
    >>>     'collection': 'LANDSAT/LC08/C02/T1_L2',
    >>>     'bands': ['SR_B4', 'SR_B5'],
    >>>     'start_date': '2020-05-01',
    >>>     'end_date': '2020-08-31',
    >>>     'geometry': WSDemo.geometry(),
    >>>     'crs': 'EPSG:3310',
    >>>     'scale': 30,
    >>>     'map_function': prep_sr_l8
    >>> }

    Where:
    `collection` is the Earth Engine path to the image collection of interest.
    `bands` is the bands the user would like to export from Earth Engine.
    `start_date` / `end_date` is the date range to filter the image collection.
    `geometry` is the geometry object that will be used to clip the image collection.
    `crs` is the coordinate system to project the image collection to.
    `scale` is the spatial resolution the user would like the image collection to be.

    For more information on these parameters, see the documentation for Earth Engine's export
    functions (link to one here: https://developers.google.com/earth-engine/apidocs/export-image-todrive)

    `map_function` is the name of the function the user would like to run on an image 
    collection before exporting the data. See the example usage below to see how this 
    is used.

    Example usage for integrating Earth Engine with a custom cloud masking algorithm:
    1. Import required libraries and modules: 
    >>> from robustraster import dataset_manager
    >>> import ee
    >>> import json

    2. Authenticate and initialize Earth Engine:
    >>> with open(json_key, 'r') as file:
    >>>     data = json.load(file)
    >>> credentials = ee.ServiceAccountCredentials(data["client_email"], json_key)
    >>> ee.Initialize(credentials=credentials, opt_url='https://earthengine-highvolume.googleapis.com')

    3. Define a cloud masking algorithm for Landsat 8 Surface Reflectance:
    >>> def prep_sr_l8(image):
    >>>     # Bit 0 - Fill
    >>>     # Bit 1 - Dilated Cloud
    >>>     # Bit 2 - Cirrus
    >>>     # Bit 3 - Cloud
    >>>     # Bit 4 - Cloud Shadow
    >>>     qa_mask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
    >>>     saturation_mask = image.select('QA_RADSAT').eq(0)

    >>>     # Apply scaling factors to appropriate bands
    >>>     optical_bands = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
    >>>     thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)

    >>>     # Return the processed image
    >>>     return (image.addBands(optical_bands, None, True)
    >>>                 .addBands(thermal_bands, None, True)
    >>>                 .updateMask(qa_mask)
    >>>                 .updateMask(saturation_mask))

    4. Prepare parameters for data processing:
    >>> WSDemo = ee.FeatureCollection("projects/robust-raster/assets/boundaries/WSDemoSHP_Albers")
    >>> test_parameters = {
    >>>     'collection': 'LANDSAT/LC08/C02/T1_L2',
    >>>     'bands': ['SR_B4', 'SR_B5'],
    >>>     'start_date': '2020-05-01',
    >>>     'end_date': '2020-08-31',
    >>>     'geometry': WSDemo.geometry(),
    >>>     'crs': 'EPSG:3310',
    >>>     'scale': 30,
    >>>     'map_function': prep_sr_l8
    >>> }

    5. Create the EarthEngineDataset object:
    >>> earth_engine = dataset_manager.EarthEngineDataset(parameters=test_parameters)

    6. Print the contents of the data:
    >>> print(earth_engine.dataset)
    """

    def __init__(self, parameters: dict) -> None:
        """
        Instantiate the EarthEngineDataset class. To instantiate an EarthEngineDataset object, 
        the user must pass in a dictionary object of parameters. Below is an example
        `test_parameters` variable. 

        >>> test_parameters = {
        >>>     'collection': 'LANDSAT/LC08/C02/T1_L2',
        >>>     'bands': ['SR_B4', 'SR_B5'],
        >>>     'start_date': '2020-05-01',
        >>>     'end_date': '2020-08-31',
        >>>     'geometry': WSDemo.geometry(),
        >>>     'crs': 'EPSG:3310',
        >>>     'scale': 30,
        >>>     'map_function': prep_sr_l8
        >>> }

        Where:
        `collection` is the Earth Engine path to the image collection of interest.
        `bands` is the bands the user would like to export from Earth Engine.
        `start_date` / `end_date` is the date range to filter the image collection.
        `geometry` is the geometry object that will be used to clip the image collection.
        `crs` is the coordinate system to project the image collection to.
        `scale` is the spatial resolution the user would like the image collection to be.

        For more information on these parameters, see the documentation for Earth Engine's export
        functions (link to one here: https://developers.google.com/earth-engine/apidocs/export-image-todrive)

        `map_function` is the name of the function the user would like to run on an image 
        collection before exporting the data. See the example usage below to see how this 
        is used.
        
        Example usage for integrating Earth Engine with a custom cloud masking algorithm:
        1. Import required libraries and modules: 
        >>> from robustraster import dataset_manager
        >>> import ee
        >>> import json

        2. Authenticate and initialize Earth Engine:
        >>> with open(json_key, 'r') as file:
        >>>     data = json.load(file)
        >>> credentials = ee.ServiceAccountCredentials(data["client_email"], json_key)
        >>> ee.Initialize(credentials=credentials, opt_url='https://earthengine-highvolume.googleapis.com')

        3. Define a cloud masking algorithm for Landsat 8 Surface Reflectance:
        >>> def prep_sr_l8(image):
        >>>     # Bit 0 - Fill
        >>>     # Bit 1 - Dilated Cloud
        >>>     # Bit 2 - Cirrus
        >>>     # Bit 3 - Cloud
        >>>     # Bit 4 - Cloud Shadow
        >>>     qa_mask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
        >>>     saturation_mask = image.select('QA_RADSAT').eq(0)

        >>>     # Apply scaling factors to appropriate bands
        >>>     optical_bands = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
        >>>     thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)

        >>>     # Return the processed image
        >>>     return (image.addBands(optical_bands, None, True)
        >>>                 .addBands(thermal_bands, None, True)
        >>>                 .updateMask(qa_mask)
        >>>                 .updateMask(saturation_mask))

        4. Prepare parameters for data processing:
        >>> WSDemo = ee.FeatureCollection("projects/robust-raster/assets/boundaries/WSDemoSHP_Albers")
        >>> test_parameters = {
        >>>     'collection': 'LANDSAT/LC08/C02/T1_L2',
        >>>     'bands': ['SR_B4', 'SR_B5'],
        >>>     'start_date': '2020-05-01',
        >>>     'end_date': '2020-08-31',
        >>>     'geometry': WSDemo.geometry(),
        >>>     'crs': 'EPSG:3310',
        >>>     'scale': 30,
        >>>     'map_function': prep_sr_l8
        >>> }

        5. Create the EarthEngineDataset object:
        >>> earth_engine = dataset_manager.EarthEngineDataset(parameters=test_parameters)

        6. Print the contents of the data:
        >>> print(earth_engine.dataset)

        Parameters:
        - parameters (dict): A dictionary containing user parameters to query Earth Engine.
        """

        self._xarray_data = self._read_data(parameters)
        self._max_chunks_limit = self._auto_compute_max_chunks()
    
    @property
    def dataset(self) -> xr.Dataset:
        """
        A property meant to retrieve the xarray Dataset stored in _xarray_data.

        Example:
        >>> earth_engine = dataset_manager.EarthEngineDataset(parameters)
        >>> dataset = earth_engine.dataset
        >>> print(dataset)
        """
        return self._xarray_data
    
    @property
    def dataframe(self) -> pd.DataFrame:
        # Convert Xarray to a Pandas DataFrame (Defaults to long format)
        df = self._xarray_data.to_dataframe().head(5).reset_index()
        return df
    
    @property
    def get_max_chunks_limit(self) -> dict:
        """
        A property not intended for user use. This is called if the user wants to use
        the tuning functionality of this package. However, the user does not need to understand
        how to use this function (unless they choose to set a chunk size themselves).
        """
        return self._max_chunks_limit

    def _get_data_type_in_bytes(self):
        """
        A private method not intended for user use. Using an xarray Dataset object derived from Google Earth Engine, 
        obtain the data type of a single data variable. Because an ee.Image object must have the same data type 
        for all bands when exporting, it does not matter which data variable we extract the data type from 
        (I arbitrarily choose the first data variable for no particular reason).
        """

        first_data_var = list(self._xarray_data.data_vars)[0]
        return self._xarray_data[first_data_var].dtype.itemsize
 
    def _auto_compute_max_chunks(self, request_byte_limit=2**20 * 48) -> dict:
        """
        A private method not intended for user use. Computes the appropriate chunk sizes for all three 
        dimension given Earth Engine's request payload size limit. Ensures the chunk size gets as close 
        to Earth Engine's request payload size without exceeding it.
        
        Parameters:
        request_byte_limit (float): The target chunk size in megabytes. Defaults to 50.331648 MB (the max
                                    you can pull from Earth Engine in a single request).
        
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
    
        return {f'{first_dim_name}': index, 'X': width, 'Y': height}


    def _construct_ee_collection(self, parameters: dict) -> ee.ImageCollection:
        """
        A private method not intended for user use. Construct an Earth Engine image collection 
        query based on user parameters.

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
            if start_date and end_date:
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
        A private method not intended for user use. Read Earth Engine data and 
        convert it to xarray format.

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

        # Extract the sizes of each dimension dynamically
        #dims_sizes = {dim: size for dim, size in ds.sizes.items()}

        # Example chunk sizes - in this case, chunk size for each dimension is set to its full size
        # You can modify the chunking size as needed for each dimension
        #chunking = {dim: size for dim, size in dims_sizes.items()}

        # Fetch data from Earth Engine
        
        xarray_data = xr.open_dataset(
            ee_collection, 
            engine='ee', 
            crs=crs, 
            scale=scale,
            geometry=geometry)
        
        xarray_data = xarray_data.sortby('time')
        return xarray_data