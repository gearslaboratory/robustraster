import xarray as xr
import dask.array as da
import pandas as pd
from typing import Optional, Callable
from datetime import datetime
import json
from .dataset_manager import RasterDataset, EarthEngineDataset
from . import performance_metric_helper as pmh

from dask.distributed import performance_report

class UserDefinedFunction:
    '''
    A class meant to allow users to apply their own custom functions on a dataset derived
    from using this package (an example dataset could be an EarthEngineDataset object).
    This class can also auto-tune the user function to run on the dataset by deriving the
    user's available computational resources.

    The only two functions you (as the user) should be concerned with are `tune_user_function` 
    and `apply_user_function`. Further information on these methods can be found in the
    docstring and annotations for each respective method.

    Any private attributes or methods (indicated with an underscore at the start of its name) 
    are not intended for use by the user. Documentation is provided should the user want to 
    delve deeper into how the class works, but it is not a requirement.

    Here is an example of how to instantiate a UserDefinedFunction object, running
    `tune_user_function`, and finally running `apply_user_function`:

    In this example, I wrote my own custom function computes the NDVI on a pandas 
    DataFrame object. I then instantiate an object of type UserDefinedFunction. 
    Finally, I call the `tune_user_function` method, passing in my EarthEngineDataset
    object as well as my function as inputs.

    >>> def compute_ndvi(df):
    >>>     # Perform your calculations
    >>>     df['ndvi'] = (df['SR_B5'] - df['SR_B4']) / (df['SR_B5'] + df['SR_B4'])
    >>>     return df

    >>> from robustraster import udf_manager

    >>> user_defined_func = udf_manager.UserDefinedFunction()
    >>> user_defined_func.tune_user_function(earth_engine, compute_ndvi)

    At this point, my function as been "tuned". We can do a full run of the function
    with the following code:
    
    >>> full_result = user_defined_func.apply_user_function(earth_engine, compute_ndvi)

    Running `tune_user_function` has an optional parameter called `max_iterations`.

    Example of running `tune_user_function` and the `max_iterations` parameter.
    >>> def compute_ndvi(df):
    >>>     # Perform your calculations
    >>>     df['ndvi'] = (df['SR_B5'] - df['SR_B4']) / (df['SR_B5'] + df['SR_B4'])
    >>>     return df

    >>> from robustraster import udf_manager

    >>> user_defined_func = udf_manager.UserDefinedFunction()
    >>> user_defined_func.tune_user_function(earth_engine, compute_ndvi, max_iteration=10)

    For more information on what `max_iterations` does, refer to the docstring
    for `tune_user_function`.
    '''
    def __init__(self):
        '''
        Instantiate the UserDefinedFunction class.
        '''
        self._tuned_chunk_size = None
        self._max_chunks_limit = None

        # Initialize iteration count and count for small differences
        self.max_iterations = None
        self._iteration_count = 0
        self._small_diff_count = 0
    
    # Chunk the whole dataset
    def _create_tuning_chunk(self, ds):
        # Extract the sizes of each dimension dynamically
        dims_sizes = {dim: size for dim, size in ds.sizes.items()}

        # Example chunk sizes - in this case, chunk size for each dimension is set to its full size
        # You can modify the chunking size as needed for each dimension
        chunking = {dim: size for dim, size in dims_sizes.items()}

        # Re-chunk the dataset with the new chunk sizes
        chunked_dataset = ds.chunk(chunking)
        
        # Chunking after loading the data bypasses a UserWarning where the chunk shape doesn't match for your
        # machine's storage array.
        return chunked_dataset
    
    # Function to compute the size of a given chunk
    def _compute_chunk_in_bytes(self, dtype_size, chunk_shape):
        """Computes the total size of a chunk in bytes."""
        # Multiply each value in the dictionary by the multiplier
        result = 1
        for value in chunk_shape.values():
            result *= value

        # Multiply the product of all values by the multiplier
        return result * dtype_size
    
    def _is_chunk_bigger_than_limit(self, ds, ee_chunk_limit):
        first_data_var = list(ds.data_vars)[0]
        dtype_size = ds[first_data_var].dtype.itemsize
        chunk_shape = {dim: chunks[0] for dim, chunks in ds.chunks.items()}
        ds_chunk_bytes = self._compute_chunk_in_bytes(dtype_size, chunk_shape)
        ee_max_chunk_bytes = self._compute_chunk_in_bytes(dtype_size, ee_chunk_limit)

        if ds_chunk_bytes > ee_max_chunk_bytes:
            return True
        else:
            return False

#    def _get_starting_slice(self, ds):
#        # Get the name of the first dimension
#        first_dim_name = list(ds.dims)[0]
#
#        # Get the size of the first dimension
#        first_dim_size = ds.sizes[first_dim_name]
#
#        # Select a single chunk
#        ds_slice = ds.isel(
#            time=slice(0, first_dim_size),  # First time chunk
#            X=slice(0, 1),   # First X chunk
#            Y=slice(0, 1)    # First Y chunk
#        )
#
#        return ds_slice  

    def _get_starting_slice(self, ds):
        # Get the dimension names
        dim_names = list(ds.dims)
        
        # Find the dimension that should be sliced specifically (e.g., first dimension)
        specific_dim_name = dim_names[0]  # Assume the first dimension needs specific slicing

        # Dynamically create slices for all dimensions
        slices = {}
        for dim in dim_names:
            if dim == specific_dim_name:
                # Slice the specific dimension from 0 to its full size
                slices[dim] = slice(0, ds.sizes[dim])  
            else:
                # Slice other dimensions from 0 to 1
                slices[dim] = slice(0, 1)

        # Select the slice for the first chunk
        ds_slice = ds.isel(**slices)

        return ds_slice

    def _write_optimal_chunks_to_file(self):
        # Generate a unique file name with a prefix and timestamp
        prefix = "optimal_chunks"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
        file_name = f"{prefix}_{timestamp}.json"

        # Write to the JSON file
        with open(file_name, "w") as json_file:
            json.dump(self._tuned_chunk_size, json_file)

        print(f"Optimal chunks written to {file_name}")
        
    def _get_tuned_xarray(self, ds, ds_slice, user_func, *args, **kwargs):
        # Check if the current chunk size exceeds the EarthEngineDataset chunk limit.
        if self._max_chunks_limit:
            if self._is_chunk_bigger_than_limit(ds_slice, self._max_chunks_limit):
                print("SLICE IS BIGGER THAN DATA SOURCE'S MAX!")
                self._tuned_chunk_size = self._max_chunks_limit

                pmh.clean_up_files()
                self._write_optimal_chunks_to_file()
                return
            
        while True:
            self._iteration_count += 1
            
            test = xr.map_blocks(self._user_function_wrapper, 
                                ds_slice, 
                                args=(user_func,) + args, 
                                kwargs=kwargs)

            # Create a Dask report of the single chunked run
            with performance_report(filename="dask_report.html"):
                test.compute()
            
            # Write performance metrics to a CSV
            pmh.write_performance_metrics_to_file(ds_slice)

            # Read the CSV file into a DataFrame
            df = pd.read_csv("metrics_report.csv")

            # Do another iteration if only one entry is in the CSV
            if len(df) == 1:
                bigger_slice = self._get_bigger_slice(ds, ds_slice)
    
                # Rerun test with this new chunk size
                return self._get_tuned_xarray(ds, bigger_slice, user_func, *args, **kwargs)

            # Check if two or more iterations have been performed
            elif len(df) >= 2:
                # Get the last two Tparallel values
                previous_tparallel = df['Tparallel(pixel/worker)'].iloc[-2]
                latest_tparallel = df['Tparallel(pixel/worker)'].iloc[-1]

                # If latest is greater or equal to previous, return chunked dataset
                if latest_tparallel >= previous_tparallel:
                    self._iteration_count = 0
                    self._small_diff_count = 0

                    pmh.clean_up_files()
                    self._write_optimal_chunks_to_file()
                    return

                # If latest is smaller, check the percentage difference
                else:
                    percentage_diff = abs((previous_tparallel - latest_tparallel) / previous_tparallel) * 100
                    
                    # If the difference is less than or equal to 1%, increment the small_diff_count
                    if percentage_diff <= 1:
                        print("Difference is only less than or equal to 1%")
                        self._small_diff_count += 1

                        # If small_diff_count reaches 3, return chunked dataset
                        if self._small_diff_count >= 3:
                            self._iteration_count = 0
                            self._small_diff_count = 0

                            pmh.clean_up_files()
                            self._write_optimal_chunks_to_file()
                            return 

                        # If small_diff_count is less than 3, rerun with the same chunk size
                        return self._get_tuned_xarray(ds, ds_slice, user_func, *args, **kwargs)

                    # If the difference is greater than 1%, adjust the chunk size and rerun the test
                    else:
                        print("Difference is GREATER than 1%")
                        self._small_diff_count = 0
                        bigger_slice = self._get_bigger_slice(ds, ds_slice)
                        return self._get_tuned_xarray(ds, bigger_slice, user_func, *args, **kwargs)
            
            # Break the loop if max_iterations is set and limit is reached
            if self.max_iterations is not None and self._iteration_count >= self.max_iterations:
                self._iteration_count = 0
                self._small_diff_count = 0

                pmh.clean_up_files()
                self._write_optimal_chunks_to_file()
                return

    def _get_bigger_slice(self, ds, ds_slice):
        self._tuned_chunk_size = {dim: chunks[0] for dim, chunks in ds_slice.chunks.items()}

        # Extract the dimension names and sizes into separate lists
        dimension_names = list(ds_slice.dims)
        dimension_sizes = list(ds_slice.sizes.values())

        # Determine which dimension to double based on the iteration count
        dimension_to_double = 1 + (self._iteration_count % (len(dimension_sizes) - 1))

        # Create a dictionary of slices dynamically
        slices = {}
        for i, dim_name in enumerate(dimension_names):
            if i == 0:
                # Keep the first dimension's slice as is
                slices[dim_name] = slice(0, dimension_sizes[i])
            elif i == dimension_to_double:
                # Double the slice size of the selected dimension
                slices[dim_name] = slice(0, dimension_sizes[i] * 2)
            else:
                # Keep the slice size of other dimensions as is
                slices[dim_name] = slice(0, dimension_sizes[i])

        # Apply the slices to the dataset using isel
        new_ds_slice = ds.isel(slices)
        return new_ds_slice

    def _create_apply_chunk(self, ds, chunks=None):
        '''
        After running `tune_user_function` to obtain an optimal chunk size, chunk
        the user's dataset using this obtained chunk size in preparation to run 
        `apply_user_function`. If the user chooses to not use `tune_user_function` 
        to find an optimal chunk size, then either use a chunk size the user provides 
        or provide a default chunk size.
        '''
        # If the user passed in the JSON created from `tune_user_function`...
        if isinstance(chunks, str):
            with open(chunks, 'r') as file:
                chunks_data = json.load(file)
            ds = ds.chunk(chunks_data)
        # If the user passed in a custom chunk size as a dictionary...
        elif isinstance(chunks, dict):
            ds = ds.chunk(chunks)
        # If the user passed in None and ran `tune_user_function`...
        elif self._tuned_chunk_size:
            ds = ds.chunk(self._tuned_chunk_size)
        # If the user passed in None, did not run `tune_user_function` AND the
        # dataset is derived from an online data catalog...
        elif self._max_chunks_limit:
            ds = ds.chunk(self._max_chunks_limit)
        # If the user passed in None, did not run `tune_user_function` AND the
        # dataset is NOT derived from an online data catalog...
        else:
            safe_chunks = (48, 512, 256)
            # Create a dictionary mapping dimension names to chunk sizes
            chunk_dict = {dim: size for dim, size in zip(ds.dims, safe_chunks)}
            # Chunk the dataset
            ds = ds.chunk(chunk_dict)
        return ds

    def _generate_template_xarray(self, ds, user_func):
        # Dynamically determine dimension names
        dim_names = list(ds.sizes.keys())
        
        # Extract a single chunk to determine the output structure using dynamic dimension names
        one_chunk_slices = {dim: slice(0, ds.chunks[dim][0]) for dim in dim_names}
        one_chunk = ds.isel(**one_chunk_slices)
        
        # Apply the processing function to this chunk
        processed_chunk = user_func(one_chunk)
        
        # Create the template using a combination of original data variables and newly created ones
        template_vars = {}
        
        for var in processed_chunk.data_vars:
            if var in ds.data_vars:
                # Use the original dataset's shape and chunking for existing variables
                template_vars[var] = (processed_chunk[var].dims, 
                                    da.empty(ds[var].shape, 
                                            chunks=ds[var].chunks, 
                                            dtype=processed_chunk[var].dtype))
            else:
                # For new variables, define the shape and chunks manually based on the original chunking strategy
                new_var_shape = tuple(ds.dims[dim] for dim in processed_chunk[var].dims)
                new_var_chunks = tuple(ds.chunks[dim][0] for dim in processed_chunk[var].dims)
                template_vars[var] = (processed_chunk[var].dims, 
                                    da.empty(new_var_shape, 
                                            chunks=new_var_chunks, 
                                            dtype=processed_chunk[var].dtype))
        
        template = xr.Dataset(
            template_vars,
            coords={coord: ds.coords[coord] for coord in ds.coords},
            attrs=ds.attrs
        )
        
        return template
    
    def _user_function_wrapper(self, ds, user_func, *args, **kwargs):
        """
        Wrapper function that applies either `tune_user_function` or `apply_user_function`.
        to the user's dataset. This will convert the user's dataset to a pandas DataFrame
        first before running the user's function.
        
        Parameters:
        - user_func: the user-defined function to apply.
        - args: positional arguments to pass to the function.
        - kwargs: keyword arguments to pass to the function.
        
        Returns:
        - result: the result of applying the function to the dataframe.
        """
        
        # Look into xarray.Dataset.from_dataframe
        # Look into loading it directly to Dask b/c of warning below.
        # UserWarning: Sending large graph of size 2.15 GiB.
        # this may cause some slowdown.
        # Consider loading the data with Dask directly
        # or using futures or delayed objects to embed the data into the graph without repetition.
        # See also https://docs.dask.org/en/stable/best-practices.html#load-data-with-dask for more information.
        df_input = ds.to_dataframe().reset_index()
        df_output = user_func(df_input, *args, **kwargs)
        df_output = df_output.set_index(list(ds.dims))
        ds_output = df_output.to_xarray()
        return ds_output
        
    def tune_user_function(self, data_source: RasterDataset | EarthEngineDataset, 
                           user_func: Callable[[], pd.DataFrame], max_iterations: Optional[int] = None, 
                           *args, **kwargs):
        """
        Taking in the user's dataset object and their custom function, tune the 
        function to fit within the constraints of the user's computational infrastructure.

        How exactly does this tuning work?

        The user's dataset is constructed using xarray, a data structure that is
        an extension to Python NumPy arrays. A major advantage in using xarray is
        its ability to parallelize large datasets. This is done through what is 
        called "chunking". Documentation for chunking can be found here:
        https://docs.xarray.dev/en/stable/user-guide/dask.html#chunking-and-performance

        Chunking in xarray refers to breaking down a dataset into smaller, 
        more manageable pieces called "chunks," which are stored and processed independently. 
        These chunks allow xarray to handle datasets that are too large to fit into memory 
        all at once. So rather than running a user's function on the entire dataset, a
        set number of chunks are loaded into memory and the function is run on each chunk
        in parallel. Computed chunks are then freed from memory (and potentially written to
        disk if the user chooses so) and more uncomputed chunks are loaded.

        A common question associated with chunking is what is the best chunk size 
        for a dataset? More information on this subject can be found here:
        https://docs.dask.org/en/latest/array-best-practices.html#select-a-good-chunk-size

        Determining an optimal chunk size can be a challenge due to many external
        factors such as CPU/RAM constraints and user function complexity. This is
        what this method tries to accomplish. `tune_user_function` does the following:
        
        1. The user passes in two parameters: the dataset object and their function name. 
        
        2. The user's function is then run on a single chunk of the dataset. The initial size 
        of the chunk is the smallest it can possibly be. 
        
        3. Write the performance metrics of this run to a file. 
        
        4. Increase the chunk size by a factor of 2. 
        
        5. Repeat steps 1-4 on the newly created chunk until an optimal chunk size is found.

        6. Writes the resulting optimal chunk size to a JSON file that can be passed into
        `apply_user_function` (although if the UserDefinedFunction object that tuned the
        user's function is still instantiated, this is not required). This is useful if you
        want to run `apply_user_function` at a later time and don't want to tune your function
        again. The JSON will have your progress from the last tuning session saved.
        
        With each iteration, checks are set in place to compare the compute time of the prior 
        iteration to the newest iteration. If compute time of the most recent iteration is 
        bigger than the compute time of the last iteration, then the last iteration's chunk 
        size is returned. There are a lot more checks than what is mentioned here; I am just
        summarizing the main idea of `tune_user_function`.

        Benefits of chunking include:

        Scalability: Enables working with datasets larger than your computer's memory.
        Parallelism: Allows operations to run across multiple CPU cores or even 
                     distributed systems.
        Lazy Evaluation: Operations are deferred until explicitly computed, reducing 
                         unnecessary computations.
        Optimized I/O: Only the chunks needed for a computation are read from disk, 
                       minimizing disk access.
        
        Parameters:
        - data_source (RasterDataset or EarthEngineDataset): The user's dataset object.
        - user_func (Callable[[], pd.DataFrame]): The user's function name. For now,
                                                  user functions need to return pandas
                                                  DataFrame. See the example below for an
                                                  example function.
        - max_iterations (int): An optional argument sets the maximum number of iterations.
        - args: Positional arguments that will be passed into their custom function.
        - kwargs: Keyword arguments that will be passed into their custom function.

        Here is an example of how to instantiate a UserDefinedFunction object, running
        `tune_user_function`, and finally running `apply_user_function`:

        In this example, I wrote my own custom function that computes the NDVI on a pandas 
        DataFrame object. I then instantiate an object of type UserDefinedFunction. 
        Finally, I call the `tune_user_function` method, passing in my EarthEngineDataset
        object as well as my function as inputs.

        >>> def compute_ndvi(df):
        >>>     # Perform your calculations
        >>>     df['ndvi'] = (df['SR_B5'] - df['SR_B4']) / (df['SR_B5'] + df['SR_B4'])
        >>>     return df

        >>> from robustraster import udf_manager
        >>> from robustraster import dataset_manager

        # See the docstring for EarthEngineDataset for more info on this object type.
        >>> earth_engine = dataset_manager.EarthEngineDataset(parameters=test_parameters)

        >>> user_defined_func = udf_manager.UserDefinedFunction()
        >>> user_defined_func.tune_user_function(data_source=earth_engine, user_func=compute_ndvi)

        At this point, my function as been "tuned". We can do a full run of the function
        with the following code:
        
        >>> full_result = user_defined_func.apply_user_function(data_source=earth_engine, 
                                            user_func=compute_ndvi)

        Running `tune_user_function` has an optional parameter called `max_iterations`.

        Example of running `tune_user_function` and the `max_iterations` parameter.
        >>> def compute_ndvi(df):
        >>>     # Perform your calculations
        >>>     df['ndvi'] = (df['SR_B5'] - df['SR_B4']) / (df['SR_B5'] + df['SR_B4'])
        >>>     return df

        >>> from robustraster import udf_manager

        >>> user_defined_func = udf_manager.UserDefinedFunction()
        >>> user_defined_func.tune_user_function(data_source=earth_engine, user_func=compute_ndvi, 
                                                 max_iterations=10)

        This will perform steps 1-4 ten times.

        If your function requires multiple parameters to be passed:

        # In this example, I added a new positional argument, `numba`.
        >>> def compute_ndvi(df, numba):
        >>>     # Perform your calculations
        >>>     df['ndvi'] = (df['SR_B5'] - df['SR_B4'] + numba) / (df['SR_B5'] + df['SR_B4'])
        >>>     return df

        You can pass in additional positional or keyword arguments like the following:

        Example passing in an additional positional argument:
        >>> from robustraster import udf_manager

        >>> user_defined_func = udf_manager.UserDefinedFunction()
        >>> user_defined_func.tune_user_function(data_source=earth_engine, 
                                                 user_func=compute_ndvi, 
                                                 max_iterations=10, 
                                                 666)

        666 in this example will get passed into `compute_ndvi` as `numba`.

        Example passing in an additional keyword argument:
        >>> from robustraster import udf_manager

        >>> user_defined_func = udf_manager.UserDefinedFunction()
        >>> user_defined_func.tune_user_function(data_source=earth_engine, 
                                                 user_func=compute_ndvi, 
                                                 max_iterations=10, 
                                                 numba=666)
        """
        if not callable(user_func):
            raise ValueError("The provided function must be callable.")

        # Create a metrics report summarizing the performance of each test run.
        pmh.create_metrics_report()

        # No need for user interaction here! This will check if the dataset was obtained
        # from an online repo (like Google Earth Engine, for example). If it is, some
        # online data catalogs have a data quota that this code accounts for.
        self._max_chunks_limit = getattr(data_source, 'get_max_chunks_limit', None)

        # Set the maximum number of times the tuning code can iterate over the data.
        self.max_iterations = max_iterations

        # Create a single chunk that's the size of the dataset. Then pull a slice of size 1
        # from the chunk to use for testing.
        ds = data_source.dataset
        ds_chunked = self._create_tuning_chunk(ds)
        ds_slice = self._get_starting_slice(ds_chunked)
        
        return self._get_tuned_xarray(ds_chunked, ds_slice, user_func, *args, **kwargs)
        
    def apply_user_function(self, data_source: RasterDataset | EarthEngineDataset, 
                            user_func: str, chunks: Optional[dict | str] = None, *args, **kwargs):
        """
        Apply the user's custom function on the dataset. If the user ran `tune_user_function`
        prior, the user can use the same UserDefinedFunction object to run `apply_user_function`
        on their data. If done this way, this will run the tuned function on the dataset.

        Parameters:
        - data_source (RasterDataset or EarthEngineDataset): The user's dataset object.
        - user_func (Callable[[], pd.DataFrame]): The user's function name. For now,
                                                  user functions need to return pandas
                                                  DataFrames. See the example below for an
                                                  example function.
        - chunks (Optional[dict | str]): An optional parameter that allows users to pass in
                                         their own custom chunk size on the dataset. For more
                                         information on chunks, refer to the docstring for 
                                         `tune_user_function`. There is an explanation for 
                                         the benefits of chunking there. The user can pass in
                                         either a dictionary object containing the chunk parameters
                                         or a file path to the output JSON file generated when
                                         running `tune_user_function`. Otherwise,
                                         `apply_user_function` will auto-determine the appropriate
                                         chunk size for the dataset.
        - args: Positional arguments that will be passed into their custom function.
        - kwargs: Keyword arguments that will be passed into their custom function.

        Example 1: Running `tune_user_function` first, and then `apply_user_function` afterwards.
        >>> def compute_ndvi(df):
        >>>     # Perform your calculations
        >>>     df['ndvi'] = (df['SR_B5'] - df['SR_B4']) / (df['SR_B5'] + df['SR_B4'])
        >>>     return df

        >>> from robustraster import udf_manager
        >>> from robustraster import dataset_manager

        # See the docstring for EarthEngineDataset for more info on this object type.
        >>> earth_engine = dataset_manager.EarthEngineDataset(parameters)
        >>> user_defined_func = udf_manager.UserDefinedFunction()
        >>> user_defined_func.tune_user_function(data_source=earth_engine, user_func=compute_ndvi)
        >>> full_result = user_defined_func.apply_user_function(data_source=earth_engine, user_func=compute_ndvi) 

        `tune_user_function` is optional. You can run `apply_user_function` without it.

        Example 2:  Running `apply_user_function` without running `tune_user_function`.
        >>> full_result = user_defined_func.apply_user_function(earth_engine, compute_ndvi)

        Example 3: Running `apply_user_function` and passing in the JSON that 
                   `tune_user_function` generates. This could be useful if you boot up 
                   your code at a later time and don't want to run `tune_user_function`
                   to tune your function again.

        >>> full_result = user_defined_func.apply_user_function(data_source=earth_engine, user_func=compute_ndvi, 
                                                               chunks="optimal_chunks_20250124_141211.json") 

        Example 4: Running `apply_user_function` and passing in a dictionary object of the chunk
                   size. If the user wants the option to specify a custom chunk size without
                   tuning, they can do so here.
        
        >>> my_custom_chunks = {'time': 48
                                'X': 256
                                'Y': 512}

        >>> full_result = user_defined_func.apply_user_function(data_source=earth_engine, user_func=compute_ndvi, 
                                                               chunks=my_custom_chunks) 
        
        If your function requires multiple parameters to be passed:

        # In this example, I added a new positional argument, `numba`.
        >>> def compute_ndvi(df, numba):
        >>>     # Perform your calculations
        >>>     df['ndvi'] = (df['SR_B5'] - df['SR_B4'] + numba) / (df['SR_B5'] + df['SR_B4'])
        >>>     return df

        You can pass in additional positional or keyword arguments like the following:

        Example passing in an additional positional argument:
        >>> from robustraster import udf_manager

        >>> user_defined_func = udf_manager.UserDefinedFunction()
        >>> user_defined_func.apply_user_function(earth_engine, compute_ndvi, max_iteration=10, 666)

        666 in this example will get passed into `compute_ndvi` as `numba`.

        Example passing in an additional keyword argument:
        >>> from robustraster import udf_manager

        >>> user_defined_func = udf_manager.UserDefinedFunction()
        >>> user_defined_func.tune_user_function(earth_engine, compute_ndvi, max_iteration=10, numba=666)
        """
        if not callable(user_func):
            raise ValueError("The provided function must be callable.")


        ds = data_source.dataset
        ds = self._create_apply_chunk(ds, chunks)
        result = xr.map_blocks(self._user_function_wrapper, 
                               ds, 
                               args=(user_func,) + args, 
                               kwargs=kwargs)

        result.compute()
        
        return result