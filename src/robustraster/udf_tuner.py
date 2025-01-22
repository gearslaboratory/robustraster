import xarray as xr
import dask.array as da
import pandas as pd
from . import performance_metric_helper as pmh

from dask.distributed import performance_report

class UserDefinedFunction:
    '''
    A class meant to allow users to apply their own custom functions on a dataset derived
    from using this package (an example dataset could be an EarthEngineData object).
    This class can also auto-tune the user function to run on the dataset by deriving the
    user's available computational resources.

    The only two functions you (as the user) should be concerned with are `tune_user_function` 
    and `apply_user_function`. Further information on these methods can be found in the
    docstring and annotations for each respective method.

    Any private attributes or methods (indicated with an underscore at the start of its name) 
    are not intended for use by the user. Documentation is provided should the user want to 
    delve deeper into how the class works, but it is not a requirement.

    Example using an EarthEngineData object:

    In this example, my custom function computes the NDVI on a pandas DataFrame object.
    I then instantiate an object of type UserDefinedFunction. Finally, I call the 
    `tune_user_function` method, passing in my EarthEngineData object as well as my 
    function as inputs.

    >>> def compute_ndvi(df):
    >>>     # Perform your calculations
    >>>     df['ndvi'] = (df['SR_B5'] - df['SR_B4']) / (df['SR_B5'] + df['SR_B4'])
    >>>     return df

    >>> from robustraster import udf_tuner

    >>> user_defined_func = udf_tuner.UserDefinedFunction()
    >>> user_defined_func.tune_user_function(earth_engine, compute_ndvi)

    At this point, my function as been "tuned". We can do a full run of the function
    with the following code:
    
    >>> full_result = user_defined_func.apply_user_function(earth_engine, compute_ndvi)

    Instantiating a UserDefinedFunction object only has one optional parameter called
    `max_iterations`. This parameter is only used if the user wants to use 
    `tune_user_function` to tune their function on their dataset. 

    Example using an EarthEngineData object and the `max_iterations` parameter.
    >>> def compute_ndvi(df):
    >>>     # Perform your calculations
    >>>     df['ndvi'] = (df['SR_B5'] - df['SR_B4']) / (df['SR_B5'] + df['SR_B4'])
    >>>     return df

    >>> from robustraster import udf_tuner

    >>> user_defined_func = udf_tuner.UserDefinedFunction(max_iteration=10)
    >>> user_defined_func.tune_user_function(earth_engine, compute_ndvi)

    For more information on what `max_iterations` does, refer to the docstring
    for `tune_user_function`.
    '''
    def __init__(self, max_iterations=None):
        '''
        Instantiate the UserDefinedFunction class.

        Parameters:
        - data_source: An optional parameter. This is only used to 
        - max_iterations
        '''
        self._chunk_size_history = None
        self._max_chunks_limit = None

        #self._max_chunks_limit = data_source.get_max_chunks_limit

        # Initialize iteration count and count for small differences
        self._max_iterations = max_iterations
        self._iteration_count = 0
        self._small_diff_count = 0
    
    # Chunk the whole dataset
    def _chunk_data(self, ds):
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
    def _compute_chunk_size(self, dtype_size, chunk_shape):
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
        ds_chunk_bytes = self._compute_chunk_size(dtype_size, chunk_shape)
        ee_max_chunk_bytes = self._compute_chunk_size(dtype_size, ee_chunk_limit)

        if ds_chunk_bytes > ee_max_chunk_bytes:
            return True
        else:
            return False

    def _get_starting_slice(self, ds):
        # Get the name of the first dimension
        first_dim_name = list(ds.dims)[0]

        # Get the size of the first dimension
        first_dim_size = ds.sizes[first_dim_name]

        # Select a single chunk
        ds_slice = ds.isel(
            time=slice(0, first_dim_size),  # First time chunk
            X=slice(0, 1),   # First X chunk
            Y=slice(0, 1)    # First Y chunk
        )

        return ds_slice

    def _get_tuned_xarray(self, ds, ds_slice, user_func, *args, **kwargs):
        # Check if the current chunk size exceeds the EarthEngineData chunk limit.
        if self._max_chunks_limit:
            if self._is_chunk_bigger_than_limit(ds_slice, self._max_chunks_limit):
                print("SLICE IS BIGGER THAN EARTH ENGINE'S MAX!")
                self._chunk_size_history = self._max_chunks_limit
                return
            
        while True:
            self._iteration_count += 1

            test = xr.map_blocks(self._user_function_wrapper, 
                                ds_slice, 
                                args=(user_func,) + args, 
                                kwargs=kwargs)
            
            #chunk_shape = {dim: chunks[0] for dim, chunks in ds_slice.chunks.items()}

            # Create a Dask report of the single chunked run
            with performance_report(filename="dask-report.html"):
                test.compute()
            
            # Write performance metrics to a CSV
            pmh.write_performance_metrics_to_file(ds_slice)

            # Read the CSV file into a DataFrame
            df = pd.read_csv('metrics_report.csv')

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
            if self._max_iterations is not None and self._iteration_count >= self._max_iterations:
                self._iteration_count = 0
                self._small_diff_count = 0
                return

    def _get_bigger_slice(self, ds, ds_slice):
        self._chunk_size_history = {dim: chunks[0] for dim, chunks in ds_slice.chunks.items()}

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
        Apply a user-defined function to the Dask DataFrame.
        
        Parameters:
        - func: the user-defined function to apply.
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
        
    
    def tune_user_function(self, data_source, user_func, *args, **kwargs):
        '''
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
        for my dataset? More information can be found here:
        https://docs.dask.org/en/latest/array-best-practices.html#select-a-good-chunk-size

        Determining what the optimal chunk size is can be a challenge due to many 
        external factors such as CPU/RAM constraints and user function complexity. 
        This is what this method tries to accomplish. 
        
        1. The user passes in two parameters: the dataset object and their function name. 
        
        2. The user's function is then run on a single chunk of the dataset. The initial size 
        of the chunk is the smallest it can possibly be. 
        
        3. Write the performance metrics of this run to a file. 
        
        4. Increase the chunk size by a factor of 2. 
        
        5. Repeat steps 1-4 on the newly created chunk. 
        
        With each iteration, checks are set in place to compare the compute time of the prior 
        iteration to the newest iteration. If compute time for the most recent iteration is 
        bigger than the compute time of the last iteration, then the last iteration's chunk 
        size is returned. There are a lot more checks than what is mentioned here, but just
        to keep things as simple as possible, I'm not going to talk about it.

        Benefits of chunking include:

        Scalability: Enables working with datasets larger than your computer's memory.
        Parallelism: Allows operations to run across multiple CPU cores or even 
                     distributed systems.
        Lazy Evaluation: Operations are deferred until explicitly computed, reducing 
                         unnecessary computations.
        Optimized I/O: Only the chunks needed for a computation are read from disk, 
                       minimizing disk access.
        
        '''
        if not callable(user_func):
            raise ValueError("The provided function must be callable.")
        
        pmh.create_metrics_report()
        
        self._max_chunks_limit = getattr(data_source, 'get_max_chunks_limit', lambda: None)()

        ds = data_source.dataset
        ds_chunked = self._chunk_data(ds)
        ds_slice = self._get_starting_slice(ds_chunked)
        
        # Run tests here! Then jump to the real run! #
        return self._get_tuned_xarray(ds_chunked, ds_slice, user_func, *args, **kwargs)
        
    def apply_user_function(self, data_source, user_func, *args, **kwargs):
        if not callable(user_func):
            raise ValueError("The provided function must be callable.")

        ds = data_source.dataset
        ds = ds.chunk(self._chunk_size_history)
        result = xr.map_blocks(self._user_function_wrapper, 
                               ds, 
                               args=(user_func,) + args, 
                               kwargs=kwargs)

        result.compute()
        
        return result