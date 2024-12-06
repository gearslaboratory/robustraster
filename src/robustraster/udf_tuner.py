import xarray as xr
import dask.array as da
import dask
import time
import re
import csv
import os
import docker
from functools import reduce
import operator
import psutil
import math
import pandas as pd
import numpy as np


from dask.distributed import performance_report

def convert_to_seconds(time_str):
    # Dictionary to store conversion factors
    conversion_factors = {
        'ms': 1e-3,  # milliseconds to seconds
        's': 1,      # seconds to seconds
        'min': 60,   # minutes to seconds
        'h': 3600,   # hours to seconds
    }
    
    # Split the string into value and unit
    time_str = time_str.strip()
    value_str = ''.join([c for c in time_str if c.isdigit() or c == '.'])
    unit = ''.join([c for c in time_str if c.isalpha()])
    
    # Convert the value to float
    value = float(value_str)
    
    # Check if the unit is valid and perform conversion
    if unit in conversion_factors:
        return value * conversion_factors[unit]
    else:
        raise ValueError(f"Unrecognized time unit: {unit}")
    
def convert_to_gigabytes(ram_str):
    # Dictionary to store conversion factors to bytes
    conversion_factors = {
        'B': 1,                 # Bytes to bytes
        'KiB': 1024,            # Kibibytes to bytes
        'MiB': 1024**2,         # Mebibytes to bytes
        'GiB': 1024**3,         # Gibibytes to bytes
        'TiB': 1024**4,         # Tebibytes to bytes
        'KB': 1000,             # Kilobytes (decimal) to bytes
        'MB': 1000**2,          # Megabytes (decimal) to bytes
        'GB': 1000**3,          # Gigabytes (decimal) to bytes
        'TB': 1000**4           # Terabytes (decimal) to bytes
    }
    
    # Split the string into value and unit
    ram_str = ram_str.strip()
    value_str = ''.join([c for c in ram_str if c.isdigit() or c == '.'])
    unit = ''.join([c for c in ram_str if c.isalpha() or c in ['iB', 'B']])

    # Convert the value to float
    value = float(value_str)

    # Check if the unit is valid and perform conversion to bytes
    if unit in conversion_factors:
        value_in_bytes = value * conversion_factors[unit]
        # Convert bytes to gigabytes
        return value_in_bytes / (1024**3)  # Divide by 1024^3 to get GiB
    else:
        raise ValueError(f"Unrecognized RAM unit: {unit}")  
      
def get_wall_time_and_memory():
    # Open the HTML file
    with open('dask-report.html', 'r', encoding='utf-8') as file:
        # Read the content of the file
        content = file.read()

    # Step 1: Create a regex pattern to search for "compute time", capturing the number and unit
    # We include the space between the number and unit, and assume the unit is followed by another space or non-alphabet character
    compute_time_pattern = r'compute\s*time:\s*(\d+\.\d+|\d+)\s+([a-zA-Z]+)\s'

    # Step 2: Use re.search to find the first occurrence of "compute time"
    match = re.search(compute_time_pattern, content)

    # Step 3: If a match is found, extract the value and the unit
    if match:
        compute_time_value = match.group(1)  # The numeric value (e.g., "13.30")
        compute_time_unit = match.group(2)  # The unit (e.g., "s")
        compute_time_string = compute_time_value + " " + compute_time_unit
    else:
        print("Compute time not found.")

    # Let's revise the regex pattern to capture the data more flexibly
    memory_pattern_final = r'"memory",\["min: [^"]+",\s*"max: ([0-9.]+) ([a-zA-Z]+)",\s*"mean: [^"]+"\]'

    # Search again for the memory max value in the html snippet
    match_final = re.search(memory_pattern_final, content)

    if match_final:
        max_memory_value = match_final.group(1)
        max_memory_unit = match_final.group(2)
        max_memory_string = max_memory_value + " " + max_memory_unit
    else:
        max_memory_value = None
        max_memory_unit = None
        max_memory_string = max_memory_value + " " + max_memory_unit

    compute_time_seconds = convert_to_seconds(compute_time_string)
    max_memory_gb = convert_to_gigabytes(max_memory_string)

    return compute_time_seconds, max_memory_gb

def get_dask_workers_count():
    client = docker.from_env()  # Initialize Docker client
    containers = client.containers.list()  # Get list of running containers
    dask_worker_containers = [container for container in containers if 'dask-worker' in container.name]
    return len(dask_worker_containers)

def get_available_system_memory():
    # Get total available RAM in bytes
    total_ram = psutil.virtual_memory().total

    # Convert from bytes to gigabytes (GB)
    total_ram_gb = total_ram / (1024 ** 3)

    return total_ram_gb

def get_compute_time_per_pixel(ds, compute_time_seconds, max_memory_gb):
    # Assuming xarray_obj is your chunked xarray dataset
    derived_chunk_size = {dim: chunks[0] for dim, chunks in ds.chunks.items()}
    pixels_per_chunk = reduce(operator.mul, derived_chunk_size.values())

    max_workers = os.cpu_count()
    
    # Get the max workers (RAM limited)
    available_system_memory = get_available_system_memory()
    ram_safety_threshold = 0.5
    max_workers_ram_limited = min(math.floor(available_system_memory * ram_safety_threshold / max_memory_gb), max_workers)

    pixel_wall_time = compute_time_seconds / pixels_per_chunk
    parallel_pixel_wall_time = pixel_wall_time / max_workers_ram_limited


    # Prepare the data for the new row, combining value and unit in one cell
    row = [f"{derived_chunk_size}", f"{pixels_per_chunk}", f"{compute_time_seconds}", f"{max_memory_gb}",
           f"{max_workers}", f"{max_workers_ram_limited}", f"{pixel_wall_time}", f"{parallel_pixel_wall_time}"]

    return row

def write_performance_metrics_to_file(ds):
    compute_time_seconds, max_memory_gb = get_wall_time_and_memory()

    row = get_compute_time_per_pixel(ds, compute_time_seconds, max_memory_gb)

    # Open the CSV file in append mode and write the header and new row
    with open('metrics_report.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the row of results
        writer.writerow(row)
        
    #os.remove('dask-report.html')

class UserDefinedFunction:
    def __init__(self, data_source=None, max_iterations=None):
        self._chunk_size_history = None
        self._max_chunks_limit = data_source.get_max_chunks_limit

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
    
    def _create_metrics_report(self):
        # Open the CSV file in append mode and write the header and new row
        with open('metrics_report.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header if the file is new
            writer.writerow(["Chunk Size", "C", "TC(s)", "RC(GiB)", "wMax", "wRAM", "Tpixel(s/pixel)", "Tparallel(pixel/worker)"])

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

            # Create a Dask report of the single chunked run
            with performance_report(filename="dask-report.html"):
                test.compute()
            
            # Write performance metrics to a CSV
            write_performance_metrics_to_file(ds_slice)

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

    def _generate_template_xarray(self, ds_slice, user_func):
        # Apply the processing function to this chunk
        processed_chunk = user_func(ds_slice)
        
        # Create the template using a combination of original data variables and newly created ones
        template_vars = {}
        
        for var in processed_chunk.data_vars:
            if var in ds_slice.data_vars:
                # Use the original dataset's shape and chunking for existing variables
                template_vars[var] = (processed_chunk[var].dims, 
                                    da.empty(ds_slice[var].shape, 
                                            chunks=ds_slice[var].chunks, 
                                            dtype=processed_chunk[var].dtype))
            else:
                # For new variables, define the shape and chunks manually based on the original chunking strategy
                new_var_shape = tuple(ds_slice.dims[dim] for dim in processed_chunk[var].dims)
                new_var_chunks = tuple(ds_slice.chunks[dim][0] for dim in processed_chunk[var].dims)
                template_vars[var] = (processed_chunk[var].dims, 
                                    da.empty(new_var_shape, 
                                            chunks=new_var_chunks, 
                                            dtype=processed_chunk[var].dtype))
        
        template = xr.Dataset(
            template_vars,
            coords={coord: ds_slice.coords[coord] for coord in ds_slice.coords},
            attrs=ds_slice.attrs
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

        # Copy global attributes
        ds_output.attrs = ds.attrs
    
        # Copy variable attributes
        for var_name in ds.data_vars:
            if var_name in ds_output:
                ds_output[var_name].attrs = ds[var_name].attrs

        return ds_output
        
    
    def tune_user_function(self, data_source, user_func, *args, **kwargs):
        if not callable(user_func):
            raise ValueError("The provided function must be callable.")
        
        self._create_metrics_report()
        
        ds = data_source.dataset
        ds_chunked = self._chunk_data(ds)
        ds_slice = self._get_starting_slice(ds_chunked)

        # Run tests here! Then jump to the real run! #
        return self._get_tuned_xarray(ds_chunked, ds_slice, user_func, *args, **kwargs)
        
    def apply_user_function(self, data_source, user_func, *args, **kwargs):
        if not callable(user_func):
            raise ValueError("The provided function must be callable.")

        ds = data_source.dataset

        if self._chunk_size_history:
            ds_chunked = ds.chunk(self._chunk_size_history)
        else:
            ds_chunked = self._chunk_data(ds)
        result = xr.map_blocks(self._user_function_wrapper, 
                               ds_chunked, 
                               args=(user_func,) + args, 
                               kwargs=kwargs)

        result.persist()
        
        return result