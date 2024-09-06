import xarray as xr
import dask.array as da

class UserDefinedFunction:
    def __init__(self):
        pass

    def _generate_template_xarray(self, ds, user_func):
        # Dynamically determine dimension names
        dim_names = list(ds.dims.keys())
        
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
        
    
    def apply_user_function(self, ds, user_func, *args, **kwargs):
        if not callable(user_func):
            raise ValueError("The provided function must be callable.")
        '''
        template = self._generate_template_xarray(ds, user_func)
        result = xr.map_blocks(self._user_function_wrapper, 
                               ds, 
                               args=(user_func,) + args, 
                               kwargs=kwargs, 
                               template=template)
        '''
        result = xr.map_blocks(self._user_function_wrapper, 
                               ds, 
                               args=(user_func,) + args, 
                               kwargs=kwargs)
        return result