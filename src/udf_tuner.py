import dask.dataframe as dd

class UserDefinedFunctionTuner:
    def __init__(self, dataframe: dd.DataFrame):
        self.dataframe = dataframe

    def apply_function(self, func, *args, **kwargs):
        """
        Apply a user-defined function to the Dask DataFrame.
        
        Parameters:
        - func: the user-defined function to apply.
        - args: positional arguments to pass to the function.
        - kwargs: keyword arguments to pass to the function.
        
        Returns:
        - result: the result of applying the function to the dataframe.
        """
        if not callable(func):
            raise ValueError("The provided function must be callable.")
        
        result = self.dataframe.map_partitions(func, *args, **kwargs)
        return result.persist()