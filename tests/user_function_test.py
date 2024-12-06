from typing import Callable
import pandas as pd

def ufunc_wrapper(ufunc: Callable[..., any], df: pd.Dataframe, *args: any, **kwargs: any) -> any:
    ''' 
        Wrapper around user's custom function to make any preliminary changes before running
        custom function.

        Parameters:
        - ufunc (Callable[..., any]): The user's custom function.
        - df (pd.DataFrame): User's custom function will run on this pandas.DataFrame.
        - *args (Any): Additional positional arguments the user requires for their custom function can be added here.
        - **kwargs (Any): Additional keyword arguments the user requires for their custom function can be added here.

        Returns:
        - Any: Depending on the user function, this could return data type.
    '''

    # Check if the DataFrame is in a valid format/structure (i.e. are the columns properly named/organized?)

    return ufunc(df, *args, **kwargs)