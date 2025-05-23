from typing import Callable, Optional
import pandas as pd
import xarray as xr


def preview_dataset_hook(
    dataset: xr.Dataset,
    user_function: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    preview_dims: int = 10,
    preview_rows: int = 5
):
    """
    Hook to preview an xarray dataset and optionally apply the user function to the preview.

    Parameters:
    - dataset (xarray.Dataset): The dataset to preview.
    - user_function (Callable, optional): A user-defined function that accepts and returns a pandas DataFrame.
    - preview_dims (int): Max number of values to slice along each dimension.
    - preview_rows (int): Number of preview rows to print.
    """
    # Dynamically create slicing across dimensions
    # Dynamically create slicing across dimensions
    isel_kwargs = {
        dim: slice(0, min(size, preview_dims))
        for dim, size in dataset.sizes.items()
    }

    # Slice and load a small sample of the dataset
    small_slice = dataset.isel(**isel_kwargs)
    eager_data = small_slice.load()

    # Convert to DataFrame
    df_preview = eager_data.to_dataframe().reset_index()

    print("\nDataset preview:")
    print(df_preview.head(preview_rows))

    # Optionally apply the user function
    if user_function:
        try:
            df_with_output = user_function(df_preview)
            print("\nUser function output preview:")
            print(df_with_output.head(preview_rows))
        except Exception as e:
            print("\nError running user function on preview:", e)

    return

# Predefined constant with NDVI-style user function testing (replace as needed)
# from robustraster.user_functions import compute_ndvi
# PREVIEW_HOOK_WITH_FUNC = lambda ds: preview_dataset_hook(ds, user_function=compute_ndvi)