# `run()` Function

The `run()` function is the primary interface for using the `robustraster` package. It coordinates everything: setting up Dask, loading your dataset, applying your custom function, and exporting the results.

This method will support both **local raster files** and **Google Earth Engine** image collections, and it handles all the heavy lifting behind the scenes (local rasters is broken at the moment. I will update this once it's fixed!).

---

## What It Does

- Loads raster data (from Earth Engine or local files)
- Sets up and configures a Dask cluster for parallel processing
- Applies your user-defined function to the dataset
- Optionally tunes your function’s performance (via the `tune_function` parameter)
- Runs function and exports results to either local GeoTIFFs or Google Cloud Storage

---

## Key Parameters

1. `dataset` (type: `str`, `list[str]`, or `ee.ImageCollection`):  
   The input raster source.  
   - Local rasters: A string or list of file paths  
   - Earth Engine: An `ee.ImageCollection` object

2. `source` (type: `str`):  
   Defines the data source.  
   - Use `"local"` for raster files on disk  
   - Use `"ee"` for Earth Engine collections

3. `dataset_params` (type: `dict`, EE only):  
   Used to configure how Earth Engine imagery is exported. Includes:  
   - `geometry` (type: `str` or `ee.Geometry` or `ee.FeatureCollection`, optional):  
     A path to a `.geojson`, `.shp`, `.zip` (containing all `.shp` dependencies), or a native EE geometry/feature collection object.  
   - `crs` (type: `str`, optional):  
     The coordinate reference system (e.g., `"EPSG:4326"`). Defaults to `"EPSG:4326"`
   - `scale` (type: `int` or `float`, optional):  
     The spatial resolution in meters (e.g., `30`). Defualts to the native resolution.  
   - `projection` (type: `ee.Projection`, optional):  
     Custom projection for advanced use cases.

4. `user_function` (type: `Callable[[pd.DataFrame], pd.DataFrame]`):  
   Your custom processing function. It must:  
   - Accept a `pandas.DataFrame`  
   - Return a modified `DataFrame` with `x` and `y` columns preserved

5. `user_function_args` (type: `tuple`, optional):  
   A tuple of positional arguments passed to your function.
   See Example 3 in [`02_quickstart.md`](./02_quickstart.md) for an example.

6. `user_function_kwargs` (type: `dict[str, Any]`, optional):  
   A dictionary of keyword arguments passed to your function.
   See Example 4 in [`02_quickstart.md`](./02_quickstart.md) for an example.

7. `tune_function` (type: `bool`, optional):  
   Set to `True` to automatically find an optimal chunk size to break apart the dataset for efficient computation. Defaults to `False`.
   For more information, see [`05_tuning.md`](./05_tuning.md)

8. `export_params` (type: `dict`):  
   Configuration options for how and where to export results. Based on `flag`:  
   - `"GTiff"` (Local GeoTIFF export):  
     - `output_folder` (type: `str`, optional): Destination folder name for output tiles. Defaults to the name `"tiles"`.
     - `vrt` (type: `bool`, optional): Set `True` to generate a VRT file alongside your GeoTIFFS. Defaults to `False`.
   - `"GCS"` (Google Cloud Storage):  
     - `gcs_credentials` (type: `str`): Path to GCP service account JSON  
     - `gcs_bucket` (type: `str`): GCS bucket name  
     - `gcs_folder` (type: `str`, optional): Target folder in the bucket  
   - `chunks` (type: `dict` or `str`, optional): If you are familar with Dask chunking and don't want to tune your function with `tune_function`, you can pass in your own chunk size.

9. `dask_mode` (type: `str`):  
   Defines how the Dask cluster is initialized:  
   - `"full"`: (Default) Use all available local CPU and memory  
   - `"test"`: Use a single worker (for testing/debugging)  
   - `"custom"`: Use `dask_kwargs` for manual control

10. `dask_kwargs` (type: `dict[str, Any]`):  
    Manual Dask cluster configuration. Required if `dask_mode="custom"`.  
    - `n_workers` (type: `int`)  
    - `threads_per_worker` (type: `int`). Defaults to 1. I HIGHLY RECOMMEND NOT CHANGING THIS UNLESS YOU ARE ABSOLUTELY SURE YOU KNOW WHAT YOU ARE DOING!  
    - `memory_limit` (type: `str`, e.g., `"2GB"`)

11. `hooks` (type: `dict[str, Callable]`, optional):  
    Custom lifecycle hooks. Functions to run at key points:  
    - `before_run`: Called first before anything happens 
    - `after_dataset_loaded`: Called after data is opened, but before running your function
    - `after_run`: Called after processing completes

    `hooks` comes with one prebuilt `after_run` function: `preview_dataset_hook`. This will allow you to preview a portion of the data before and after the run to ensure the general structure of the data looks good before doing a full computation of the data. `preview_dataset_hook` will terminate `run()` once it gives you the preview. Remove `preview_dataset_hook` from your `run()` parameters if you ready to run your function on the dataset.
    See Example 2 in [`02_quickstart.md`](./02_quickstart.md) for an example of how to use a hook function.