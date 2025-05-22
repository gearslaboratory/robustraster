# `run()` Function

The `run()` function is the primary interface for using the `robustraster` package. It coordinates everything: setting up Dask, loading your dataset, applying your custom function, and exporting the results.

This method supports both **local raster files** and **Google Earth Engine (EE)** image collections, and it handles all the heavy lifting behind the scenes.

---

## What It Does

- Loads raster data (from Earth Engine or local files)
- Sets up and configures a Dask cluster for parallel processing
- Applies your user-defined function to the dataset
- Optionally tunes your function’s performance (via the `tune_function` parameter)
- Exports results to either local GeoTIFFs or Google Cloud Storage

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
   - `geometry` (type: `str` or `ee.Geometry` or `ee.FeatureCollection`):  
     A path to a `.geojson`, `.shp`, `.zip` (containing all `.shp` dependencies), or a native EE geometry/feature collection object.  
   - `crs` (type: `str`, optional):  
     The coordinate reference system (e.g., `"EPSG:4326"`)  
   - `scale` (type: `int` or `float`, optional):  
     The spatial resolution in meters (e.g., `30`)  
   - `projection` (type: `ee.Projection`, optional):  
     Custom projection for advanced use cases

4. `user_function` (type: `Callable[[pd.DataFrame], pd.DataFrame]`):  
   Your custom processing function. It must:  
   - Accept a `pandas.DataFrame`  
   - Return a modified `DataFrame` with `x` and `y` columns preserved

5. `user_function_args` (type: `tuple`, optional):  
   A tuple of positional arguments passed to your function.

6. `user_function_kwargs` (type: `dict[str, Any]`, optional):  
   A dictionary of keyword arguments passed to your function.

7. `tune_function` (type: `bool`):  
   Set to `True` to automatically find an optimal chunk size to break apart the dataset for efficient computation.  
   - For more, see [`tuning.md`](./tuning.md)

8. `export_params` (type: `dict`):  
   Configuration options for how and where to export results. Based on `flag`:  
   - `"GTiff"` (GeoTIFF export):  
     - `output_folder` (type: `str`): Destination folder for output tiles  
     - `vrt` (type: `bool`): Set `True` to generate a VRT mosaic  
   - `"GCS"` (Google Cloud Storage):  
     - `gcs_credentials` (type: `str`): Path to GCP service account JSON  
     - `gcs_bucket` (type: `str`): GCS bucket name  
     - `gcs_folder` (type: `str`): Target folder in the bucket  
     - `chunks` (type: `dict` or `str`): Dask-style chunk sizes, or path to tuning JSON

9. `dask_mode` (type: `str`):  
   Defines how the Dask cluster is initialized:  
   - `"full"`: (Default) Use all available local CPU and memory  
   - `"test"`: Use a single worker (for testing/debugging)  
   - `"custom"`: Use `dask_kwargs` for manual control

10. `dask_kwargs` (type: `dict[str, Any]`):  
    Manual Dask cluster configuration. Required if `dask_mode="custom"`.  
    - `n_workers` (type: `int`)  
    - `threads_per_worker` (type: `int`)  
    - `memory_limit` (type: `str`, e.g., `"2GB"`)

11. `hooks` (type: `dict[str, Callable]`, optional):  
    Custom lifecycle hooks. Functions to run at key points:  
    - `before_run`: Called before processing starts  
    - `after_dataset_loaded`: Called after data is opened  
    - `after_run`: Called after processing completes

---

## Example

```python
from robustraster import run

def compute_ndvi(df):
    df["ndvi"] = (df["SR_B5"] - df["SR_B4"]) / (df["SR_B5"] + df["SR_B4"])
    return df

run(
    dataset=ic,
    source="ee",
    dataset_kwargs={
        "vector": "my_aoi.geojson",
        "crs": "EPSG:3310",
        "scale": 30
    },
    user_function=compute_ndvi,
    tune_function=True,
    export_kwargs={
        "flag": "GTiff",
        "output_folder": "output_tiles",
        "export_vrt": True
    },
    dask_mode="custom",
    dask_kwargs={
        "n_workers": 4,
        "threads_per_worker": 1,
        "memory_limit": "2GB"
    }
)
```