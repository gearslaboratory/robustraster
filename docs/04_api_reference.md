# `run()` Function

The `run()` function is the main entry point for processing raster data with your custom logic, whether you're working with Earth Engine or local GeoTIFFs (local rasters is broken at the moment. I will update this once it's fixed!).

It handles:
- Dataset loading
- Dask cluster setup
- Running your user-defined function
- Exporting results
- Optional performance tuning

---

## Required Parameters

### 1. `dataset: str | list[str] | ee.ImageCollection`  
The input raster source.  

- For **local rasters**: A file path string or list of file path strings  
- For **Earth Engine**: An `ee.ImageCollection` object

---

### 2. `source: str`  
A string that defines the dataset source.  

- Use `"local"` for raster files on disk  
- Use `"ee"` for Earth Engine collections

---

### 3. `dataset_config: dict[str, Any]`
A dictionary of dataset configuration options. 

- `user_function Callable[[pd.DataFrame], pd.DataFrame]`: Name of the user function to be used for processing the dataset.  
- `user_function_args`: Tuple of user function arguments to pass into `user_function`.
- `user_function_kwargs`: Dictionary of user function keyword arguments to pass into `user_function`.

---

### 4. `function_tuning_config: dict[str, Any]`
A dictionary of function tuning configuration options. 

- `chunks`: Tuple of chunk sizes for optimized processing. See [`03_what_is_dask.md`](./03_what_is_dask.md) for details.
- `max_iterations`: If `tune_function=True`, you can set the amount of times the tuning process iterates to find an optimal chunk size. Defaults to `None`. See [`05_tuning.md`](./05_tuning.md) for details.
- `output_column_names`: List of column names that will be in the output result.

---

### 5. `export_config: dict[str, Any]`  
A dictionary of export configuration options.   

- `mode`: `raster` for raster export and `vector` for vector export.  
- `file_format` `GTiff` for GeoTIFF export. `CSV` and `parquet` for vector export. 
  - `output_folder`: Folder name to save outputs. 
  - `vrt`: `True` to generate a VRT mosaic. Only available for `raster` mode. Cannot be used if exporting results to the cloud.

- `export_to_gcs: boolean` True to export results to GCS. 
  - `gcs_credentials`: Path to service account credentials JSON  
  - `gcs_bucket`: Name of the GCS bucket  
  - `gcs_folder`: (Optional) Folder within the bucket to store outputs

- `upload_results_to_gee: boolean` True to automate generating an Earth Engine manifest and uploading the results from GCS to Earth Engine. Requires `export_to_gcs` to be True.
  - `gee_asset_path`: The full path to the Earth Engine Folder or ImageCollection to upload into (e.g., `projects/my-project/assets/my_ndvi_collection`). The engine will automatically append the time tag (e.g., the year) to the asset path for each generated image.

---

## Optional Parameters

### 6. `dataset_config: dict[str, Any]`  
Required only for Earth Engine datasets. Includes:

- `geometry`: Path to `.geojson`, `.shp`, `.gpkg`, zipped shapefiles, an ee.Geometry() or an ee.FeatureCollection() 
- `crs`: Coordinate reference system (e.g., `"EPSG:4326"`)  
- `scale`: Spatial resolution (e.g., `30` for 30 meters)  
- `projection`: An `ee.Projection` object  

---

### 7. `preview_dataset: bool`
Set to `True` to display a small preview of the dataset before and after excecuting your function.
This allows users to inspect the structure and content of the data to ensure it behaves as expected prior to running a full computation.
Useful for debugging.
Defaults to `False`.

---

### 8. `tune_function: bool`  
Set to `True` to automatically find an appropriate chunk size for optimized processing.  
Defaults to `False`.
See [`05_tuning.md`](./05_tuning.md) for details.

---

### 9. `dask_mode: str`  
Defines how to initialize the Dask cluster.  
Defaults to `"full"`.

- `"full"`: Use all available cores and memory  
- `"test"`: Single-threaded/single worker mode for debugging  
- `"custom"`: Requires `dask_kwargs`

---

### 10. `dask_config: dict[str, Any]`  
Used only when `dask_mode="custom"`.  

- `n_workers`: Number of workers  
- `threads_per_worker`: Threads per worker. Defaults to 1. I HIGHLY RECOMMEND NOT CHANGING THIS UNLESS YOU ARE ABSOLUTELY SURE YOU KNOW WHAT YOU ARE DOING!   
- `memory_limit`: RAM per worker (e.g., `"2g"`)

---

### 14. `hooks: dict[str, Callable]`  
Allows you to inject functions at various stages of the run lifecycle.  

Hook keys include:

- `before_run`: Runs before any processing begins  
- `after_dataset_loaded`: Runs after dataset is loaded, but before your function runs 
- `after_run`: Runs after everything is complete  


`preview_dataset` will allow you to preview a portion of the data before and after the run to ensure the general structure of the data looks good before doing a full computation. `preview_dataset` will terminate `run()` once it gives you the preview. Remove `preview_dataset` from your `run()` parameters if you ready to run your function on the dataset.

---