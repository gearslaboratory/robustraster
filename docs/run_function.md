# `run()` Function

The `run()` function is the primary interface for using the `robustraster` package. It coordinates everything: setting up Dask, loading your dataset, applying your custom function, and exporting the results.

This method supports both **local raster files** and **Google Earth Engine (EE)** image collections, and it handles all the heavy lifting behind the scenes.

---

## ✅ What It Does

- Loads raster data (from Earth Engine or local files)
- Sets up and configures a Dask cluster for parallel processing
- Applies your user-defined function to the dataset
- Optionally tunes your function’s performance (via `tune_function`)
- Exports results to either local GeoTIFFs or Google Cloud Storage

---

## 📥 Parameters

### `dataset` (str | list[str] | `ee.ImageCollection`)
Depending on the data source:
- Local: A file path or list of raster file paths
- Earth Engine: An `ee.ImageCollection` already filtered and ready

---

### `source` (str)
- `"local"` → local raster files
- `"ee"` → Google Earth Engine collection

---

### `dataset_kwargs` (dict, EE only)
Only required when `source == "ee"`. Used to describe how the EE data should be exported.

Options include:
- `vector`: Path to `.geojson`, `.shp`, or `.zip` (or an `ee.Geometry` or `ee.FeatureCollection`)
- `crs`: Desired coordinate reference system (e.g., `"EPSG:4326"`)
- `scale`: Spatial resolution (e.g., `30`)
- `projection`: An `ee.Projection()` if needed for custom transformations

---

### `user_function` (Callable[[pd.DataFrame], pd.DataFrame])
Your custom processing function. **It must:**
- Accept a `pandas.DataFrame` as input
- Return a modified `DataFrame` with the original `x` and `y` columns intact

---

### `user_function_args` (tuple, optional)
Extra positional arguments passed to your function.

### `user_function_kwargs` (dict[str, Any], optional)
Extra keyword arguments passed to your function.

---

### `tune_function` (bool)
If `True`, the function will be run repeatedly on test chunks of increasing size until an optimal configuration is found. This is useful for large datasets and can save time during processing. See [`tuning.md`](./tuning.md) for more.

---

### `export_kwargs` (dict)
Controls where and how the output is saved.

#### Common keys:
- `flag`: One of `"GTiff"` or `"GCS"`
- `chunks`: Either:
  - A dictionary specifying chunk dimensions (`{"time": 64, "X": 512, "Y": 256}`)
  - A path to a previously saved JSON tuning file

#### For `"GTiff"` export:
- `output_folder`: Folder path to write GeoTIFF tiles
- `export_vrt`: `True` to create a VRT file for easy reassembly

#### For `"GCS"` export:
- `gcs_credentials`: Path to GCP service account credentials JSON
- `gcs_bucket`: Destination bucket name (auto-creates if missing)
- `gcs_folder`: Target folder inside the bucket

---

### `dask_mode` (str)
Sets how resources are allocated for parallel computation:
- `"full"` (default): Use all CPU cores and memory
- `"test"`: Use 1 CPU core (for debugging)
- `"custom"`: Define your own cluster settings using `dask_kwargs`

---

### `dask_kwargs` (dict)
Only needed if `dask_mode == "custom"`. Keys include:
- `n_workers`: Number of Dask workers
- `threads_per_worker`: Threads per worker (default: 1)
- `memory_limit`: RAM limit per worker (e.g., `"2GB"`)

---

### `hooks` (dict[str, Callable])
Optional lifecycle hooks. Functions to run at key points:
- `"before_run"`
- `"after_dataset_loaded"`
- `"after_run"`

Each hook is passed relevant context objects if needed.

---

## 🛠️ Example

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