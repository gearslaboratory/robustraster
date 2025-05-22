# Quick Start

## Try Out the Demo!

I highly recommend trying the demo in the GitHub repo before continuing. You don’t need a full understanding of the code yet, but briefly familiarizing yourself with it will make the rest of the documentation easier to follow.

---

## Example 1: Exporting to Local Machine

In this example, I define a function called `compute_ndvi()` that accepts and returns a pandas DataFrame.  
**Note:** Your custom function must use a DataFrame as both input and output — this is the only supported format for user-defined functions.

Then I query Earth Engine for some Landsat imagery:

- I apply a cloud masking function to clean the data.
- I filter the results by date.
- I select the Red and NIR bands.

I also use a local GEOJSON file to define my area of interest for clipping.

All dataset parameters are bundled into a Python dictionary and passed into `run()`.  

### Key Parameters

1. `dataset`: The input imagery or raster. Here, it's an Earth Engine image collection.
2. `source`: Set to `"ee"` to indicate Earth Engine is the data source.
3. `dataset_params`: A dictionary of dataset configuration options when using Earth Engine. This includes:
   - `geometry`: A file path to a local `.geojson` or `.shp` file (or an `ee.Geometry`)
   - `crs`: The coordinate reference system (e.g., `"EPSG:3310"`)
   - `scale`: The pixel resolution in meters
4. `user_function`: The function to apply to each tile of data.
5. `tune_function`: Set to `True` to auto-tune chunk sizes relative to your machine’s resources.
   - For more info, see [`tuning.md`](./tuning.md).
6. `export_params`: Defines how and where to export results.
   - Use `"GTiff"` to export GeoTIFFs locally.
   - Specify the output folder and whether to create a VRT file.
7. `dask_mode`: Set to `"full"` to use all available CPU/memory.
   - For details, see [`what_is_dask.md`](./what_is_dask.md).

---

```python
def compute_ndvi(df):
    df["ndvi"] = (df["SR_B5"] - df["SR_B4"]) / (df["SR_B5"] + df["SR_B4"])
    return df

import ee
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

# Basic cloud masking algorithm
def prep_sr_l8(image):
    # Bit 0 - Fill
    # Bit 1 - Dilated Cloud
    # Bit 2 - Cirrus
    # Bit 3 - Cloud
    # Bit 4 - Cloud Shadow
    qa_mask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
    saturation_mask = image.select('QA_RADSAT').eq(0)

    # Apply scaling
    optical_bands = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
    thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)

    return (image.addBands(optical_bands, None, True)
                 .addBands(thermal_bands, None, True)
                 .updateMask(qa_mask)
                 .updateMask(saturation_mask))

ic = (
    ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterDate('2018-05-08', '2018-05-10')
    .map(prep_sr_l8)
    .select(['SR_B4', 'SR_B5'])
)

from robustraster import run

run(
    dataset=ic,
    source="ee",
    dataset_params={
        "geometry": r"path\to\geojson-file.geojson",
        "crs": "EPSG:3310",
        "scale": 30
},
    user_function=compute_ndvi,
    tune_function=True,
    export_params={
        "flag": "GTiff", 
        "output_folder": "rush123", 
        "vrt": True},
    dask_mode="full"
)
```

---

## Example 2: Exporting to Google Cloud Storage with Custom Dask Configuration

In this example, I reuse the same `compute_ndvi()` function but change two things:

- I export the output to **Google Cloud Storage** using `flag="GCS"`.
- I configure a custom Dask setup by passing in `dask_kwargs` and setting `dask_mode="custom"`.

This provides finer control over your Dask cluster and is useful when scaling jobs or running in managed environments like HPCs or cloud platforms.

---

### Key Parameters

1. `dataset_params`: A dictionary of dataset configuration options when using Earth Engine. This includes:
   - `geometry`: A file path to a local `.geojson` or `.shp` file (or an `ee.Geometry`)
   - `crs`: The coordinate reference system (e.g., `"EPSG:3310"`)
   - `scale`: The pixel resolution in meters

2. `export_params`: Switches export mode to `"GCS"` and includes:
   - `gcs_credentials`: Path to your GCP service account credentials JSON file
   - `gcs_bucket`: Destination GCS bucket name
   - `gcs_folder`: Folder inside the bucket to save the files

3. `dask_mode`: Set to `"custom"` to enable custom cluster configuration.
4. `dask_kwargs`: A dictionary of Dask cluster settings:
   - `n_workers`: Number of Dask workers
   - `threads_per_worker`: Number of threads per worker
   - `memory_limit`: RAM limit per worker (e.g., `"2GB"`)

📌 For more details on any of these, see [`run_function.md`](./run_function.md).

---

```python
from robustraster import run

run(
    dataset=ic,
    source="ee",
    dataset_params={
        "geometry": r"path\to\geojson-file.geojson",
        "crs": "EPSG:3310",
        "scale": 30
    },
    user_function=compute_ndvi,
    export_params={
        "flag": "GCS",
        "gcs_credentials": r"path\to\service-account-credentials.json",
        "gcs_bucket": "test-bucket",
        "gcs_folder": "test-folder"
    },
    dask_mode="custom",
    dask_kwargs={
        "n_workers": 4,
        "threads_per_worker": 1,
        "memory_limit": "2GB"
    }
)
```