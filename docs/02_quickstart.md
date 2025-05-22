# Quick Start

## Try Out the Demo!

I highly recommend trying the demo in the GitHub repo. You don’t need a full understanding of the code yet, but briefly familiarizing yourself with it will make the rest of the documentation easier to follow.
---

## Example 1: Exporting to Local Machine

In this example, we’ll export NDVI tiles to your local machine using Earth Engine data and a user-defined function.

1. **Define a custom function** called `compute_ndvi()` that accepts and returns a pandas DataFrame.  
   - **Note** Custom functions must accept and return pandas DataFrames! This is required for all user-defined functions in `robustraster`.

2. **Query Earth Engine for Landsat imagery**:
   - **Note** Don't forget to authenticate to Google Earth Engine first!
   - Use `.filterDate()` to select a time range.
   - Apply a cloud-masking function (`prep_sr_l8()`).
   - Select bands `SR_B4` and `SR_B5`.

3. **Specify the dataset parameters** such as geometry, CRS, and scale.
   - The geometry in this example is a GEOSJON file. For more details on dataset parameters, see [`04_run_function.md`](./04_run_function.md)

4. **Call `run()`** with all parameters:
   - This example include an optional `tune_function=True` to auto-optimize chunk size. For more details function tuning and chunks, see [`05_tuning.md`](./05_tuning.md) and [`03_what_is_dask.md`](./03_what_is_dask.md)
   - Set `export_params["flag"] = "GTiff"` to export locally.
   - Set `dask_mode="full"` to use all cores. Again, refer to [`03_what_is_dask.md`](./03_what_is_dask.md) for more information.

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
        "vrt": True}
)
```

## Example 2: Exporting to Local Machine + Using a Function with Positional Arguments
```python
def compute_ndvi(df, number_to_add):
    df["ndvi"] = (df["SR_B5"] - df["SR_B4"] + number_to_add) / (df["SR_B5"] + df["SR_B4"])
    return df

run(
    dataset=ic,
    source="ee",
    dataset_params={
        "geometry": r"path\to\geojson-file.geojson",
        "crs": "EPSG:3310",
        "scale": 30
},
    user_function=compute_ndvi,
    user_function_args=(666)
    tune_function=True,
    export_params={
        "flag": "GTiff", 
        "output_folder": "rush123", 
        "vrt": True}
)
```
## Example 3: Exporting to Local Machine + Using a Function with Keyword Arguments
```python

def compute_ndvi(df, number_to_add):
    df["ndvi"] = (df["SR_B5"] - df["SR_B4"] + number_to_add) / (df["SR_B5"] + df["SR_B4"])
    return df

run(
    dataset=ic,
    source="ee",
    dataset_params={
        "geometry": r"path\to\geojson-file.geojson",
        "crs": "EPSG:3310",
        "scale": 30
},
    user_function=compute_ndvi,
    user_function_args={"number_to_add": 666}
    tune_function=True,
    export_params={
        "flag": "GTiff", 
        "output_folder": "rush123", 
        "vrt": True}
)
```
---

## Example 4: Exporting to Google Cloud Storage with Custom Dask Configuration

In this example, we export the NDVI results to **Google Cloud Storage** instead of the local disk.  
We also manually configure a custom Dask cluster.

---

### Steps

1. **Re-use the same `compute_ndvi()` function** from Example 1.

2. **Use the same Earth Engine image collection** setup and `dataset_params` from Example 1.

3. **Set up `export_params` for GCS**:
   - `flag = "GCS"` tells the exporter to use Google Cloud Storage.
   - Provide the path to your service account credentials (`gcs_credentials`).
   - Specify the bucket and folder to export to (`gcs_bucket`, `gcs_folder`).

4. **Configure Dask manually using `dask_kwargs`**:
   - Define number of workers, threads, and memory per worker.
   - Set `dask_mode = "custom"` to activate these settings.

5. **Run the pipeline** using the `run()` function with all the values above.

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