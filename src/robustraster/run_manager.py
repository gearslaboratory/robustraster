# run.py

from .dataset_manager import RasterDataset, EarthEngineDataset
from .dask_cluster_manager import DaskClusterManager
from .export_manager import ExportProcessor
from .udf_manager import UserFunctionHandler
from .dask_plugins import EEPlugin

import pandas as pd
import ee
from typing import Any, Callable, Optional

def DatasetAdapterFactory(source, dataset, dataset_kwargs=None):
    """
    Smart dispatcher:
    - if source == "ee" → EarthEngineDataset
    - if source is a file path (string or list) → RasterDataset
    """
    if source == "ee":
        return EarthEngineDataset(image_collection=dataset, dataset_kwargs=dataset_kwargs)
    elif source == "local":
        return RasterDataset(file_path=dataset)
    else:
        raise ValueError("source must be 'ee' or a file path (str or list of str).")

def run(
    dataset: str | list[str] | ee.imagecollection.ImageCollection,
    source: str,
    dataset_kwargs: dict[str, Any] = None,
    user_function: Callable[[], pd.DataFrame] = None,
    user_function_args: tuple = (),
    user_function_kwargs: dict[str, Any] = None,
    tune_function: bool = False,
    export_kwargs: dict[str, Any] = None,
    dask_mode: str = "full",
    dask_kwargs: dict[str, Any] = None,
    hooks: Optional[dict[str, Callable[..., Any]]] = None
):
    """
    Run a user-defined function on a dataset.

    Parameters:

    dataset (str or list[str] or ee.ImageCollection): Can current accept the following:
        1. A path to a raster file stored locally.
        2. A list of raster files stored locally.
        3. A Google Earth Engine-dervived image collection.

    source (str): A string telling the function where this data came from. Acceptable
                  strings include "local" for locally stored rasters and "ee" for 
                  Earth Engine data.
    
    dataset_kwargs (str): A dictionary object (currently just for Google Earth Engine data) 
                          that contains the export parameters for Earth Engine data. kwargs include:
        1. "vector": Your area of interest or boundaries. This can be:
            a. A path to a local shapefile
            b. A path to a ZIP file containing your shapefile and it's dependent files.
            c. A GEOJSON file
            d. An ee.FeatureCollection() object
            e. An ee.Geometry() object
        2. "crs" (str): The coordinate system at which you want your dataset to be in. Required an EPSG code.
        3. "scale" (int): The spatial resolution of your dataset.
        4. "projection" (ee.Projection()): An ee.Projection() object. You can use "crs" and "projection" interchangably. 
                         "projection" will allow for transformations, if you need that sort of thing.
    user_function (Callable[[], pd.DataFrame]): Pass the function name here with this parameter. IMPORTANT:
        The function must take in a pandas Dataframe as input and must return a pandas Dataframe with
        the original x and y columns in tact.
    user_function_args (tuple): If your function requires arguments, pass them along here.
    user_function_kwargs (dict[str, Any]): If your function requires keyword arguments, pass them along here.
    tune_function (bool): If True, the run() function will do some test runs of your function on your
                          dataset to determine how to best do an optimized run of your function.
    export_kwargs (dict[str, Any]): A dictionary object that contains export parameters of your fully
                                    processed dataset (after running your function). This can be:
        1. "flag" (str): There are currently only two available options here - "GTiff" and "GCS". "GTiff"
                         will export the results to tiled geotiffs on your machine. If "GTiff" is selected,
                         there are unique keyword arguments:
                            a. "output_folder" (str): The folder path to store your geotiffs.
                            b. "export_vrt" (bool): If True, export VRT files with your geotiffs.
                         
                         "GCS" will store the tiled geotiffs to your Google Cloud Storage bucket. If "GCS" is 
                         selected, there are unique keyword arguments that go along with it: 
                            a. "gcs_credentials" (str): A path to your service account's JSON key that has the 
                                                        needed permissions to access your Google Cloud Storage
                                                        buckets.
                            b. "gcs_bucket" (str): The name of your Cloud Storage bucket. This will create the bucket
                                                   if it is not created already.
                            c. "gcs_folder" (str): The name of your Cloud Storage bucket.
        2. "chunks" (dict or str): If "tune_function" is True, a JSON file will be generated that contains tuning information
                                   needed to run your function on your dataset optimally. You can pass that file here as a path.
                                   Optionally, you can pass in your own custom chunk size if you are familiar with how xarray
                                   datasets work. 
    dask_mode (str): Dask is a Python library that will do parallelization of your function. "dask_mode" refers to the amount
                     of resources you would like to allocate to Dask to handle parallelization. There are three available options:
                            a. "full" is the default parameter. This will allocate all of your CPU and RAM to Dask.
                            b. "test" will allocate 1 CPU core to Dask.
                            c. "custom" allows the user to allocate their own threshold of resources to Dask. Use "dask_kwargs" to
                                pass your specified resources.
    dask_kwargs (dict[str, Any]): Only required if "dask_mode" is set to "custom".
        1. n_workers (int): Number of workers to create in the cluster. 
                           Overrides the default value determined by the mode.
        2. threads_per_worker (int): Number of threads per worker. 
                                     Default is 1.
        3. memory_limit (str): Memory limit per worker (e.g., "2GB"). 
                               Overrides the default value determined by the mode.
    
    """
    dataset_kwargs = dataset_kwargs or {}
    export_kwargs = export_kwargs or {}
    dask_kwargs = dask_kwargs or {}
    hooks = hooks or {}
    user_function_kwargs = user_function_kwargs or {}

    cluster_manager = None
    client = None

    try:
        # ========== HOOK: before_run ==========
        if "before_run" in hooks:
            hooks["before_run"]()

        # ========== Dask Setup ==========
        cluster_manager = DaskClusterManager()
        cluster_manager.create_cluster(mode=dask_mode, **dask_kwargs)
        client = cluster_manager.get_dask_client
        print(f"[robustraster] Dask cluster started: {client}")

        # Earth Engine Dask auth
        if source == 'ee':
            ee_plugin = EEPlugin()
            client.register_plugin(ee_plugin)
            print("[robustraster] Dask workers authenticated to Earth Engine.")

        # ========== Dataset Load ==========
        adapter = DatasetAdapterFactory(source, dataset, dataset_kwargs)
        data_source = adapter

        # ========== HOOK: after_dataset_loaded ==========
        if "after_dataset_loaded" in hooks:
            hooks["after_dataset_loaded"](data_source.dataset)

        # ========== User Function + Export ==========
        if user_function is not None:
            handler = UserFunctionHandler(
                user_function=user_function,
                *user_function_args,
                **user_function_kwargs
            )

            # Run tuning if requested
            if tune_function:
                print("[robustraster] Tuning user function...")
                handler.tune_user_function(data_source)

            processor = ExportProcessor(
                user_function_handler=handler,
                **export_kwargs
            )
            print("[robustraster] Running user function...")
            processor.run_and_export_results(data_source)

            client.close()
            client.shutdown()
        else:
            print("[robustraster] No user function provided. Skipping export.")

        # ========== HOOK: after_run ==========
        if "after_run" in hooks:
            hooks["after_run"](data_source.dataset)

    except Exception as e:
        print("[robustraster] ❌ ERROR during run():", str(e))
        raise  # Re-raise so users see the traceback unless you want silent failure

    finally:
        if client is not None:
            client.close()
            client.shutdown()
            print("[robustraster] Dask client closed.")
        if cluster_manager is not None:
            print("[robustraster] Dask cluster shut down.")