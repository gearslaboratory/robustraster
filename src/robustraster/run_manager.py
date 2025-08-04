# run.py

from .dataset_manager import RasterDataset, EarthEngineDataset
from .dask_cluster_manager import DaskClusterManager
from .raster_export_manager import RasterExportProcessor
from .vector_export_manager import VectorExportProcessor
from .udf_manager import UserFunctionHandler
from .dask_plugins import EEPlugin
from .hooks import preview_dataset_hook

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
        raise ValueError("Source must be 'ee' or a file path (str or list of str).")

def run(
    dataset: str | list[str] | ee.imagecollection.ImageCollection,
    source: str,
    dataset_kwargs: dict[str, Any] = None,
    user_function: Callable[[], pd.DataFrame] = None,
    user_function_args: tuple = (),
    user_function_kwargs: dict[str, Any] = None,
    preview_dataset: bool = False,
    tune_function: bool = False,
    max_iterations: int = None,
    export_kwargs: dict[str, Any] = None,
    dask_mode: str = "full",
    dask_kwargs: dict[str, Any] = None,
    hooks: Optional[dict[str, Callable[..., Any]]] = None
):
    """
    Main interface to run a user-defined function across geospatial raster data.

    This function handles:
    - Loading either local raster files or Google Earth Engine collections
    - Configuring a Dask cluster for distributed computation
    - Running your custom `pandas` function on each tile
    - Exporting the results to GeoTIFF or Google Cloud Storage
    - (Optional) Automatically tuning the chunk size for optimal performance

    Parameters
    ----------
    dataset : str | list[str] | ee.ImageCollection
        The input raster source.
        - For local rasters: a file path or list of file paths to `.tif` files
        - For Earth Engine: an `ee.ImageCollection` object

    source : str
        Must be either `"local"` or `"ee"`, indicating the data source type.

    dataset_kwargs : dict[str, Any], optional
        Required only for Earth Engine. Includes:
        - `geometry`: GeoJSON path, shapefile, zipped `.shp`, or native EE geometry/collection
        - `crs`: Coordinate reference system (e.g., "EPSG:4326")
        - `scale`: Spatial resolution (e.g., 30)
        - `projection`: An `ee.Projection` object

    user_function : Callable[[pd.DataFrame], pd.DataFrame]
        Your custom function to apply. Must accept a `DataFrame` and return a `DataFrame`
        with `x` and `y` columns preserved.

    user_function_args : tuple, optional
        Positional arguments passed to your function.
        See Example 3 in `02_quickstart.md`.

    user_function_kwargs : dict[str, Any], optional
        Keyword arguments passed to your function.
        See Example 4 in `02_quickstart.md`.

    tune_function : bool, optional
        Set to `True` to automatically determine optimal chunk size.
        See `05_tuning.md` for more.

    max_iterations : int, optional
        Max steps to take during tuning (if `tune_function=True`).

    export_kwargs : dict[str, Any]
        Configuration for export:
        - `"GTiff"`:
            - `output_folder`: local output path
            - `vrt`: whether to generate a VRT mosaic
        - `"GCS"`:
            - `gcs_credentials`, `gcs_bucket`, `gcs_folder`
            - `chunks`: manually specify Dask chunk sizes

    dask_mode : str, optional
        Controls cluster setup:
        - `"full"`: Use all local resources (default)
        - `"test"`: Single-threaded mode
        - `"custom"`: Use `dask_kwargs`

    dask_kwargs : dict[str, Any], optional
        Required if `dask_mode="custom"`. Includes:
        - `n_workers`: Number of workers
        - `threads_per_worker`: Threads per worker (default is 1)
        - `memory_limit`: RAM per worker (e.g., "2GB")

    hooks : dict[str, Callable], optional
        Functions triggered at key stages of execution:
        - `before_run`: Run before any processing begins
        - `after_dataset_loaded`: Run after loading the dataset
        - `after_run`: Run after export is complete

        One predefined hook is included: `preview_dataset_hook`, which previews a sample of
        the dataset before and after applying your function. This halts execution immediately after.

    Returns
    -------
    None
        Results are exported to disk or cloud depending on `export_params`.
    """
    dataset_kwargs = dataset_kwargs or {}
    export_kwargs = export_kwargs or {}
    dask_kwargs = dask_kwargs or {}
    hooks = hooks or {}
    user_function_kwargs = user_function_kwargs or {}

    cluster_manager = None
    client = None

    try:
        # ========== PREVIEW DATASET ===========
        if preview_dataset:
            adapter = DatasetAdapterFactory(source, dataset, dataset_kwargs)
            data_source = adapter

            preview_dataset_hook(data_source.dataset, user_function, *user_function_args, **user_function_kwargs)
            return
        
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

        # ===== Check for export keyword arguments ====
        gcs = export_kwargs.get("export_to_gcs", None)

        if gcs:
            required_gcs_keys = ["gcs_credentials", "gcs_bucket"]
            missing_keys = [k for k in required_gcs_keys if k not in export_kwargs]
            if missing_keys:
                raise ValueError(f"Missing required GCS export configuration: {', '.join(missing_keys)}")


        # ========== User Function + Export ==========
        if callable(user_function):
            handler = UserFunctionHandler(
                user_function,
                *user_function_args,
                **user_function_kwargs
            )

            # Run tuning if requested
            if tune_function or max_iterations:
                print("[robustraster] Tuning user function...")
                handler.tune_user_function(data_source, max_iterations)
            
            if "mode" not in export_kwargs:
                raise ValueError("Missing required export configuration: 'mode'")
        
            mode = export_kwargs["mode"]

            if mode == "raster":
                processor = RasterExportProcessor(
                    user_function_handler=handler,
                    **export_kwargs
                )
            elif mode == "vector":
                processor = VectorExportProcessor(
                    user_function_handler=handler,
                    **export_kwargs
                )
            print("[robustraster] Running user function...")
            processor.run_and_export_results(data_source)

            client.close()
            client.shutdown()
        else:
            raise ValueError("No user function was specified or user function is not callable! Please provide a function that accepts and returns a pandas DataFrame.")

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