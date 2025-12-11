from .dataset_manager import RasterDataset, EarthEngineDataset
from .dask_cluster_manager import DaskClusterManager
from .dask_docker_cluster_manager import DDClusterManager  # NEW: Docker-based cluster manager
from .raster_export_manager import RasterExportProcessor
from .vector_export_manager import VectorExportProcessor
from .udf_manager import UserFunctionHandler
from .dask_plugins import EEPlugin
from .hooks import preview_dataset_hook

import pandas as pd
import ee
from typing import Any, Callable, Optional, Union

def DatasetAdapterFactory(source, dataset, dataset_kwargs=None):
    """
    Smart dispatcher:
    - if source == "ee" → EarthEngineDataset
    - if source is a file path (string or list) → RasterDataset
    """
    if source == "ee":
        return EarthEngineDataset(image_collection=dataset, dataset_kwargs=dataset_kwargs)
    elif source == "local":
        return RasterDataset(file_path=dataset, dataset_kwargs=dataset_kwargs)
    else:
        raise ValueError("Source must be 'ee' or a file path (str or list of str).")

def run(
    dataset: str | list[str] | ee.imagecollection.ImageCollection,
    source: str,
    dataset_kwargs: dict[str, Any] = None,
    user_function_params: dict[str, Any] = None,
    #user_function: Callable[[], pd.DataFrame] = None,
    #user_function_args: tuple = (),
    #user_function_kwargs: dict[str, Any] = None,
    preview_dataset: bool = False,
    tune_function: bool = False,
    export_kwargs: dict[str, Any] = None,
    dask_mode: str = "full",
    dask_kwargs: dict[str, Any] = None,
    hooks: Optional[dict[str, Callable[..., Any]]] = None,
    # --- NEW ---
    dask_use_docker: bool = False,
    dask_docker_image: Optional[str] = None,
    dask_docker_kwargs: Optional[dict[str, Any]] = None,
):
    """
    Main interface to run a user-defined function across geospatial raster data.
    """
    dataset_kwargs = dataset_kwargs or {}
    export_kwargs = export_kwargs or {}
    dask_kwargs = dask_kwargs or {}
    dask_docker_kwargs = dask_docker_kwargs or {}
    hooks = hooks or {}

    user_function = user_function_params.get("user_function")
    user_function_args = user_function_params.get("user_function_args")
    user_function_kwargs = user_function_params.get("user_function_kwargs")

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
        if dask_use_docker:
            if not dask_docker_image:
                raise ValueError("dask_docker_image is required when dask_use_docker=True")
            cluster_manager = DDClusterManager(docker_image=dask_docker_image, **dask_docker_kwargs)
            cluster_manager.create_cluster(mode=dask_mode, **dask_kwargs)
            client = cluster_manager.get_dask_client
            print(f"[robustraster] Docker Dask cluster started: {client}")
        else:
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
                **user_function_params
            )

            # Run tuning if requested
            if tune_function:
                print("[robustraster] Tuning user function...")
                handler.tune_user_function(data_source)

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
            if source == 'local':
                processor.local_run_and_export_results(data_source)
            elif source == 'ee':
                processor.ee_run_and_export_results(data_source)

            client.close()
            client.shutdown()
        else:
            raise ValueError("No user function was specified or user function is not callable! Please provide a function that accepts and returns a pandas DataFrame.")

        # ========== HOOK: after_run ==========
        if "after_run" in hooks:
            hooks["after_run"](data_source.dataset)

    except Exception as e:
        print("[robustraster] ❌ ERROR during run():", str(e))
        raise

    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
            try:
                client.shutdown()
            except Exception:
                pass
            print("[robustraster] Dask client closed.")
        if cluster_manager is not None:
            if hasattr(cluster_manager, "shutdown"):
                try:
                    cluster_manager.shutdown()
                except Exception:
                    pass
            print("[robustraster] Dask cluster shut down.")