# run.py

from .dataset_manager import RasterDataset, EarthEngineDataset
from .dask_cluster_manager import DaskClusterManager
from .export_manager import ExportProcessor
from .udf_manager import UserFunctionHandler
from .dask_plugins import EEPlugin


def DatasetAdapterFactory(source, dataset, dataset_params=None):
    """
    Smart dispatcher:
    - if source == "ee" → EarthEngineDataset
    - if source is a file path (string or list) → RasterDataset
    """
    if source == "ee":
        return EarthEngineDataset(image_collection=dataset, dataset_params=dataset_params)
    elif source == "local":
        return RasterDataset(file_path=dataset)
    else:
        raise ValueError("source must be 'ee' or a file path (str or list of str).")

def run(
    dataset,
    source,
    dataset_params=None,
    user_function=None,
    user_function_args=(),
    user_function_kwargs=None,
    tune_function=False,
    export_params=None,
    dask_mode="full",
    dask_kwargs=None,
    hooks=None
):
    dataset_params = dataset_params or {}
    export_params = export_params or {}
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
        adapter = DatasetAdapterFactory(source, dataset, dataset_params)
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
                **export_params
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