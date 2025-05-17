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
        return RasterDataset(file_path=source)
    else:
        raise ValueError("source must be 'ee' or a file path (str or list of str).")

def run(
    dataset,
    source,
    dataset_params=None,
    user_function=None,
    user_function_args=(),
    user_function_kwargs=None,
    export_params=None,
    dask_mode="full",
    dask_kwargs=None,
    hooks=None
):
    """
    Main pipeline function.

    Parameters:
    - source: 'ee' for Earth Engine OR file path (str or list of str) for local rasters
    - dataset_params: dict (only used if source='ee')
    - user_function: user function (must accept dataframe, must return dataframe)
    - export_params: dict of export options
    - dask_mode: Dask cluster mode ('full' or 'test')
    - dask_kwargs: optional custom Dask cluster settings
    - hooks: optional dict of hooks {before_run, after_dataset_loaded, before_export, after_run}
    """
    dataset_params = dataset_params or {}
    export_params = export_params or {}
    dask_kwargs = dask_kwargs or {}
    hooks = hooks or {}
    user_function_kwargs = user_function_kwargs or {}

    # ===============================
    # Call before_run hook
    # ===============================
    if "before_run" in hooks:
        hooks["before_run"]()

    # ===============================
    # Setup Dask cluster
    # ===============================
    cluster_manager = DaskClusterManager()
    cluster_manager.create_cluster(mode=dask_mode, **dask_kwargs)
    client = cluster_manager.get_dask_client
    print(f"[robustraster] Dask cluster started: {client}")

    if source == 'ee':
        ee_plugin = EEPlugin()
        client.register_plugin(ee_plugin)
        print("Dask workers authenticated via to Earth Engine...")
    # ===============================
    # Load dataset
    # ===============================
    adapter = DatasetAdapterFactory(source, dataset, dataset_params)
    data_source = adapter

    # ===============================
    # Call after_dataset_loaded hook
    # ===============================
    if "after_dataset_loaded" in hooks:
        hooks["after_dataset_loaded"](data_source.dataset)

    # ===============================
    # Apply user function + export
    # ===============================
    if user_function is not None:
        handler = UserFunctionHandler(
            user_function=user_function,
            *user_function_args,
            **user_function_kwargs
        )

        processor = ExportProcessor(
            user_function_handler=handler,
            **export_params
        )
        processor.run_and_export_results(data_source)
    else:
        print("[robustraster] No user function provided. Skipping export.")

    # ===============================
    # Call after_run hook
    # ===============================
    if "after_run" in hooks:
        hooks["after_run"](data_source.dataset)

    # ===============================
    # Shutdown Dask cluster
    # ===============================
    client.close()
    print("[robustraster] Dask cluster shut down.")
