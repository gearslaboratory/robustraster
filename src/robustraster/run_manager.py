from .dataset_manager import RasterDataset, EarthEngineDataset
from .dask_cluster_manager import DaskClusterManager
from .dask_docker_cluster_manager import DDClusterManager  # NEW: Docker-based cluster manager
#from .raster_export_manager import RasterExportProcessor
from .raster_export_manager_batched import RasterExportProcessor
from .vector_export_manager import VectorExportProcessor
from .udf_manager import UserFunctionHandler
from .dask_plugins import EEPlugin
from .hooks import preview_dataset_hook
from .ee_grid_tiles import ee_covering_grid_tiles, clip_tiles_to_aoi
from .ee_grid_tiles import iter_tiles_from_fc
from .vrt_export_manager import export_vrt
from distributed import Semaphore
import ee
from typing import Any, Callable, Optional, Union
from dask import config
from pathlib import Path
import json
import time

def DatasetAdapterFactory(source, dataset, dataset_config=None, *, chunks=None):
    """
    Smart dispatcher:
    - if source == "ee" → EarthEngineDataset
    - if source is a file path (string or list) → RasterDataset
    """
    if source == "ee":
        return EarthEngineDataset(image_collection=dataset, dataset_config=dataset_config)
    elif source == "local":
        if chunks is not None:
            dataset_config = dict(dataset_config)
            dataset_config.setdefault("chunks", chunks)
        return RasterDataset(file_path=dataset, dataset_config=dataset_config)
    else:
        raise ValueError("Source must be 'ee' or a file path (str or list of str).")

def report_dir(out_dir):
    d = Path(out_dir) / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d

def write_success(tile_id, out_dir):
    marker = report_dir(out_dir) / f"tile_{tile_id}.success.json"
    marker.write_text(json.dumps({
        "tile_id": tile_id,
        "status": "success",
        "timestamp": time.time()
    }))
    return marker

def write_failure(tile_id, out_dir, exc):
    marker = report_dir(out_dir) / f"tile_{tile_id}.failure.json"
    marker.write_text(json.dumps({
        "tile_id": tile_id,
        "status": "failure",
        "error": repr(exc),
        "timestamp": time.time()
    }))
    return marker

def run(
    dataset: str | list[str] | ee.imagecollection.ImageCollection,
    source: str,
    preview_dataset: bool = False,
    tune_function: bool = False,
    max_pixels_per_tile: int = None,
    dataset_config: dict[str, Any] = None,
    user_function_config: dict[str, Any] = None,
    function_tuning_config: Optional[dict[str, Any]] = None,
    export_config: dict[str, Any] = None,
    dask_mode: str = "full",
    dask_config: dict[str, Any] = None,
    hooks: Optional[dict[str, Callable[..., Any]]] = None,
    # --- NEW ---
    dask_use_docker: bool = False,
    dask_docker_image: Optional[str] = None,
    dask_docker_kwargs: Optional[dict[str, Any]] = None,
):
    """
    Main interface to run a user-defined function across geospatial raster data.
    """
    dataset_config = dataset_config or {}
    function_tuning_config = function_tuning_config or {}
    export_config = export_config or {}
    dask_config = dask_config or {}
    dask_docker_kwargs = dask_docker_kwargs or {}
    hooks = hooks or {}

    # user_function_config parameters extracted
    user_function = user_function_config.get("user_function")
    user_function_args = user_function_config.get("user_function_args")
    user_function_kwargs = user_function_config.get("user_function_kwargs")

    chunks = function_tuning_config.get("chunks", None)

    cluster_manager = None
    client = None

    try:
         # ========== PREVIEW DATASET ===========
        if preview_dataset:
            adapter = DatasetAdapterFactory(source, dataset, dataset_config, chunks=chunks)
            data_source = adapter

            preview_dataset_hook(data_source.dataset, user_function, *user_function_args, **user_function_kwargs)
            return

        # ========== HOOK: before_run ==========
        if "before_run" in hooks:
            hooks["before_run"]()

        # ========== Dask Setup ==========
        config.set({
            "distributed.worker.memory.target": False,
            "distributed.worker.memory.spill": False,
            "distributed.worker.memory.pause": 0.90,
            "distributed.worker.memory.terminate": 0.98,
        })
        if dask_use_docker:
            if not dask_docker_image:
                raise ValueError("dask_docker_image is required when dask_use_docker=True")
            cluster_manager = DDClusterManager(docker_image=dask_docker_image, **dask_docker_kwargs)
            cluster_manager.create_cluster(mode=dask_mode, **dask_config)
            client = cluster_manager.get_dask_client
            print(f"[robustraster] Docker Dask cluster started: {client}")
        else:
            cluster_manager = DaskClusterManager()
            cluster_manager.create_cluster(mode=dask_mode, **dask_config)
            client = cluster_manager.get_dask_client
            print(f"[robustraster] Dask cluster started: {client}")

        # ========== Earth Engine Dask auth + concurrency settings ============
        ee_semaphore = None
        # Earth Engine Dask auth
        if source == 'ee':
            ee_plugin = EEPlugin()
            client.register_plugin(ee_plugin)
            print("[robustraster] Dask workers authenticated to Earth Engine.")

            # ------------------------------------------------------------------
            # Global semaphore to limit Earth Engine concurrency across workers
            # ------------------------------------------------------------------
            #ee_max_concurrency = dask_config.get("ee_max_concurrency", 3)  # default 3
            #ee_semaphore = Semaphore(name="ee_global", max_leases=int(ee_max_concurrency))

        # ===== Check for export keyword arguments ====
        gcs = export_config.get("export_to_gcs", None)

        if gcs:
            required_gcs_keys = ["gcs_credentials", "gcs_bucket"]
            missing_keys = [k for k in required_gcs_keys if k not in export_config]
            if missing_keys:
                raise ValueError(f"Missing required GCS export configuration: {', '.join(missing_keys)}")

        # ========== User Function ==========
        if callable(user_function) and tune_function:
            # No tiling necessary! As the tuning code will only grab a slice of the data at a time.
            # ONLY WORKS FOR EE I THINK!
            data_source = DatasetAdapterFactory(source, dataset, dataset_config, chunks=chunks)
            function_tuning_config.pop
            handler = UserFunctionHandler(
                **user_function_config,
                **function_tuning_config
            )

            print("[robustraster] Tuning user function...")
            handler.tune_user_function(data_source)
    
        elif callable(user_function) and not tune_function:
            handler = UserFunctionHandler(
                    **user_function_config,
                    **function_tuning_config
                )
        else:
            raise ValueError("No user function was specified or user function is not callable! Please provide a function that accepts and returns a pandas DataFrame.")

        # ========= Export ==========
        if "mode" not in export_config:
            raise ValueError("Missing required export configuration: 'mode'")

        mode = export_config["mode"]

        if mode == "raster":
            processor = RasterExportProcessor(
                user_function_handler=handler,
                ee_semaphone=ee_semaphore,
                **export_config
            )
        elif mode == "vector":
            processor = VectorExportProcessor(
                user_function_handler=handler,
                **export_config
            )
        # ========== Dataset Load + Run ==========
        if source == "ee" and max_pixels_per_tile is not None:
            # Required tiling inputs
            crs = dataset_config.get("crs")
            scale = dataset_config.get("scale")
            aoi = dataset_config.get("geometry")

            if crs is None or scale is None or aoi is None:
                raise ValueError(
                    "max_pixels_per_tile requires dataset_config['geometry'], "
                    "dataset_config['crs'], and dataset_config['scale']."
                )

            # Build server-side tile collection
            tiles_fc = ee_covering_grid_tiles(
                aoi=aoi,
                crs=crs,
                scale=scale,
                max_pixels_per_tile=int(max_pixels_per_tile),
            )

            clipped_tiles = clip_tiles_to_aoi(tiles_fc, aoi)
            print("[robustraster] AOI tiling enabled. Streaming tiles in batches...")

            total_tiles = tiles_fc.size().getInfo()

            out_root = export_config.get("output_folder", "tiles")

            # Process tiles sequentially, but retrieve geometries in batches
            for tile_i, tile_geom in enumerate(iter_tiles_from_fc(clipped_tiles, batch_size=200), start=1):
                tile_dataset_config = dict(dataset_config)
                tile_dataset_config["geometry"] = tile_geom

                print(f"[robustraster] Processing tile {tile_i} of {total_tiles}")

                # Resume logic: skip tiles that already succeeded
                success_marker = report_dir(out_root) / f"tile_{tile_i}.success.json"
                if success_marker.exists():
                    print(f"[robustraster] ✅ Tile {tile_i} already succeeded; skipping.")
                    continue

                data_source = DatasetAdapterFactory(source, dataset, tile_dataset_config, chunks=chunks)
                
                print("[robustraster] Running user function...")
                processor._tile_id = tile_i

                try:
                    #if tile_i == 2:
                    #    raise RuntimeError("TEST: simulated tile failure")
                    
                    processor.run_and_export_results(data_source)
                    write_success(tile_i, out_root)
                except Exception as e:
                    write_failure(tile_i, out_root, e)
                    raise
            
            client.close()
            client.shutdown()

            if export_config.get("vrt"):
                export_vrt(data_source, out_root)

        else:
            out_root = export_config.get("output_folder", "tiles")

            # existing behavior
            data_source = DatasetAdapterFactory(source, dataset, dataset_config, chunks=chunks)

            if "after_dataset_loaded" in hooks:
                hooks["after_dataset_loaded"](data_source.dataset)

            report_dir(out_root)
            print("[robustraster] Running user function...")
            processor.run_and_export_results(data_source)

            client.close()
            client.shutdown()

            if export_config.get("vrt"):
                export_vrt(data_source, out_root)

        # ========== HOOK: after_run ==========
        if "after_run" in hooks:
            hooks["after_run"](data_source.dataset)

    except Exception as e:
        print(f"[robustraster] ❌ {type(e).__name__} during run():", str(e))
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