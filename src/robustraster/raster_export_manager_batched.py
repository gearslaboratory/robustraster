from osgeo import gdal
import rasterio
from rasterio.io import MemoryFile
from .dataset_manager import RasterDataset, EarthEngineDataset
from google.cloud import storage
import xarray as xr
import gcsfs
import os
import glob
import posixpath
from dask.distributed import performance_report, print

import numpy as np
import pandas as pd
import datetime as dt

class RasterExportProcessor:
    def __init__(self, user_function_handler=None, **kwargs):
        """
        Initialize ExportProcessor.
        - If a UserFunctionHandler instance is provided, use it.
        - If not, create a new instance.
        
        :param user_function_handler: Optional existing UserFunctionHandler object.
        """
        self.user_function_handler = user_function_handler
        self.kwargs = kwargs

        self._first_dim = None
        self._time_value = None
        self._output_basename = None
        self._gcs_prefix = None

        self._template_xarray = None
        self._data_source = None

        # For the naming scheme of my output files
        self._tile_id = None

    def _iter_chunk_batches(
        self,
        ds: xr.Dataset,
        split_dim: str,
        target_blocks_per_run: int,
    ):
        """Yield (indexers, tag) batches aligned to *existing* Dask chunks.

        Parameters
        ----------
        ds:
            Chunked xarray Dataset (Dask-backed).
        split_dim:
            Dimension to batch along (e.g., "X", "Y", "time").
        target_blocks_per_run:
            Approximate upper bound on number of *chunks* (blocks) processed per run.
            Task count is typically a multiple of block count, so this keeps graphs manageable.
        """
        if ds.chunks is None:
            raise ValueError("Dataset is not chunked; cannot batch by chunk groups.")

        if split_dim not in ds.dims:
            raise ValueError(f"split_dim={split_dim!r} not in ds.dims={list(ds.dims)}")

        # Find a representative dask-backed variable
        rep_var = None
        for v in ds.data_vars:
            a = ds[v].data
            if hasattr(a, "numblocks"):
                rep_var = v
                break
        if rep_var is None:
            raise ValueError("No dask-backed variables found; nothing to batch.")

        rep = ds[rep_var]
        numblocks_by_dim = dict(zip(rep.dims, rep.data.numblocks))

        # blocks_per_run ≈ (split_chunks_in_run) * product(other_dim_chunks)
        other_blocks = 1
        for d, nb in numblocks_by_dim.items():
            if d != split_dim:
                other_blocks *= int(nb)

        k = max(1, int(target_blocks_per_run) // max(1, other_blocks))

        split_chunks = ds.chunks[split_dim]
        if split_chunks is None:
            raise ValueError(f"No chunking info available for split_dim={split_dim!r}")

        # cumulative element indices at chunk boundaries
        boundaries = [0]
        for sz in split_chunks:
            boundaries.append(boundaries[-1] + int(sz))

        n_split_chunks = len(split_chunks)
        for ci0 in range(0, n_split_chunks, k):
            ci1 = min(n_split_chunks, ci0 + k)
            i0 = boundaries[ci0]
            i1 = boundaries[ci1]
            indexers = {split_dim: slice(i0, i1)}
            tag = f"{split_dim}_{ci0:06d}_{ci1:06d}"
            yield indexers, tag

    def _pick_default_split_dim(self, ds: xr.Dataset) -> str:
        """Choose a reasonable batching dimension if the user doesn't specify one."""
        for cand in ("X", "x", "Y", "y"):
            if cand in ds.dims:
                return cand
        # Fall back: choose the last dimension (often spatial)
        return list(ds.dims)[-1]
    
    def run_and_export_results(self, data_source: RasterDataset | EarthEngineDataset):
        # Keyword arguments:
        # flag: str
        # chunks: Optional[dict | str]
        # output_folder: str
        # gcs_bucket: str
        # gcs_folder: str
        """Main function to apply user function and export results."""
        
        if not callable(self.user_function_handler.user_function):
            raise ValueError("The provided function must be callable.")
        
        # We do this check to ensure _user_function_export_wrapper formats the data
        # properly for the map_blocks return statement. 
        if isinstance(data_source , RasterDataset):
            self._data_source = "local"
        elif isinstance(data_source, EarthEngineDataset):
            self._data_source = "ee"

        self._first_dim = list(data_source.dataset.dims)[0]
        ds = self.user_function_handler._create_apply_chunk(data_source.dataset)
        
        # Generate template xarray (for non-batched runs)
        self._template_xarray = self.user_function_handler._generate_template_xarray(ds)

        if self.kwargs.get("export_to_gcs"):
            self._gcs_prefix = self._create_bucket_and_folder(
                self.kwargs.get("gcs_credentials"),
                self.kwargs.get("gcs_bucket"),
                self.kwargs.get("gcs_folder", None),
            )

        # Optional batching to avoid enormous Dask graphs when chunk sizes must stay small
        target_blocks_per_run = self.kwargs.get("target_blocks_per_run", None)
        split_dim = self.kwargs.get("batch_split_dim", None)

        do_batching = (target_blocks_per_run is not None) or (split_dim is not None)

        if do_batching:
            if target_blocks_per_run is None:
                # reasonable default if user only provided split_dim
                target_blocks_per_run = 20000
            target_blocks_per_run = int(target_blocks_per_run)

            if split_dim is None:
                split_dim = self._pick_default_split_dim(ds)

            for indexers, tag in self._iter_chunk_batches(
                ds, split_dim=split_dim, target_blocks_per_run=target_blocks_per_run
            ):
                ds_part = ds.isel(indexers)

                template_part = self.user_function_handler._generate_template_xarray(ds_part)

                result = xr.map_blocks(
                    self._user_function_export_wrapper,
                    ds_part,
                    template=template_part,
                )

                if self.kwargs.get("report") is True:
                    # Avoid overwriting the same report file per batch
                    with performance_report(filename=f"dask_report_{tag}.html"):
                        result.compute()
                else:
                    result.compute()
        else:
            result = xr.map_blocks(
                self._user_function_export_wrapper,
                ds,
                template=self._template_xarray,
            )

            if self.kwargs.get("report") is True:
                with performance_report(filename=f"dask_report_tile_{self._tile_id}.html"):
                    result.compute()
            else:
                result.compute()

        if self.kwargs.get("vrt"):
            self._export_vrt(data_source)
    
    def _create_bucket_and_folder(self, gcs_credentials, gcs_bucket, gcs_folder):
        # Initialize GCS client
        storage_client = storage.Client.from_service_account_json(gcs_credentials)

        # Check if bucket exists, create if not
        try:
            bucket = storage_client.get_bucket(gcs_bucket)
            print(f"Bucket {gcs_bucket} already exists.")
        except Exception:
            bucket = storage_client.create_bucket(gcs_bucket)
            print(f"Created bucket: {gcs_bucket}")

        # Handle optional folder creation
        if gcs_folder:
            folder_blob = f"{gcs_folder}/"
            if not any(blob.name.startswith(folder_blob) for blob in storage_client.list_blobs(gcs_bucket, prefix=folder_blob, max_results=1)):
                blob = bucket.blob(folder_blob)
                blob.upload_from_string('')
                print(f"Created folder: {gcs_folder}")

        # Create GCS path
        if gcs_folder:
            gcs_prefix = f"gcs://{gcs_bucket}/{gcs_folder}"
        else:
            gcs_prefix = f"gcs://{gcs_bucket}"
        return gcs_prefix
    
    def _user_function_export_wrapper(self, ds, *args):
        """
        Wrapper function that applies either `tune_user_function` or `apply_user_function`.
        to the user's dataset. This will convert the user's dataset to a pandas DataFrame
        first before running the user's function.
        
        Parameters:
        - user_func: the user-defined function to apply.
        - args: positional arguments to pass to the function.
        - kwargs: keyword arguments to pass to the function.
        
        Returns:
        - result: the result of applying the function to the dataframe.
        """

        df_input = ds.to_dataframe().reset_index()
        df_output = self.user_function_handler.user_function(df_input, *self.user_function_handler.args, **self.user_function_handler.kwargs)
        df_output = df_output.set_index(list(ds.dims))
        ds_output = df_output.to_xarray()
        
        ds_transposed = self._format_dataset(ds, ds_output)
        for i, time_val in enumerate(ds_transposed[self._first_dim].values):
            self._time_value = time_val
            slice_2d = ds_transposed.isel({self._first_dim: i})
            self._output_basename = self._create_output_basename(slice_2d)
            self._compute_chunks_and_export(slice_2d)
        
        if self._data_source == "local":
            ds_final = self._format_back(ds_transposed)
            return ds_final

        return ds_output
    
    def _format_dataset(self, ds, ds_output):
        # Format dataset by renaming, transposing, and ensuring CRS.
        crs = ds.attrs.get('crs', None)
         # Rename dimensions only if needed
        rename_dims = {}
        if 'X' in ds_output.dims:
            rename_dims['X'] = 'x'
        if 'Y' in ds_output.dims:
            rename_dims['Y'] = 'y'
        ds_renamed = ds_output.rename(rename_dims)
        ds_transposed = ds_renamed.transpose(self._first_dim, 'y', 'x').rio.write_crs(crs)
        return ds_transposed.sortby("y", ascending=False)
    
    def _format_back(self, ds_output):
        rename_dims = {}
        if 'x' in ds_output.dims:
            rename_dims['x'] = 'X'
        if 'y' in ds_output.dims:
            rename_dims['y'] = 'Y'
        ds_renamed = ds_output.rename(rename_dims)
        return ds_renamed
    
    def _safe_time(self, v):
        if isinstance(v, np.datetime64):
            v = pd.to_datetime(v).to_pydatetime()
        if isinstance(v, dt.date) and not isinstance(v, dt.datetime):
            v = dt.datetime(v.year, v.month, v.day)
        return str(v).replace(":", "").replace("-", "").replace(" ", "T").split(".")[0]

    def _create_output_basename(self, ds_block):
        """
        Short deterministic basename:
        tile_<tile_id>__<first_dim>_<time>
        """
        chunk_tag = hash(tuple(ds_block.coords[dim].values[0] for dim in ds_block.dims))

        # tile id (if not set, fall back to "chunk")
        #if getattr(self, "_tile_id", None) is not None:
        #    self._tile_id = f"tile_{int(self._tile_id):03d}"
        #else:
        #    self._tile_id = "tile_unknown"
    
        time_tag = self._safe_time(self._time_value)

        return f"{chunk_tag}_tile_{self._tile_id}__{self._first_dim}_{time_tag}"

    def _compute_chunks_and_export(self, ds_transposed):
        """Export a single block (already chunked by Dask) using the appropriate method."""
        stacked = self._convert_to_multiband(ds_transposed)

        if self.kwargs.get('file_format') == "GTiff" and not self.kwargs.get("export_to_gcs"):
            self._export_to_geotiff(stacked)

        elif self.kwargs.get('file_format') == "GTiff" and self.kwargs.get('export_to_gcs'):
            self._export_to_gcs(stacked)
    
    def _convert_to_multiband(self, chunk_dataset):
        """Convert chunk dataset into a multi-band xarray DataArray."""
        stacked = chunk_dataset.to_array(dim="band")
        stacked = stacked.assign_coords(band=list(chunk_dataset.data_vars))

        if "spatial_ref" not in stacked.coords:
            print("CRS IS NOT SET! SET IT IN YOUR EARTH ENGINE CODE!")
        return stacked
    
    def _export_to_geotiff(self, stacked):
        output_folder = self.kwargs.get('output_folder', 'tiles')
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"{self._output_basename}.tif")

        band_names = list(stacked.band.values)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=stacked.rio.height,
            width=stacked.rio.width,
            count=len(band_names),
            dtype=str(stacked.dtype),
            crs=stacked.rio.crs,
            transform=stacked.rio.transform(),
        ) as dst:
            for idx, name in enumerate(band_names, start=1):
                dst.write(stacked[idx - 1].values, indexes=idx)
                dst.set_band_description(idx, str(name))
                dst.update_tags(
                x_min=float(stacked.coords["x"].values[0]),
                x_max=float(stacked.coords["x"].values[-1]),
                y_min=float(stacked.coords["y"].values[-1]),
                y_max=float(stacked.coords["y"].values[0]),
                )

        print(f"Exported: {output_path} with bands {band_names}")
    
    def _export_to_gcs(self, stacked):
        """Export dataset chunk to Google Cloud Storage as a COG."""
        gcs_credentials = self.kwargs.get('gcs_credentials', None) 
        fs = gcsfs.GCSFileSystem(token=gcs_credentials)
        gcs_path = posixpath.join(self._gcs_prefix, f"{self._output_basename}.tif")
        
        with MemoryFile() as memfile:
            with memfile.open(
                driver="COG",
                width=stacked.rio.width,
                height=stacked.rio.height,
                count=len(stacked.band),
                dtype=stacked.dtype,
                crs=stacked.rio.crs,
                transform=stacked.rio.transform(),
            ) as dataset:
                dataset.write(stacked.values)

            with fs.open(gcs_path, "wb") as f:
                f.write(memfile.read())

        print(f"Exported to GCS: {gcs_path} with bands {list(stacked.band.values)}")
    
    def _export_vrt(self, data_source: RasterDataset | EarthEngineDataset):
        for i, time_val in enumerate(data_source.dataset[self._first_dim].values):
            print(i)
            self._generate_vrt_from_tifs(time_val)

    def _generate_vrt_from_tifs(self, time_val):
        """Generate a VRT from all GeoTIFF files in a given directory."""
        output_folder = self.kwargs.get('output_folder', 'tiles')
        #time_str = str(time_val).replace(":", "_").replace("-", "_").replace(" ", "_")
        time_tag = self._safe_time(time_val)
        file_basename = f"tile_{self._tile_id}__{self._first_dim}_{time_tag}"
        output_vrt_path = os.path.join(output_folder, f"{file_basename}.vrt")
        tif_files = glob.glob(os.path.join(output_folder, f"*{file_basename}.tif"))
        self._generate_vrt(tif_files, output_vrt_path)
    
    def _generate_vrt(self, input_files: list, output_vrt: str):
        """Generate a VRT file from a list of GeoTIFF files."""
        if not input_files:
            print("No GeoTIFF files found to create VRT.")
            return
        
        vrt_dataset = gdal.BuildVRT(output_vrt, input_files)

        if vrt_dataset:
            vrt_dataset.FlushCache()  # Save changes
            vrt_dataset = None  # Close dataset
            print(f"VRT file created successfully: {output_vrt}")
        else:
            print("Failed to create VRT file.")