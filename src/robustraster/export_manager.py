from osgeo import gdal
from rasterio.io import MemoryFile
from .dataset_manager import RasterDataset, EarthEngineDataset
from .udf_manager import UserFunctionHandler
from typing import Optional, Callable
from google.cloud import storage
import xarray as xr
import pandas as pd
import gcsfs
import os
import glob
import posixpath

class ExportProcessor:
    def __init__(self, user_function_handler=None):
        """
        Initialize ExportProcessor.
        - If a UserFunctionHandler instance is provided, use it.
        - If not, create a new instance.
        
        :param user_function_handler: Optional existing UserFunctionHandler object.
        """
        self.user_function_handler = user_function_handler

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

    def _generate_vrt_from_tifs(self, output_dir: str = "tiles"):
        """Generate a VRT from all GeoTIFF files in a given directory."""
        output_vrt = os.path.join(output_dir, "output.vrt")
        tif_files = glob.glob(os.path.join(output_dir, "*.tif"))
        self._generate_vrt(tif_files, output_vrt)

    def _format_dataset(self, result, ds):
        """Format dataset by renaming, transposing, and ensuring CRS."""
        crs = ds.attrs.get('crs', None)
        ds_renamed = result.rename({'X': 'x', 'Y': 'y'})
        ds_transposed = ds_renamed.transpose('time', 'y', 'x').rio.write_crs(crs)
        return ds_transposed.sortby("y", ascending=False)
    
    def _compute_chunks_and_export(self, ds_transposed, **kwargs):
        """Iterate over dataset chunks and export accordingly."""
        flag = kwargs.get('flag', 'GTiff')
        output_folder = kwargs.get('output_folder', 'tiles')
        gcs_credentials = kwargs.get('gcs_credentials', None)
        gcs_bucket = kwargs.get('gcs_bucket', 'buckets-of-fun')
        gcs_folder = kwargs.get('gcs_folder', None)

        if flag == "GTiff":
            self._compute_chunks(
                ds_transposed,
                lambda stacked, chunk_index, time_str: self._export_to_geotiff(stacked, chunk_index, time_str, output_folder)
            )
            self._generate_vrt_from_tifs()
        elif flag == "GCS":
            gcs_prefix = self._create_bucket_and_folder(gcs_credentials, gcs_bucket, gcs_folder)
            self._compute_chunks(
                ds_transposed,
                lambda stacked, chunk_index, time_str: self._export_to_gcs(
                    stacked, chunk_index, time_str, gcs_prefix, gcs_credentials
                )
            )

    def _compute_chunks(self, ds_transposed, export_func):
        first_dim_name = list(ds_transposed.dims)[0]

        for time_index in ds_transposed[first_dim_name].values:
            ds_time_slice = ds_transposed.sel({first_dim_name: time_index})
            time_str = self._format_time_string(time_index)

            chunk_size_x = ds_time_slice.chunks["x"][0]
            chunk_size_y = ds_time_slice.chunks["y"][0]

            for chunk_index, chunk in enumerate(ds_time_slice.chunk({"x": chunk_size_x, "y": chunk_size_y}).data_vars.items()):
                var_name, chunk_data = chunk
                chunk_dataset = ds_time_slice.isel(
                    x=slice(chunk_data.chunks[1][0]),
                    y=slice(chunk_data.chunks[0][0])
                )

                stacked = self._convert_to_multiband(chunk_dataset)
                export_func(stacked, chunk_index, time_str)

    def _compute_chunks_and_export_2(self, ds_transposed, **kwargs):
        #Iterate over dataset chunks and export accordingly.
        flag = kwargs.get('flag', 'GTiff')
        output_folder = kwargs.get('output_folder', 'tiles')
        gcs_credentials = kwargs.get('gcs_credentials', None)
        gcs_bucket = kwargs.get('gcs_bucket', 'buckets-of-fun')
        gcs_folder = kwargs.get('gcs_folder', None)

        if flag == "GTiff":
            first_dim_name = list(ds_transposed.dims)[0]

            for time_index in ds_transposed[first_dim_name].values:
                ds_time_slice = ds_transposed.sel({first_dim_name: time_index})
                time_str = self._format_time_string(time_index)

                chunk_size_x = ds_time_slice.chunks["x"][0]
                chunk_size_y = ds_time_slice.chunks["y"][0]

                for chunk_index, chunk in enumerate(ds_time_slice.chunk({"x": chunk_size_x, "y": chunk_size_y}).data_vars.items()):
                    var_name, chunk_data = chunk
                    chunk_dataset = ds_time_slice.isel(
                        x=slice(chunk_data.chunks[1][0]),
                        y=slice(chunk_data.chunks[0][0])
                    )

                    stacked = self._convert_to_multiband(chunk_dataset)
                    self._export_to_geotiff(stacked, chunk_index, time_str, output_folder)
            self._generate_vrt_from_tifs()     
        elif flag == "GCS":
            first_dim_name = list(ds_transposed.dims)[0]

            for time_index in ds_transposed[first_dim_name].values:
                ds_time_slice = ds_transposed.sel({first_dim_name: time_index})
                time_str = self._format_time_string(time_index)

                chunk_size_x = ds_time_slice.chunks["x"][0]
                chunk_size_y = ds_time_slice.chunks["y"][0]

                for chunk_index, chunk in enumerate(ds_time_slice.chunk({"x": chunk_size_x, "y": chunk_size_y}).data_vars.items()):
                    var_name, chunk_data = chunk
                    chunk_dataset = ds_time_slice.isel(
                        x=slice(chunk_data.chunks[1][0]),
                        y=slice(chunk_data.chunks[0][0])
                    )

                    stacked = self._convert_to_multiband(chunk_dataset)
                    self._export_to_gcs(stacked, chunk_index, time_str, gcs_credentials, gcs_bucket, gcs_folder)

        flag = kwargs.get('flag', 'GTiff')
        output_folder = kwargs.get('output_folder', 'tiles')
        gcs_credentials = kwargs.get('gcs_credentials', None)
        gcs_bucket = kwargs.get('gcs_bucket', 'buckets-of-fun')
        gcs_folder = kwargs.get('gcs_folder', None)

        first_dim_name = list(ds_transposed.dims)[0]

        for time_index in ds_transposed[first_dim_name].values:
            ds_time_slice = ds_transposed.sel({first_dim_name: time_index})
            time_str = self._format_time_string(time_index)

            chunk_size_x = ds_time_slice.chunks["x"][0]
            chunk_size_y = ds_time_slice.chunks["y"][0]

            for chunk_index, chunk in enumerate(ds_time_slice.chunk({"x": chunk_size_x, "y": chunk_size_y}).data_vars.items()):
                var_name, chunk_data = chunk
                chunk_dataset = ds_time_slice.isel(
                    x=slice(chunk_data.chunks[1][0]),
                    y=slice(chunk_data.chunks[0][0])
                )

                stacked = self._convert_to_multiband(chunk_dataset)

                if flag == "GTiff":
                    self._export_to_geotiff(stacked, chunk_index, time_str, output_folder)
                elif flag == "GCS":
                    self._export_to_gcs(stacked, chunk_index, time_str, gcs_credentials, gcs_bucket, gcs_folder)
        if flag == "GTiff":
            self._generate_vrt_from_tifs()

    def _format_time_string(self, time_index):
        """Convert time index to string for filenames."""
        return str(time_index).replace(":", "_").replace("-", "_").replace(" ", "_")

    def _convert_to_multiband(self, chunk_dataset):
        """Convert chunk dataset into a multi-band xarray DataArray."""
        stacked = chunk_dataset.to_array(dim="band")
        stacked = stacked.assign_coords(band=list(chunk_dataset.data_vars))

        if "spatial_ref" not in stacked.coords:
            print("CRS IS NOT SET! SET IT IN YOUR EARTH ENGINE CODE!")
        return stacked

    def _export_to_geotiff(self, stacked, chunk_index, time_str, output_folder):
        """Export dataset chunk as a GeoTIFF."""
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"chunk_{chunk_index}_time_{time_str}.tif")

        stacked.rio.to_raster(output_path, driver="GTiff")
        print(f"Exported: {output_path} with bands {list(stacked.band.values)}")

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

    def _export_to_gcs(self, stacked, chunk_index, time_str, gcs_prefix, gcs_credentials):
        """Export dataset chunk to Google Cloud Storage as a COG."""
        fs = gcsfs.GCSFileSystem(token=gcs_credentials)
        gcs_path = posixpath.join(gcs_prefix, f"chunk_{chunk_index}_time_{time_str}.tif")
        
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

    def run_and_export_results(self, data_source: RasterDataset | EarthEngineDataset, **kwargs):
        # Keyword arguments:
        # flag: str
        # chunks: Optional[dict | str]
        # output_folder: str
        # gcs_bucket: str
        # gcs_folder: str
        """Main function to apply user function and export results."""
        if not callable(self.user_function_handler.user_function):
            raise ValueError("The provided function must be callable.")

        ds = data_source.dataset
        chunks = kwargs.get('chunks', None)
        ds = self.user_function_handler._create_apply_chunk(ds, chunks)
        
        template_xarray = self.user_function_handler._generate_template_xarray(ds)

        result = xr.map_blocks(self.user_function_handler._user_function_wrapper, 
                               ds, 
                               args=(self.user_function_handler.user_function,),
                               template=template_xarray)

        ds_transposed = self._format_dataset(result, ds)

        self._compute_chunks_and_export(ds_transposed, **kwargs)
        return result