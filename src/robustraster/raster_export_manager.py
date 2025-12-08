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
import math 

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

    def _generate_vrt_from_tifs(self, time_val):
        """Generate a VRT from all GeoTIFF files in a given directory."""
        output_folder = self.kwargs.get('output_folder', 'tiles')
        time_str = str(time_val).replace(":", "_").replace("-", "_").replace(" ", "_")
        vrt_file_name = f"{self._first_dim}_{time_str}.vrt"
        output_vrt_path = os.path.join(output_folder, vrt_file_name)
        tif_files = glob.glob(os.path.join(output_folder, f"*{self._first_dim}_{time_str}*.tif"))
        self._generate_vrt(tif_files, output_vrt_path)
    
    def _create_output_basename(self, ds_block):
        time_str = str(self._time_value).replace(":", "_").replace("-", "_").replace(" ", "_")
        chunk_hash = hash(tuple(ds_block.coords[dim].values[0] for dim in ds_block.dims))

        return f"chunk_{chunk_hash}_{self._first_dim}_{time_str}"

    def _convert_to_multiband(self, chunk_dataset):
        """Convert chunk dataset into a multi-band xarray DataArray."""
        stacked = chunk_dataset.to_array(dim="band")
        stacked = stacked.assign_coords(band=list(chunk_dataset.data_vars))

        if "spatial_ref" not in stacked.coords:
            print("CRS IS NOT SET! SET IT IN YOUR EARTH ENGINE CODE!")
        return stacked

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
        
        ds_final = self._format_back(ds_transposed)
        print("FINAL DATASET!")
        print()
        print()
        print(ds_final)
        return ds_final
    
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
    
    def _compute_chunks_and_export(self, ds_transposed):
        """Export a single block (already chunked by Dask) using the appropriate method."""
        stacked = self._convert_to_multiband(ds_transposed)

        if self.kwargs.get('file_format') == "GTiff" and not self.kwargs.get("export_to_gcs"):
            self._export_to_geotiff(stacked)

        elif self.kwargs.get('file_format') == "GTiff" and self.kwargs.get('export_to_gcs'):
            self._export_to_gcs(stacked)

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

        self._first_dim = list(data_source.dataset.dims)[0]
        ds = self.user_function_handler._create_apply_chunk(data_source.dataset)
        #ds = data_source.dataset
        # Generate template xarray
        self._template_xarray = self.user_function_handler._generate_template_xarray(ds)
        print("TEMPLATE!!!")
        print()
        print(self._template_xarray)
        if self.kwargs.get("export_to_gcs"):
            self._gcs_prefix = self._create_bucket_and_folder(self.kwargs.get("gcs_credentials"), self.kwargs.get("gcs_bucket"), self.kwargs.get("gcs_folder", None))
        
        print("DATASET")
        print(ds)
        result = xr.map_blocks(self._user_function_export_wrapper,
                                   ds,
                                   template=self._template_xarray)
        
        if self.kwargs.get("report") is True:
            with performance_report(filename="dask_report.html"):
                result.compute()
        else:
            result.compute()

        if self.kwargs.get('vrt'):
            self.export_vrt(data_source)

    def export_vrt(self, data_source: RasterDataset | EarthEngineDataset):
        for i, time_val in enumerate(data_source.dataset[self._first_dim].values):
            self._generate_vrt_from_tifs(time_val)
    
    ##### NEW CODE TESTING TILING ###
    # def _iter_tiles(self, ds, tile_size_px):
    #     Y = ds.sizes["Y"]
    #     X = ds.sizes["X"]

    #     tile_y = tile_size_px.get("y", Y)
    #     tile_x = tile_size_px.get("x", X)

    #     ny = math.ceil(Y / tile_y)
    #     nx = math.ceil(X / tile_x)

    #     for iy in range(ny):
    #         y0 = iy * tile_y
    #         y1 = min((iy + 1) * tile_y, Y)
    #         for ix in range(nx):
    #             x0 = ix * tile_x
    #             x1 = min((ix + 1) * tile_x, X)
    #             yield slice(y0, y1), slice(x0, x1)
    
    # def run_and_export_results(self, data_source):
    #     if not callable(self.user_function_handler.user_function):
    #         raise ValueError("The provided function must be callable.")

    #     self._first_dim = list(data_source.dataset.dims)[0]

    #     # Make full dataset lazy + chunked (your handler already does this)
    #     #full_ds = self.user_function_handler._create_apply_chunk(data_source.dataset)

    #     full_ds = data_source.dataset
    #     # Full template, then we slice it per tile
    #     template_full = self.user_function_handler._generate_template_xarray(full_ds)
    #     print("TEMPLATE!!!")
    #     print()
    #     print(template_full)
    #     # Configure GCS once
    #     if self.kwargs.get("export_to_gcs"):
    #         self._gcs_prefix = self._create_bucket_and_folder(
    #             self.kwargs.get("gcs_credentials"),
    #             self.kwargs.get("gcs_bucket"),
    #             self.kwargs.get("gcs_folder", None),
    #         )

    #     tile_size_px = self.kwargs.get("tile_size_px", {"Y": 10000, "X": 10000})

    #     for y_slc, x_slc in self._iter_tiles(full_ds, tile_size_px):
    #         ds_tile = full_ds.isel(Y=y_slc, X=x_slc)
    #         template_tile = template_full.isel(Y=y_slc, X=x_slc)
    #         #print(ds_tile)
    #         result_tile = xr.map_blocks(
    #             self._user_function_export_wrapper,
    #             ds_tile,
    #             template=template_tile,
    #         )

    #         result_tile.compute()

    #     if self.kwargs.get("vrt"):
    #         self.export_vrt(data_source)