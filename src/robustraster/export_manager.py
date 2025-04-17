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
import dask
from dask import delayed

class ExportProcessor:
    def __init__(self, user_function_handler=None, **kwargs):
        """
        Initialize ExportProcessor.
        - If a UserFunctionHandler instance is provided, use it.
        - If not, create a new instance.
        
        :param user_function_handler: Optional existing UserFunctionHandler object.
        """
        self.user_function_handler = user_function_handler
        self.kwargs = kwargs

        self._time_value = None
        self._gcs_prefix = None
    
    def _generate_vrt(self, input_files: list, output_vrt: str):
        """Generate a VRT file from a list of GeoTIFF files."""
        if not input_files:
            print("No GeoTIFF files found to create VRT.")
            return
        
        vrt_dataset = gdal.BuildVRT(output_vrt, input_files)
        #vrt_dataset = gdal.BuildVRT(output_vrt, input_files)

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

    def _format_dataset2(self, result, ds):
        # Format dataset by renaming, transposing, and ensuring CRS.
        crs = ds.attrs.get('crs', None)
        ds_renamed = result.rename({'X': 'x', 'Y': 'y'})
        ds_transposed = ds_renamed.transpose('time', 'y', 'x').rio.write_crs(crs)
        return ds_transposed.sortby("y", ascending=False)

    def _get_block_time_str(self, ds_block):
        first_dim = list(ds_block.dims)[0]
        time_val = ds_block[first_dim].values
        if hasattr(time_val, "__len__"):
            time_val = time_val[0]
        return self._format_time_string(time_val)

    def _get_block_chunk_index(self, ds_block):
        # Optional — derive a unique index from coordinate values
        # This could be a hash or tuple of indices depending on your needs
        return hash(tuple(ds_block.coords[dim].values[0] for dim in ds_block.dims))

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

    def _export_to_gcs(self, stacked, chunk_index, time_str, gcs_credentials):
        """Export dataset chunk to Google Cloud Storage as a COG."""
        fs = gcsfs.GCSFileSystem(token=gcs_credentials)
        gcs_path = posixpath.join(self._gcs_prefix, f"chunk_{chunk_index}_time_{time_str}.tif")
        
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
        
        # Look into xarray.Dataset.from_dataframe
        # Look into loading it directly to Dask b/c of warning below.
        # UserWarning: Sending large graph of size 2.15 GiB.
        # this may cause some slowdown.
        # Consider loading the data with Dask directly
        # or using futures or delayed objects to embed the data into the graph without repetition.
        # See also https://docs.dask.org/en/stable/best-practices.html#load-data-with-dask for more information.
        df_input = ds.to_dataframe().reset_index()
        df_output = self.user_function_handler.user_function(df_input, *self.user_function_handler.args, **self.user_function_handler.kwargs)
        df_output = df_output.set_index(list(ds.dims))
        ds_output = df_output.to_xarray()
        
        ds_transposed = self._format_dataset(ds, ds_output)

        #time_val = ds['time'].values[0]
        #export_ds = ds_transposed.isel(time=0).drop_vars('time')

        self._compute_chunks_and_export(ds_transposed)
        return ds_output
    
    def _format_dataset(self, ds, ds_output):
        # Format dataset by renaming, transposing, and ensuring CRS.
        crs = ds.attrs.get('crs', None)
        ds_renamed = ds_output.rename({'X': 'x', 'Y': 'y'})
        ds_transposed = ds_renamed.transpose('y', 'x').rio.write_crs(crs)
        return ds_transposed.sortby("y", ascending=False)
    
    def _compute_chunks_and_export(self, ds_transposed):
        """Export a single block (already chunked by Dask) using the appropriate method."""

        flag = self.kwargs.get('flag', 'GTiff')
        #vrt_flag = self.kwargs.get('output_vrt', False)
        output_folder = self.kwargs.get('output_folder', 'tiles')
        gcs_credentials = self.kwargs.get('gcs_credentials', None)
        gcs_bucket = self.kwargs.get('gcs_bucket', 'buckets-of-fun')
        gcs_folder = self.kwargs.get('gcs_folder', None)

        # Determine time string from block content
        #time_str = self._get_block_time_str(ds_transposed)
        time_str = self._format_time_string(self._time_value)
        # Generate a chunk_index from block content if needed
        chunk_index = self._get_block_chunk_index(ds_transposed)

        stacked = self._convert_to_multiband(ds_transposed)

        if flag == "GTiff":
            self._export_to_geotiff(stacked, chunk_index, time_str, output_folder)

        elif flag == "GCS":
            #gcs_prefix = self._create_bucket_and_folder(gcs_credentials, gcs_bucket, gcs_folder)
            self._export_to_gcs(stacked, chunk_index, time_str, gcs_credentials)

    def _export_to_geotiff(self, stacked, chunk_index, time_str, output_folder):
        """Export dataset chunk as a GeoTIFF."""
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"chunk_{chunk_index}_time_{time_str}.tif")

        stacked.rio.to_raster(output_path, driver="GTiff")
        print(f"Exported: {output_path} with bands {list(stacked.band.values)}")
        
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

        ds = data_source.dataset
        chunks = self.kwargs.get('chunks', None)
        ds = self.user_function_handler._create_apply_chunk(ds, chunks)
        #template_xarray = self.user_function_handler._generate_template_xarray(ds)


        if self.kwargs.get("flag") == "GCS":
            self._gcs_prefix = self._create_bucket_and_folder(self.kwargs.get("gcs_credentials"), self.kwargs.get("gcs_bucket"), self.kwargs.get("gcs_folder", None))
            
        # Loop through each time step and export the chunks. The chunks in each time step will
        # parallel write.
        first_dim = list(ds.sizes)[0]
        for i in range(ds.sizes[first_dim]):
            self._time_value = ds['time'].values[i]
            dim_slice = ds.isel({first_dim: i}).drop_vars('time')
            template_xarray = self.user_function_handler._generate_template_xarray(dim_slice)
            result = xr.map_blocks(self._user_function_export_wrapper,
                                   dim_slice,
                                   template=template_xarray)
            result.compute()






        #result = xr.map_blocks(self._user_function_export_wrapper, 
        #                       ds, 
        #                       #args=(self.user_function_handler.user_function,),
        #                       template=template_xarray)
        #result.compute()
        

        #ds_transposed = self._format_dataset(result, ds)
        #self._compute_chunks_and_export(ds_transposed)
        #vrt_flag = self.kwargs.get("output_vrt", False)
        #if vrt_flag:
        #    self._generate_vrt_from_tifs()
        #return result
        
        
        """
        ds = data_source.dataset
        ds_block = ds.isel(time=0).chunk({'X': 512, 'Y': 512}).drop_vars("time")
        #template = self.user_function_handler._generate_template_xarray(ds_block)

        # Run the wrapper on just one block directly
        result = self._user_function_export_wrapper(ds_block)
        result.compute()
        """
        


# CHATGPT WAY TO PARALLELIZE EXPORT PROCESS USING DASK DELAYED OBJECT
# TAKES FOREVER...
"""
    def _compute_chunks(self, ds_transposed, export_func):
        first_dim_name = list(ds_transposed.dims)[0]
        export_tasks = []

        for time_index in ds_transposed[first_dim_name].values:
            ds_time_slice = ds_transposed.sel({first_dim_name: time_index})
            time_str = self._format_time_string(time_index)

            x_chunks = ds_time_slice.chunks["x"]
            y_chunks = ds_time_slice.chunks["y"]

            x_starts = [sum(x_chunks[:i]) for i in range(len(x_chunks))]
            y_starts = [sum(y_chunks[:i]) for i in range(len(y_chunks))]

            chunk_index = 0
            for x_start, x_size in zip(x_starts, x_chunks):
                for y_start, y_size in zip(y_starts, y_chunks):
                    chunk_dataset = ds_time_slice.isel(
                        x=slice(x_start, x_start + x_size),
                        y=slice(y_start, y_start + y_size)
                    )

                    stacked = self._convert_to_multiband(chunk_dataset)

                    # Wrap the export function in a Dask delayed task
                    task = delayed(export_func)(stacked, chunk_index, time_str)
                    export_tasks.append(task)
                    chunk_index += 1

        # Trigger all exports in parallel
        dask.compute(*export_tasks)
"""