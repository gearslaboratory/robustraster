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
import io

class VectorExportProcessor:
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
    
    def _create_output_basename(self, ds_block):
        time_str = str(self._time_value).replace(":", "_").replace("-", "_").replace(" ", "_")
        chunk_hash = hash(tuple(ds_block.coords[dim].values[0] for dim in ds_block.dims))

        return f"chunk_{chunk_hash}_{self._first_dim}_{time_str}"

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

    def _export_csv_to_gcs(self, df_output):
        """Export dataset chunk to Google Cloud Storage as a COG."""
        gcs_credentials = self.kwargs.get('gcs_credentials', None) 
        fs = gcsfs.GCSFileSystem(token=gcs_credentials)

        # Compose full GCS path
        gcs_path = posixpath.join(self._gcs_prefix, f"{self._output_basename}.csv")

        # Create in-memory CSV
        buffer = io.StringIO()
        df_output.to_csv(buffer, index=False)
        buffer.seek(0)

        # Upload to GCS
        with fs.open(gcs_path, "w") as f:
            f.write(buffer.read())

        print(f"[robustraster] Exported CSV to GCS: {gcs_path}")

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
    
    def _user_function_export_csv_wrapper(self, ds, *args):
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
            if self.kwargs.get('export_to_gcs'):
                self._export_csv_to_gcs(df_output)
            else:
                output_folder = self.kwargs.get('output_folder', 'tiles')
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, f"{self._output_basename}.csv")
                df_output.to_csv(output_path)
        
        return ds_output
    
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
        chunks = self.kwargs.get('chunks', None)
        ds = self.user_function_handler._create_apply_chunk(data_source.dataset, chunks)
        template_xarray = self.user_function_handler._generate_template_xarray(ds)

        if self.kwargs.get("export_to_gcs"):
            self._gcs_prefix = self._create_bucket_and_folder(self.kwargs.get("gcs_credentials"), self.kwargs.get("gcs_bucket"), self.kwargs.get("gcs_folder", None))

        if self.kwargs.get("file_format") == "CSV":
            result = xr.map_blocks(self._user_function_export_csv_wrapper,
                                    ds,
                                    template=template_xarray)
            result.compute()