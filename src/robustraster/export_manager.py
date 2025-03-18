from osgeo import gdal
from rasterio.io import MemoryFile
from .dataset_manager import RasterDataset, EarthEngineDataset
from .udf_manager import UserDefinedFunction
from typing import Optional, Callable
import xarray as xr
import pandas as pd
import gcsfs
import os
import glob

class ExportProcessor:
    def __init__(self, user_function_handler=None):
        """
        Initialize ExportProcessor.
        - If a UserDefinedFunction instance is provided, use it.
        - If not, create a new instance.
        
        :param user_function_handler: Optional existing UserDefinedFunction object.
        """
        if user_function_handler is not None:
            self.user_function_handler = user_function_handler
        else:
            self.user_function_handler = UserDefinedFunction()  # Automatically create a new instance

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

    def _process_chunks(self, ds_transposed, flag, gcs_bucket=None, gcs_folder=None):
        """Iterate over dataset chunks and export accordingly."""
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

                if flag == "GEOTIFF":
                    self._export_to_geotiff(stacked, chunk_index, time_str)
                elif flag == "GCS":
                    self._export_to_gcs(stacked, chunk_index, time_str, gcs_bucket, gcs_folder)
        if flag == "GEOTIFF":
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

    def _export_to_geotiff(self, stacked, chunk_index, time_str):
        """Export dataset chunk as a GeoTIFF."""
        output_dir = "tiles"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"chunk_{chunk_index}_time_{time_str}.tif")

        stacked.rio.to_raster(output_path, driver="GTiff")
        print(f"Exported: {output_path} with bands {list(stacked.band.values)}")

    def _export_to_gcs(self, stacked, chunk_index, time_str, gcs_bucket, gcs_folder):
        """Export dataset chunk to Google Cloud Storage as a COG."""
        fs = gcsfs.GCSFileSystem()
        gcs_path = f"gcs://{gcs_bucket}/{gcs_folder}/chunk_{chunk_index}_time_{time_str}.tif"

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

    def run_and_export_results(self, data_source: RasterDataset | EarthEngineDataset, 
                       user_func: Callable[[], pd.DataFrame], flag: str, 
                       chunks: Optional[dict | str] = None,  *args, **kwargs):
        """Main function to apply user function and export results."""
        if not callable(user_func):
            raise ValueError("The provided function must be callable.")

        ds = data_source.dataset
        ds = self.user_function_handler._create_apply_chunk(ds, chunks)

        template_xarray = self.user_function_handler._generate_template_xarray(ds, user_func)

        result = xr.map_blocks(self.user_function_handler._user_function_wrapper, 
                               ds, 
                               args=(user_func,),
                               template=template_xarray)

        ds_transposed = self._format_dataset(result, ds)

        self._process_chunks(ds_transposed, flag)
        return result