import os
import numpy as np
import rasterio
import pytest
import ee
import xarray as xr
from robustraster.dataset_manager import RasterDataset, EarthEngineDataset


@pytest.fixture
def temp_raster_file():
    """Create a temporary raster file for testing."""
    temp_file = "temp_raster.tif"
    with rasterio.open(temp_file, 'w', driver='GTiff', width=10, height=10, count=1, dtype='uint8') as dst:
        dst.write_band(1, np.zeros((10, 10), dtype='uint8'))
    yield temp_file
    os.remove(temp_file)

@pytest.fixture
def temp_invalid_file():
    """Create a temporary invalid file."""
    temp_file = "invalid_file.txt"
    with open(temp_file, "w") as file:
        pass
    yield temp_file
    os.remove(temp_file)

def test_read_data(temp_raster_file):
    """Test reading in a raster file from a local machine."""
    reader = RasterDataset(temp_raster_file)
    xarray_data = reader._xarray_data
    assert xarray_data is not None
    assert isinstance(xarray_data, xr.Dataset)

def test_read_data_file_not_found():
    """Test if the raster file is not found when running read_data."""
    non_existing_file = "non_existing_file.tif"
    with pytest.raises(rasterio.errors.RasterioIOError, match="No such file or directory"):
        RasterDataset(non_existing_file)

def test_read_data_invalid_file(temp_invalid_file):
    """Test if the raster file found is invalid."""
    with pytest.raises(rasterio.errors.RasterioIOError, match="not recognized as being in a supported file format"):
        RasterDataset(temp_invalid_file)

@pytest.fixture
def setup_earth_engine():
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    print(f"CREDENTIALS PATH: {credentials_path}")
    if credentials_path and os.path.exists(credentials_path):
        ee.Initialize(ee.ServiceAccountCredentials(None, credentials_path),
                      opt_url='https://earthengine-highvolume.googleapis.com')
    else:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set correctly!")

def _construct_test_fc_object(geometry):
    feature = ee.Feature(geometry)
    featureCollection = ee.FeatureCollection([feature])
    
    return featureCollection

def test_construct_ee_collection_no_collection(setup_earth_engine):
    """Test if no ImageCollection is passed."""
    featureCollection = _construct_test_fc_object(geometry=ee.Geometry.Point(-122.082, 37.42))
    parameters = {
        'start_date': '2021-01-01',
        'end_date': '2021-01-31',
        'vector_path': featureCollection
    }
    with pytest.raises(ee.EEException, match="Earth Engine collection was not provided."):
        EarthEngineDataset(parameters)

def test_construct_ee_collection_invalid_collection_type(setup_earth_engine):
    """Test to ensure the ImageCollection string is a valid collection type."""
    featureCollection = _construct_test_fc_object(geometry=ee.Geometry.Point(-122.082, 37.42))
    parameters = {
        'collection': 500,
        'start_date': '1992-10-05',
        'end_date': '1993-03-31',
        'vector_path': featureCollection
    }
    with pytest.raises(ee.EEException, match="Unrecognized argument type"):
        EarthEngineDataset(parameters)

def test_map_function_applied(setup_earth_engine):
    """Test if the map_function is applied to the ImageCollection."""
    def map_function(image):
        return image.add(10)
    
    def reproject_geometry(geom, target_image):
        projection = target_image.projection()
        return geom.transform(projection)

    geometry = reproject_geometry(ee.Geometry.Rectangle(-122.5, 37.0, -121.5, 38.0), ee.ImageCollection('MODIS/061/MOD13A2').first())
    featureCollection = _construct_test_fc_object(geometry)

    parameters = {
        'collection': 'MODIS/061/MOD13A2',
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'vector_path': featureCollection,
        'map_function': map_function,
        'crs': 'EPSG:4326',
        'scale': 1000
    }
    reader = EarthEngineDataset(parameters)
    ee_collection = reader._construct_ee_collection(parameters)
    first_image = ee_collection.first().getInfo()
    assert first_image is not None, "Map function was not applied correctly."

def test_read_data_earth_engine(setup_earth_engine):
    """Test reading Earth Engine data."""
    featureCollection = _construct_test_fc_object(ee.Geometry.Rectangle(113.33, -43.63, 153.56, -10.66))
    parameters = {
        'collection': 'ECMWF/ERA5_LAND/HOURLY',
        'start_date': '1992-10-05',
        'end_date': '1993-03-31',
        'vector_path': featureCollection,
        'crs': 'EPSG:4326',
        'scale': 0.25
    }
    reader = EarthEngineDataset(parameters)
    xarray_data = reader._xarray_data
    assert xarray_data is not None
    assert isinstance(xarray_data, xr.Dataset)
