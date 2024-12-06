import ee
import json
from google.auth.exceptions import RefreshError
from dask.distributed import Client

def initialize_earth_engine(json_key: str = None):
    '''
    Initialize the Earth Engine API. If a service account JSON key is provided, use it to authenticate.
    Otherwise, try to use the existing credentials or prompt for authentication.

    Parameters:
    - json_key (str): Path to the service account JSON credentials file.

    Returns:
    - None
    '''
    if json_key:
        with open(json_key, 'r') as file:
            data = json.load(file)
        credentials = ee.ServiceAccountCredentials(data["client_email"], json_key)
        ee.Initialize(credentials, opt_url='https://earthengine-highvolume.googleapis.com')
    else:
        try:
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        except ee.EEException:
            ee.Authenticate()
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        except RefreshError:
            ee.Authenticate()
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')