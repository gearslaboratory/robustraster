"""
run() function

Parameters I would need from the user:
1. OPTIONAL: DASK CONFIGURATION PARAMETERS (# WORKERS, RAM)
    - If the user doesn't supply this, it will use as much as it can? Maybe?
2. Dataset
    - If Earth Engine, the user passes their image collection. I should be able to derive
      the object type and read their data as an EarthEngineDataset. 
3. Source
    - "ee" for Earth Engine. This will allow me to run conditional code to authenticate the user
      and the Dask workers automatically.
4. Dataset parameters
    - Subparameters for the dataset (scale, crs, etc.)
5. User function
    - User writes their function as passes it into run()
6. Export parameters
    - flag="GTiff", output_folder="test231", chunks={"time": 48, "X": 512, "Y": 256}
    - If flag is "GCS", then gcs_credentials=gcs_json_key, gcs_bucket='gears-bucket-88', gcs_folder="fun"

I think all parameters that have subparameters should be passed in as a dictionary object to simply
the function signature.
"""