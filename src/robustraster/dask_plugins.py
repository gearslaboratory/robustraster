import os
import json
import ee
from distributed import WorkerPlugin

class EEPlugin(WorkerPlugin):
    def __init__(self, json_key: str = None):
        self.json_key = json_key
    
    def setup(self, worker):
        self.worker = worker
        try:
            if self.json_key and os.path.exists(self.json_key):
                with open(self.json_key, 'r') as file:
                     data = json.load(file)
                credentials = ee.ServiceAccountCredentials(data["client_email"], self.json_key)
                ee.Initialize(credentials, opt_url='https://earthengine-highvolume.googleapis.com')
            else:
                ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        except ee.EEException as e:
            if "Please authorize access to your Earth Engine account" in str(e):
                ee.Authenticate()
                ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com') 

    def teardown(self, worker):
        pass

    def transition(self, key, start, finish, **kwargs):
        pass

    def release_key(self, key, state, cause, reason, report):
        pass