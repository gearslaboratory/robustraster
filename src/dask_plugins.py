from dask.distributed import Worker
from dask.distributed import WorkerPlugin
import ee 
import json

class EEPlugin(WorkerPlugin):
      def __init__(self, *args, **kwargs):
            pass  # the constructor is up to you
      def setup(self, json_key: str, worker: Worker):
        self.worker = worker
        try:
            if json_key:
                with open(json_key, 'r') as file:
                     data = json.load(file)
                credentials = ee.ServiceAccountCredentials(data["client_email"], json_key)
                ee.Initialize(credentials, opt_url='https://earthengine-highvolume.googleapis.com')
            else:
                ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        except ee.EEException as e:
            if "Please authorize access to your Earth Engine account" in str(e):
                ee.Authenticate()
                ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com') 
      def teardown(self, worker: Worker):
          pass
      def transition(self, key: str, start: str, finish: str,
                      **kwargs):
          pass
      def release_key(self, key: str, state: str, cause: str | None, reason: None, report: bool):
          pass