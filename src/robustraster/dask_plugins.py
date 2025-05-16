import os
import json
import ee
from distributed import WorkerPlugin

class EEPlugin(WorkerPlugin):
    def __init__(self):
        pass
    def setup(self, worker):
        self.worker = worker
        try:
            # Assume credentials already exist at default location
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        except ee.EEException as e:
            raise RuntimeError("Earth Engine initialization failed. "
                            "Run ee.Authenticate() once on the client before starting the cluster.")