from dask.distributed import Client, LocalCluster
import xarray as xr
import multiprocessing
import psutil

class DaskHandler:
    def __init__(self, dask_client: Client = None) -> None:
        self.dask_client = dask_client

    def _bytes_to_gigabytes(self, memory: int) -> str:
        gigabytes = memory / (1024 ** 3)
        return gigabytes
    
    def create_local_threads(self):
        self.dask_client = Client(processes=False)

    def create_local_cluster(self):
        num_cores = multiprocessing.cpu_count()
        total_memory = psutil.virtual_memory().total
        total_memory_gb = self._bytes_to_gigabytes(total_memory)
        memory_per_worker = f"{int(total_memory_gb / num_cores)}GB"
        self.dask_client = Client(LocalCluster(n_workers=num_cores, threads_per_worker=1, memory_limit=memory_per_worker))

    def connect_to_cloud_cluster(self, scheduler_address: str):
        self.dask_client = Client(scheduler_address)

    def process_with_dask(self, dataset: xr.Dataset) -> xr.Dataset:
        if self.dask_client:
            return dataset.chunk({'time': -1, 'latitude': 256, 'longitude': 256})
        return dataset