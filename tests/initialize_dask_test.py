import multiprocessing
from dask.distributed import Client, LocalCluster
import dask.array as da
import xarray as xr

class DaskHandler:
    def __init__(self, dask_client: Client = None) -> None:
        self.dask_client = dask_client

    def _calculate_memory_limit(self, memory: int) -> str:
        if memory >= 1024:
            return f"{memory // 1024}GB"  # Convert to GB if memory is more than or equal to 1GB
        return f"{memory}MB"  # Use MB otherwise
    
    def create_local_threads(self):
        self.dask_client = Client(processes=False)

    def create_local_cluster(self):
        num_cores = multiprocessing.cpu_count()
        memory = int(multiprocessing.virtual_memory().total / num_cores)
        memory_per_worker = memory // num_cores
        memory_limit = self._calculate_memory_limit(memory_per_worker)
        self.dask_client = Client(LocalCluster(n_workers=num_cores, threads_per_worker=1, memory_limit=memory_limit))

    def connect_to_cloud_cluster(self, scheduler_address: str):
        self.dask_client = Client(scheduler_address)

    def process_with_dask(self, dataset: xr.Dataset) -> xr.Dataset:
        if self.dask_client:
            return dataset.chunk({'time': -1, 'latitude': 256, 'longitude': 256})
        return dataset
