import unittest
from unittest.mock import patch, MagicMock
from dask.distributed import Client, LocalCluster
import xarray as xr
import numpy as np
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

class TestDaskHandler(unittest.TestCase):

    def test_initialization(self):
        handler = DaskHandler()
        self.assertIsNone(handler.dask_client)

    @patch('__main__.Client')
    def test_create_local_threads(self, mock_client):
        handler = DaskHandler()
        handler.create_local_threads()
        mock_client.assert_called_once_with(processes=False)
        self.assertIsNotNone(handler.dask_client)

    @patch('psutil.virtual_memory')
    @patch('multiprocessing.cpu_count', return_value=4)
    @patch('__main__.Client')
    def test_create_local_cluster(self, mock_client, mock_cpu_count, mock_virtual_memory):
        mock_virtual_memory.return_value.total = 16 * 1024**3  # Mock 16GB total memory

        def side_effect(*args, **kwargs):
            self.assertEqual(kwargs["n_workers"], 4)
            self.assertEqual(kwargs["threads_per_worker"], 1)
            self.assertEqual(kwargs["memory_limit"], '4GB')
            return MagicMock()

        with patch('__main__.LocalCluster', side_effect=side_effect):
            handler = DaskHandler()
            handler.create_local_cluster()
            self.assertIsNotNone(handler.dask_client)
            mock_client.assert_called_once()

    @patch('__main__.Client')
    def test_connect_to_cloud_cluster(self, mock_client):
        handler = DaskHandler()
        handler.connect_to_cloud_cluster('tcp://scheduler-address:8786')
        mock_client.assert_called_once_with('tcp://scheduler-address:8786')
        self.assertIsNotNone(handler.dask_client)

    @patch('xarray.Dataset.chunk', return_value=MagicMock())
    @patch('__main__.Client')
    def test_process_with_dask(self, mock_client, mock_chunk):
        # Create a mock dask_client
        mock_dask_client = MagicMock()
        handler = DaskHandler(dask_client=mock_dask_client)
        
        # Create a dummy dataset
        data = np.random.rand(10, 256, 256)
        dataset = xr.Dataset({'data': (['time', 'latitude', 'longitude'], data)})
        
        chunked_dataset = handler.process_with_dask(dataset)
        
        # Ensure chunk was called with the correct arguments
        mock_chunk.assert_called_once_with({'time': -1, 'latitude': 256, 'longitude': 256})
        
        # Ensure the result is the mock_chunked_dataset
        self.assertEqual(chunked_dataset, mock_chunk.return_value)

    def test_process_without_dask(self):
        handler = DaskHandler()
        data = np.random.rand(10, 256, 256)
        dataset = xr.Dataset({'data': (['time', 'latitude', 'longitude'], data)})
        result = handler.process_with_dask(dataset)
        self.assertEqual(result, dataset)

if __name__ == '__main__':
    unittest.main()