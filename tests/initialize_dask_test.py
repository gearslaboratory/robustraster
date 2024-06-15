import unittest
import os
import sys
import xarray as xr
import numpy as np
from unittest.mock import patch, MagicMock
from dask.distributed import Client, LocalCluster

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from initialize_dask import DaskHandler

class TestDaskHandler(unittest.TestCase):
    def test_initialization(self):
        handler = DaskHandler()
        self.assertIsNone(handler.dask_client)

    @patch('initialize_dask.Client')
    def test_create_local_threads(self, mock_client):
        handler = DaskHandler()
        handler.create_local_threads()
        mock_client.assert_called_once_with(processes=False)
        self.assertIsNotNone(handler.dask_client)

    @patch('psutil.virtual_memory')
    @patch('multiprocessing.cpu_count', return_value=4)
    @patch('initialize_dask.Client')
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

    @patch('initialize_dask.Client')
    def test_connect_to_cloud_cluster(self, mock_client):
        handler = DaskHandler()
        handler.connect_to_cloud_cluster('tcp://scheduler-address:8786')
        mock_client.assert_called_once_with('tcp://scheduler-address:8786')
        self.assertIsNotNone(handler.dask_client)

    @patch('xarray.Dataset.chunk', return_value=MagicMock())
    @patch('initialize_dask.Client')
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

    def no_test_process_without_dask(self):
        handler = DaskHandler()
        data = np.random.rand(10, 256, 256)
        dataset = xr.Dataset({'data': (['time', 'latitude', 'longitude'], data)})
        result = handler.process_with_dask(dataset)
        self.assertEqual(result, dataset)

if __name__ == '__main__':
    unittest.main()