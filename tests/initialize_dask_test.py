import unittest
import os
import sys
import xarray as xr
import numpy as np
import dask.array as da
import dask.dataframe as dd
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from initialize_dask import DaskHandler

class TestDaskHandler(unittest.TestCase):
    def test_initialization(self):
        ''' 
        Test initializing an attribute "self.dask_client" with a Client object containing (for now) None.
        The methods below will assign this attribute with a Client object depending on how they choose to
        configure their Dask environment.

        Test Assertions:
        - assertIsNotNone: Check if the attribute "self.dask_client" is not None.
        '''
        # Test reading raster data successfully
        handler = DaskHandler()
        self.assertIsNone(handler.dask_client)

    @patch('initialize_dask.Client')
    def test_create_local_threads(self, mock_client):
        ''' 
        Test assigning attribute "self.dask_client" a Client object meant to create a mock Dask Client object 
        using the threads of the local machine. Uses the @patch decorator to replace dask.distributed.Client
        with a mock object.

        Test Assertions:
        - assertIsNotNone: Check if the attribute "self.dask_client" is not None.
        '''
        handler = DaskHandler()
        handler.create_local_threads()
        mock_client.assert_called_once_with(processes=False)
        self.assertIsNotNone(handler.dask_client)

    @patch('psutil.virtual_memory')
    @patch('multiprocessing.cpu_count', return_value=4)
    @patch('initialize_dask.Client')
    def test_create_local_cluster_defaults(self, mock_client, mock_cpu_count, mock_virtual_memory):
        ''' 
        Test assigning attribute "self.dask_client" a Client object meant to create a mock Dask Client object.
        This mock Client object tests the creation of a local cluster of workers with default settings. Uses the @patch decorator to 
        replace the following:
        
        1. @patch('psutil.virtual_memory')
            This will replace the psutil.virtual_memory method with a mock value of "16GB" meant to test the use
            of this method.
        2. @patch('multiprocessing.cpu_count', return_value=4)
            This will replace the multiprocessing.cpu_count method with a mock return value of 4. This is meant
            to test the use of this method.
        3. @patch('initialize_dask.Client')
            This will replace dask.distributed.Client with a mock object meant to test the use of this method.
        
        In order to test the 3 methods mentioned above, another patch is done on dask.distributed.LocalCluster to
        run the side_effect() method (see docstring for side_effect() below).

        Test Assertions:
        - assertEqual: Assert the values of "n_workers", "threads_per_worker", and "memory_limit" are of the same
                       values as the mock objects (see side_effect() method below).
        - assertIsNotNone: Check if the attribute "self.dask_client" is not None.
        '''
        mock_virtual_memory.return_value.total = 16 * 1024**3  # Mock 16GB total memory

        def side_effect(*args, **kwargs):
            '''
            When dask.distributed.LocalCluster is called via handler.create_local_cluster(), replace it with 
            this method. Tests dask.distributed.LocalCluster using the 3 mock objects created above.

            Test Assertions:
            - assertEqual: Assert the values of "n_workers", "threads_per_worker", and "memory_limit" are of 
            the same values as the mock objects. 
            '''
            self.assertEqual(kwargs["n_workers"], 4)
            self.assertEqual(kwargs["threads_per_worker"], 1)
            self.assertEqual(kwargs["memory_limit"], '4GB')

        with patch('initialize_dask.LocalCluster', side_effect=side_effect):
            handler = DaskHandler()
            handler.create_local_cluster()
            self.assertIsNotNone(handler.dask_client)
            mock_client.assert_called_once()
    
    @patch('initialize_dask.Client')
    @patch('initialize_dask.LocalCluster')
    def test_create_local_cluster_with_kwargs(self, mock_local_cluster, mock_client):
        '''
        Test assigning attribute "self.dask_client" a Client object meant to create a mock Dask Client object.
        This mock Client object tests the creation of a local cluster of workers with user-provided kwargs.

        Test Assertions:
        - assertEqual: Assert the values of "n_workers", "threads_per_worker", and "memory_limit" are of the same
                       values as the user-provided kwargs.
        - assertIsNotNone: Check if the attribute "self.dask_client" is not None.
        '''
        handler = DaskHandler()
        handler.create_local_cluster(n_workers=2, threads_per_worker=2, memory_limit='2GB')
        
        mock_local_cluster.assert_called_once_with(n_workers=2, threads_per_worker=2, memory_limit='2GB')
        self.assertIsNotNone(handler.dask_client)
        mock_client.assert_called_once()

    @patch('initialize_dask.Client')
    def test_connect_to_cloud_cluster(self, mock_client):
        '''
        WRITE THIS DOCUMENTATION ONCE I FINISH WRITING THE CODE FOR THE REAL METHOD!11
        '''
        handler = DaskHandler()
        handler.connect_to_cloud_cluster('tcp://scheduler-address:8786')
        mock_client.assert_called_once_with('tcp://scheduler-address:8786')
        self.assertIsNotNone(handler.dask_client)

    @patch('xarray.Dataset.chunk', return_value=MagicMock())
    @patch('initialize_dask.Client')
    def test_process_with_dask(self, mock_client, mock_chunk):
        '''
        Test chunking a xarray.Dataset to convert it to a Dask array. This method patches two other methods using
        @patch decorator:

        1. @patch('xarray.Dataset.chunk', return_value=MagicMock())
           This will replace the xarray.Dataset.chunk method that, when called, will return a MagicMock object.
           This MagicMock object is meant to test the xarray.Dataset.chunk method.
        
        2. @patch('initialize_dask.Client')
           This will replace dask.distributed.Client with a mock object meant to test the use of this method.

        Test Assertions:
        - assertEqual: Asserts the value of "chunked_dataset" is equal to the return value of "mock_chunk". In 
                       case, because we are testing with mock objects, the values of each of these variables should
                       be the same MagicMock objects.
        '''
        # Create a mock dask_client
        mock_dask_client = MagicMock()
        handler = DaskHandler(dask_client=mock_dask_client)
        
        # Create a dummy dataset
        data = np.random.rand(10, 256, 256)
        dataset = xr.Dataset({'data': (['time', 'X', 'Y'], data)})
        
        chunked_dataset = handler.process_with_dask(dataset)
        
        # Ensure chunk was called with the correct arguments
        mock_chunk.assert_called_once_with({'time': 48, 'X': 512, 'Y': 256})
        
        # Ensure the result is the mock_chunked_dataset
        self.assertEqual(chunked_dataset, mock_chunk.return_value)

    def test_process_without_dask(self):
        '''
        (probably) NOT NEEDED! IF WE AREN'T GOING TO CHUNK THE DATA, WHY DO WE NEED A DASK HANDLER?
        Test handling an xarray.Dataset object. In other words, we are not converting the xarray.Dataset into a Dask 
        array, as we will not be doing any chunking.

        Test Assertions:
        - assertEqual: Assert that the return value of process_with_dask() is the same as "dataset".
        '''
        handler = DaskHandler()
        data = np.random.rand(10, 256, 256)
        dataset = xr.Dataset({'data': (['time', 'latitude', 'longitude'], data)})
        result = handler.process_with_dask(dataset)
        self.assertEqual(result, dataset)

    def test_dataset_to_dask_dataframe(self):
        '''
        Test converting a chunked xarray.Dataset to a Dask DataFrame.

        Test Assertions:
        - assertIsInstance: Check if the result is an instance of dask.dataframe.DataFrame.
        - assertEqual: Check if the columns of the DataFrame match the variables in the Dataset.
        '''
        handler = DaskHandler()

        # Create a sample xarray Dataset with Dask arrays
        data = da.random.random((10, 256, 256), chunks=(5, 128, 128))
        data_array = xr.DataArray(data, dims=["time", "latitude", "longitude"])
        dataset = xr.Dataset({"variable1": data_array})

        # Process the dataset to create chunks
        chunked_dataset = dataset.chunk({"time": 5, "latitude": 128, "longitude": 128})

        # Convert the chunked xarray Dataset to a Dask DataFrame
        dask_df = handler.dataset_to_dask_dataframe(chunked_dataset)

        # Check that the result is a Dask DataFrame
        self.assertIsInstance(dask_df, dd.DataFrame)

        # Check that the Dask DataFrame has the expected columns
        expected_columns = list(chunked_dataset.to_dask_dataframe().columns)
        self.assertEqual(list(dask_df.columns), expected_columns)


if __name__ == '__main__':
    unittest.main()