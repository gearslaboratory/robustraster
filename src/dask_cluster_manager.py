from dask.distributed import Client, LocalCluster
import xarray as xr
from .dask_plugins import EEPlugin
import multiprocessing
import psutil

class DaskClusterManager:
    """
    A manager class for handling Dask client operations and configurations.

    This class provides functionality to manage and interact with a Dask Client, which is used 
    for distributed computing tasks. It includes a method to create a Dask cluster of workers 
    needed to parallelize tasks and a method to interact with the workers directly.

    Attributes:
    - dask_client (Client): An instance of a Dask Client. If not provided during initialization, 
                            this can later be set using the methods provided.

    Methods:
    - get_dask_client: A property that returns the current Dask Client instance, allowing the user 
                       to monitor or interact with it.

    - _bytes_to_gigabytes: A private method to convert memory values from bytes to gigabytes.
    
    - create_cluster: A public method that creates a Dask cluster of workers needed to parallelize tasks.

    Example Usage:
    >>> from robustraster import dask_cluster_manager
    >>> dask_cluster = dask_cluster_manager.DaskClusterManager()
    >>> dask_cluster.create_cluster(mode="test")
    >>> dask_client = dask_cluster.get_dask_client
    """
    def __init__(self, dask_client: Client = None) -> None:
        '''
        Initialize the DaskHandler class. Creates a dask_client attribute that will
        be used to store the Dask Client information.
        '''
        self.dask_client = dask_client

    @property
    def get_dask_client(self) -> Client:
        """
        Get the `dask_client` attribute. It also prints IP address to the Dask dashboard, a tool
        the user can use to interact directly with the workers.

        Returns:
        - Client: Gives the user access to the Dask client. This can be useful to monitor your code.
        """
        # Print the dashboard address
        dashboard_address = self.dask_client.dashboard_link
        print(f"Dask dashboard is available at: {dashboard_address}")
        return self.dask_client
    
    def _bytes_to_gigabytes(self, memory: int) -> int:
        '''
        Private method that takes the system memory in bytes and converts it to gigabytes.

        Parameters:
        - memory (int): The total system memory of the machine in bytes. 

        Returns:
        - int: The system memory converted to gigabytes
        '''

        gigabytes = memory / (1024 ** 3)
        return gigabytes

    def create_cluster(self, mode="full", **kwargs) -> None:
        """
        Create a Dask cluster and store it in the `dask_client` attribute.

        Parameters:
        - mode (str): The mode for configuring the cluster. 
                      "full" (default): Optimized for one worker per CPU core. RAM will be split evenly between each worker.
                                         For example, if you have a machine with 16 CPU cores and 32GB of RAM, "full" will
                                         create a cluster of 16 workers with each worker having 2GB of RAM. User can also 
                                         specify specific configurations by adding in keyword arguments.
                      "test": Optimized for a single worker. Only use this mode if you want the code to auto-determine the
                              appropriate chunk size for your machine.

        Keyword Arguments:
        - n_workers (int): Number of workers to create in the cluster. 
                           Overrides the default value determined by the mode.
        - threads_per_worker (int): Number of threads per worker. 
                                    Default is 1.
        - memory_limit (str): Memory limit per worker (e.g., "2GB"). 
                              Overrides the default value determined by the mode.

        Raises:
        - ValueError: If an invalid mode is provided.

        Example Usage:
        - Cluster with default settings:
          >>> from robustraster import dask_cluster_manager
          >>> dask_cluster = dask_cluster_manager.DaskClusterManager()
          >>> dask_cluster.create_cluster(mode="full")

        - Cluster with custom settings:
          >>> from robustraster import dask_cluster_manager
          >>> dask_cluster = dask_cluster_manager.DaskClusterManager()
          >>> dask_cluster.create_cluster(mode="full", n_workers=2, memory_limit="2GB")

        - Test cluster:
          >>> from robustraster import dask_cluster_manager
          >>> dask_cluster = dask_cluster_manager.DaskClusterManager()
          >>> dask_cluster.create_cluster(mode="test")
        """
        num_cores = multiprocessing.cpu_count()
        total_memory = psutil.virtual_memory().total
        total_memory_gb = self._bytes_to_gigabytes(total_memory)

         # Determine default settings based on mode
        if mode == "local":
            # Use kwargs values if provided, otherwise use default values
            n_workers = kwargs.get('n_workers', num_cores)
            threads_per_worker = kwargs.get('threads_per_worker', 1)
            memory_limit = kwargs.get('memory_limit', f"{total_memory_gb / num_cores}GB")
        elif mode == "test":
            n_workers = 1
            threads_per_worker = 1
            memory_limit = f"{int(total_memory_gb)}GB"
        else:
            raise ValueError("Invalid mode. Choose either 'local' or 'test'.")

        # Create the Dask client with a LocalCluster
        self.dask_client = Client(
            LocalCluster(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                memory_limit=memory_limit
            )
        )

        # Print the dashboard address
        dashboard_address = self.dask_client.dashboard_link
        print(f"Dask dashboard is available at: {dashboard_address}")

    def connect_to_cloud_cluster(self, scheduler_address: str):
        '''
        Store dask_client attribute with a Dask Client object configured (WIP) to connect to a Cloud service.
        '''
        self.dask_client = Client(scheduler_address)