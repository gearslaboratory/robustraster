import docker
import psutil  # To get system memory information
from dask.distributed import Client

class DaskClusterManager:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.network_name = "dask-network"
        self.scheduler = None
        self.workers = []
        self.dask_client = None  # To store the Dask client object

        # Create Docker network if it doesn't exist
        try:
            self.docker_client.networks.get(self.network_name)
        except docker.errors.NotFound:
            self.docker_client.networks.create(self.network_name, driver="bridge")

    def create_test_cluster(self, *args, **kwargs):
        # Get half of the available system memory
        total_memory = psutil.virtual_memory().total
        memory_limit = f"{total_memory // (2 * (1024**3))}GB"  # Convert to GB
        
        # Start Dask Scheduler
        self.scheduler = self.docker_client.containers.run(
            "adrianomdocker/rrtest",
            command="dask-scheduler",
            name="dask-scheduler",
            network=self.network_name,
            detach=True,
            ports={'8786/tcp': 8786, '8787/tcp': 8787},
        )
        print(f"Dask Scheduler started with ID {self.scheduler.id}")

        # Extract any volume mounts passed via kwargs
        volumes = kwargs.get('volumes', {})

        # Start a single worker with 1 thread and half the system memory
        worker = self.docker_client.containers.run(
            "adrianomdocker/rrtest",
            command=f"dask-worker dask-scheduler:8786 --nthreads 1 --memory-limit {memory_limit}",
            name="dask-worker-1",
            network=self.network_name,
            detach=True,
            mem_limit=memory_limit,
            volumes=volumes
        )
        self.workers.append(worker)
        print(f"Dask Worker started with ID {worker.id}")
        
        print("Test cluster created with 1 worker and half the system memory.")
        print("Dask dashboard available at http://localhost:8787")

    def create_cluster(self, num_workers=1, n_threads=1, memory_limit="8GB", *args, **kwargs):
        # Start Dask Scheduler
        self.scheduler = self.docker_client.containers.run(
            "adrianomdocker/rrtest",
            command="dask-scheduler",
            name="dask-scheduler",
            network=self.network_name,
            detach=True,
            ports={'8786/tcp': 8786, '8787/tcp': 8787},
        )
        print(f"Dask Scheduler started with ID {self.scheduler.id}")

        # Extract any volume mounts passed via kwargs
        volumes = kwargs.get('volumes', {})
        
        # Start specified number of workers
        for i in range(num_workers):
            worker = self.docker_client.containers.run(
                "adrianomdocker/rrtest",
                command=f"dask-worker dask-scheduler:8786 --nthreads {n_threads} --memory-limit {memory_limit}",
                name=f"dask-worker-{i+1}",
                network=self.network_name,
                detach=True,
                mem_limit=memory_limit,
                volumes=volumes  # Pass volumes if specified
            )
            self.workers.append(worker)
            print(f"Dask Worker {i+1} started with ID {worker.id}")
        
        print(f"Cluster created with {num_workers} workers, each with {n_threads} threads and {memory_limit} memory limit.")
        print("Dask dashboard available at http://localhost:8787")

    def stop_and_remove_containers(self):
        # Stop and remove all workers
        for worker in self.workers:
            worker.stop()
            worker.remove()
            print(f"Dask Worker {worker.id} stopped and removed.")
        self.workers = []

        # Stop and remove the scheduler
        if self.scheduler:
            self.scheduler.stop()
            self.scheduler.remove()
            print(f"Dask Scheduler {self.scheduler.id} stopped and removed.")
            self.scheduler = None

    def get_dask_client(self):
        # Ensure that the scheduler is running
        if not self.scheduler:
            raise RuntimeError("Dask Scheduler is not running. Please start the cluster first.")
        
        self.dask_client = Client("tcp://localhost:8786")
        return self.dask_client