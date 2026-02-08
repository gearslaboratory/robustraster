import uuid
import multiprocessing
import psutil
from typing import Optional, Dict, Any

import docker
from dask.distributed import Client

import time
from distributed import Client

class DDClusterManager:
    """
    Docker-backed Dask cluster manager.
    Mirrors the public API of `DaskClusterManager`.
    """
    def __init__(
        self,
        dask_client: Client = None,
        docker_image: Optional[str] = None,
        network_name: str = "dask-network",
        **kwargs,
    ) -> None:
        self.dask_client = dask_client
        self.docker_image = docker_image
        self.network_name = kwargs.get("network_name", network_name)
        self.docker_client = docker.from_env()
        self.scheduler = None
        self.workers = []
        self._container_prefix = f"dask-{uuid.uuid4().hex[:6]}"

        if not self.docker_image:
            raise ValueError("docker_image must be provided to DDClusterManager")

        # Ensure network exists
        try:
            self.docker_client.networks.get(self.network_name)
        except docker.errors.NotFound:
            self.docker_client.networks.create(self.network_name, driver="bridge")

    @property
    def get_dask_client(self) -> Client:
        return self.dask_client

    def _default_memory_limit(self, n_workers: int) -> str:
        total_gb = psutil.virtual_memory().total / (1024 ** 3)
        per_worker = max(1, int(total_gb // max(1, n_workers)))
        return f"{per_worker}GB"

    def _launch_scheduler(self, ports: Dict[str, int] | None = None) -> None:
        ports = ports or {'8786/tcp': 8786, '8787/tcp': 8787}
        name = f"{self._container_prefix}-scheduler"
        self.scheduler = self.docker_client.containers.run(
            self.docker_image,
            command="dask-scheduler --host 0.0.0.0 --dashboard-address :8787",
            name=name,
            network=self.network_name,
            detach=True,
            ports=ports,
        )
        print(f"Dask Scheduler started: {name} ({self.scheduler.short_id})")

    def _launch_worker(self, idx: int, n_threads: int, memory_limit: str, volumes: Dict[str, dict] | None) -> None:
        name = f"{self._container_prefix}-worker-{idx}"
        worker = self.docker_client.containers.run(
            self.docker_image,
            command=f"dask-worker tcp://{self._container_prefix}-scheduler:8786 --nthreads {n_threads} --memory-limit {memory_limit}",
            name=name,
            network=self.network_name,
            detach=True,
            mem_limit=memory_limit,
            volumes=volumes or {},
        )
        self.workers.append(worker)
        print(f"Dask Worker {idx} started: {name} ({worker.short_id})")

    def create_cluster(self, mode: str = "full", **kwargs: Any) -> None:
        """
        Create a Docker-backed Dask cluster.

        kwargs:
          - n_workers / num_workers (int)
          - threads_per_worker / n_threads (int)
          - memory_limit (str)
          - volumes (dict)
          - ports (dict)
        """
        print("APPLES")
        cpu_cores = multiprocessing.cpu_count()

        # harmonize kwargs
        n_workers = kwargs.get("n_workers", kwargs.get("num_workers"))
        threads_per_worker = kwargs.get("threads_per_worker", kwargs.get("n_threads", 1))
        memory_limit = kwargs.get("memory_limit")
        volumes = kwargs.get("volumes", {})
        ports = kwargs.get("ports", None)

        if mode == "full":
            n_workers = n_workers or cpu_cores
            threads_per_worker = threads_per_worker or 1
            memory_limit = memory_limit or self._default_memory_limit(n_workers)
        elif mode == "test":
            n_workers = 1
            threads_per_worker = 1
            total_gb = int(psutil.virtual_memory().total / (1024 ** 3))
            memory_limit = f"{max(1, total_gb // 2)}GB"
        elif mode == "custom":
            n_workers = n_workers or cpu_cores
            threads_per_worker = threads_per_worker or 1
            memory_limit = memory_limit or self._default_memory_limit(n_workers)
        else:
            raise ValueError("Invalid mode. Choose 'full', 'test', or 'custom'.")

        # Launch scheduler and workers
        self._launch_scheduler(ports=ports)
        for i in range(1, n_workers + 1):
            self._launch_worker(i, threads_per_worker, memory_limit, volumes)
        
        #self.dask_client = Client("tcp://127.0.0.1:8786")
        self.dask_client = self._connect_client_with_retry()

    def _connect_client_with_retry(
        self,
        attempts: int = 40,
        delay: float = 0.25,
    ) -> Client:
        """
        Connect to the Dask scheduler using the *actual* published host port,
        retrying until the scheduler is ready.
        """
        # Make sure we have the latest container info
        self.scheduler.reload()

        port_info = self.scheduler.attrs["NetworkSettings"]["Ports"].get("8786/tcp")
        if not port_info:
            raise RuntimeError("Scheduler port 8786 is not published")

        host_port = port_info[0]["HostPort"]
        address = f"tcp://127.0.0.1:{host_port}"

        last_err = None
        for _ in range(attempts):
            try:
                return Client(address, timeout="2s")
            except Exception as e:
                last_err = e
                time.sleep(delay)

        # If we get here, connection never succeeded
        logs = self.scheduler.logs(tail=200).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Could not connect to Dask scheduler at {address}\n"
            f"Last error: {last_err}\n\n"
            f"Scheduler logs:\n{logs}"
        )
