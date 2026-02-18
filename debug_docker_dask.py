import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

try:
    from robustraster.dask_docker_cluster_manager import DDClusterManager
except ImportError:
    # If src is not in path correctly, try to adjust
    sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
    from robustraster.dask_docker_cluster_manager import DDClusterManager

from dask.distributed import Client
import docker

def cleanup(prefix):
    client = docker.from_env()
    for c in client.containers.list():
        if c.name.startswith(prefix):
            print(f"Stopping {c.name}...")
            c.stop()
            c.remove()

def test():
    with open("debug_output.txt", "w") as log_file:
        def log(msg):
            print(msg)
            log_file.write(str(msg) + "\n")
            log_file.flush()

        log("Testing Docker Dask Cluster...")
        # Use dask image
        image = "ghcr.io/dask/dask:latest"
        
        manager = None
        try:
            manager = DDClusterManager(docker_image=image)
            log(f"Created manager with prefix: {manager._container_prefix}")
            
            log("Creating cluster...")
            manager.create_cluster(n_workers=1, threads_per_worker=1, memory_limit='1GB')
            
            client = manager.get_dask_client
            log(f"Client connected: {client}")
            
            log("Waiting for workers...")
            # Check for workers
            start = time.time()
            while not client.scheduler_info()['workers']:
                if time.time() - start > 60:
                    log("Timeout waiting for workers to register")
                    break
                time.sleep(1)
                
            info = client.scheduler_info()
            log(f"Scheduler Info Workers: {info['workers']}")
            
            if not info['workers']:
                log("No workers registered!")
                return

            log("Submitting simple task...")
            f = client.submit(lambda x: x + 1, 10)
            log(f"Result: {f.result(timeout=10)}")
            log("SUCCESS: Task completed.")
            
        except Exception as e:
            log("Test FAILED with exception:")
            log(e)
            import traceback
            traceback.print_exc(file=log_file)
            
        finally:
            if manager:
                log("Cleaning up...")
                cleanup(manager._container_prefix)

if __name__ == "__main__":
    test()
