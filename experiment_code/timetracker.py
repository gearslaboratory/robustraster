import os
import csv
from time import time, sleep
from threading import Lock, Thread
from collections import defaultdict

from distributed.diagnostics.plugin import SchedulerPlugin
from distributed.client import Client
from distributed.scheduler import Scheduler


class TaskTimePlugin(SchedulerPlugin):
    """Plugin to track task start and finish times."""

    def __init__(self, scheduler, csv_path):
        SchedulerPlugin.__init__(self)
        self.scheduler = scheduler
        self._lock = Lock()
        self._start_times = {}  # To store start time of each task
        self._task_durations = []  # To store task durations
        self._wall_start = None  # Start time of the overall Dask graph
        self._wall_end = None  # End time of the overall Dask graph
        self.total_tasks = 0  # Track total number of tasks
        self.completed_tasks = 0  # Track completed tasks
        f = open(os.path.join(csv_path), "w", buffering=1)
        self._csv = csv.writer(f)
        self._csv.writerow(["task_key", "duration_seconds"])
        self._csv_walltime = csv.writer(open(f"{csv_path}_walltime.csv", "w", buffering=1))
        self._csv_walltime.writerow(["total_wall_time_seconds"])

    def transition(self, key, start, finish, *args, **kwargs):
        """Called every time a task changes status."""
        current_time = time()

        # Record the start time of the first task (marks the start of the overall graph execution)
        if self._wall_start is None:
            self._wall_start = current_time
            print(f"Started tracking wall time: {self._wall_start}")

        # If the task starts, record the start time
        if start == "waiting" and finish == "processing":
            with self._lock:
                self._start_times[key] = current_time

        # If the task finishes, record the finish time and calculate duration
        if start == "processing" and finish in ("memory", "erred"):
            with self._lock:
                if key in self._start_times:
                    start_time = self._start_times.pop(key)
                    duration = current_time - start_time
                    self._task_durations.append(duration)
                    self._csv.writerow([key, duration])
                    self.completed_tasks += 1

                    # Check if this is the last task
                    if self.completed_tasks == self.total_tasks:
                        self._wall_end = current_time
                        total_wall_time = self._wall_end - self._wall_start
                        self._csv_walltime.writerow([total_wall_time])
                        print(f"Finished tracking wall time: {total_wall_time}")

    def add_worker(self, worker=None, **kwargs):
        """Update total task count when workers are added."""
        with self._lock:
            self.total_tasks += len(worker['tasks'])


def install(scheduler: Scheduler, csv_path: str):
    """Register the time tracking plugin with a distributed Scheduler."""
    plugin = TaskTimePlugin(scheduler, csv_path)
    scheduler.add_plugin(plugin)


# CLI setup for preloading with dask-scheduler
import click

@click.command()
@click.option("--timetracker-csv", default="timetracker.csv")
def dask_setup(scheduler, timetracker_csv):
    install(scheduler, timetracker_csv)
