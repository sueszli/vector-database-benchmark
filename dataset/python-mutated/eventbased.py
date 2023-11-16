import threading
from typing import Callable, Optional
import watchdog.observers
from watchdog import events
from chalice.cli.filewatch import FileWatcher, WorkerProcess

class WatchdogWorkerProcess(WorkerProcess):
    """Worker that runs the chalice dev server."""

    def _start_file_watcher(self, project_dir):
        if False:
            print('Hello World!')
        restart_callback = WatchdogRestarter(self._restart_event)
        watcher = WatchdogFileWatcher()
        watcher.watch_for_file_changes(project_dir, restart_callback)

class WatchdogFileWatcher(FileWatcher):

    def watch_for_file_changes(self, root_dir, callback):
        if False:
            while True:
                i = 10
        observer = watchdog.observers.Observer()
        observer.schedule(callback, root_dir, recursive=True)
        observer.start()

class WatchdogRestarter(events.FileSystemEventHandler):

    def __init__(self, restart_event):
        if False:
            for i in range(10):
                print('nop')
        self.restart_event = restart_event

    def on_any_event(self, event):
        if False:
            return 10
        if event.is_directory:
            return
        self()

    def __call__(self):
        if False:
            return 10
        self.restart_event.set()