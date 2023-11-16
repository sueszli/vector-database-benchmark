"""
HandlerObserver and its helper classes.
"""
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional
from watchdog.events import EVENT_TYPE_OPENED, FileSystemEvent, FileSystemEventHandler, RegexMatchingEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import DEFAULT_OBSERVER_TIMEOUT, ObservedWatch
LOG = logging.getLogger(__name__)

@dataclass
class PathHandler:
    """PathHandler is an object that can be passed into
    Bundle Observer directly for watching a specific path with
    corresponding EventHandler

    Fields:
        event_handler : FileSystemEventHandler
            Handler for the event
        path : Path
            Path to the folder to be watched
        recursive : bool, optional
            True to watch child folders, by default False
        static_folder : bool, optional
            Should the observed folder name be static, by default False
            See StaticFolderWrapper on the use case.
        self_create : Optional[Callable[[], None]], optional
            Callback when the folder to be observed itself is created, by default None
            This will not be called if static_folder is False
        self_delete : Optional[Callable[[], None]], optional
            Callback when the folder to be observed itself is deleted, by default None
            This will not be called if static_folder is False
    """
    event_handler: FileSystemEventHandler
    path: Path
    recursive: bool = False
    static_folder: bool = False
    self_create: Optional[Callable[[], None]] = None
    self_delete: Optional[Callable[[], None]] = None

class StaticFolderWrapper:
    """This class is used to alter the behavior of watchdog folder watches.
    https://github.com/gorakhargosh/watchdog/issues/415
    By default, if a folder is renamed, the handler will still get triggered for the new folder
    Ex:
        1. Create FolderA
        2. Watch FolderA
        3. Rename FolderA to FolderB
        4. Add file to FolderB
        5. Handler will get event for adding the file to FolderB but with event path still as FolderA
    This class watches the parent folder and if the folder to be watched gets renamed or deleted,
    the watch will be stopped and changes in the renamed folder will not be triggered.
    """

    def __init__(self, observer: 'HandlerObserver', initial_watch: ObservedWatch, path_handler: PathHandler):
        if False:
            for i in range(10):
                print('nop')
        '[summary]\n\n        Parameters\n        ----------\n        observer : HandlerObserver\n            HandlerObserver\n        initial_watch : ObservedWatch\n            Initial watch for the folder to be watched that gets returned by HandlerObserver\n        path_handler : PathHandler\n            PathHandler of the folder to be watched.\n        '
        self._observer = observer
        self._path_handler = path_handler
        self._watch: Optional[ObservedWatch] = initial_watch

    def _on_parent_change(self, event: FileSystemEvent) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Callback for changes detected in the parent folder\n\n        Parameters\n        ----------\n        event: FileSystemEvent\n            event\n        '
        if event.event_type == EVENT_TYPE_OPENED:
            LOG.debug('Ignoring file system OPENED event.')
            return
        if self._watch and (not self._path_handler.path.exists()):
            if self._path_handler.self_delete:
                self._path_handler.self_delete()
            self._observer.unschedule(self._watch)
            self._watch = None
        elif not self._watch and self._path_handler.path.exists():
            if self._path_handler.self_create:
                self._path_handler.self_create()
            self._watch = self._observer.schedule_handler(self._path_handler)

    def get_dir_parent_path_handler(self) -> PathHandler:
        if False:
            print('Hello World!')
        'Get PathHandler that watches the folder changes from the parent folder.\n\n        Returns\n        -------\n        PathHandler\n            PathHandler for the parent folder. This should be added back into the HandlerObserver.\n        '
        dir_path = self._path_handler.path.resolve()
        parent_dir_path = dir_path.parent
        parent_folder_handler = RegexMatchingEventHandler(regexes=[f'^{re.escape(str(dir_path))}$'], ignore_regexes=[], ignore_directories=False, case_sensitive=True)
        parent_folder_handler.on_any_event = self._on_parent_change
        return PathHandler(path=parent_dir_path, event_handler=parent_folder_handler)

class HandlerObserver(Observer):
    """
    Extended WatchDog Observer that takes in a single PathHandler object.
    """

    def __init__(self, timeout=DEFAULT_OBSERVER_TIMEOUT):
        if False:
            print('Hello World!')
        super().__init__(timeout=timeout)

    def schedule_handlers(self, path_handlers: List[PathHandler]) -> List[ObservedWatch]:
        if False:
            while True:
                i = 10
        'Schedule a list of PathHandlers\n\n        Parameters\n        ----------\n        path_handlers : List[PathHandler]\n            List of PathHandlers to be scheduled\n\n        Returns\n        -------\n        List[ObservedWatch]\n            List of ObservedWatch corresponding to path_handlers in the same order.\n        '
        watches = list()
        for path_handler in path_handlers:
            watches.append(self.schedule_handler(path_handler))
        return watches

    def schedule_handler(self, path_handler: PathHandler) -> ObservedWatch:
        if False:
            while True:
                i = 10
        'Schedule a PathHandler\n\n        Parameters\n        ----------\n        path_handler : PathHandler\n            PathHandler to be scheduled\n\n        Returns\n        -------\n        ObservedWatch\n            ObservedWatch corresponding to the PathHandler.\n            If static_folder is True, the parent folder watch will be returned instead.\n        '
        watch: ObservedWatch = self.schedule(path_handler.event_handler, str(path_handler.path), path_handler.recursive)
        if path_handler.static_folder:
            static_wrapper = StaticFolderWrapper(self, watch, path_handler)
            parent_path_handler = static_wrapper.get_dir_parent_path_handler()
            watch = self.schedule_handler(parent_path_handler)
        return watch