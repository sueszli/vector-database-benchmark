import collections
import os
import sys
import types
from typing import Callable, Dict, List, Optional, Set
from streamlit import config, file_util
from streamlit.folder_black_list import FolderBlackList
from streamlit.logger import get_logger
from streamlit.source_util import get_pages
from streamlit.watcher.path_watcher import NoOpPathWatcher, get_default_path_watcher_class
LOGGER = get_logger(__name__)
WatchedModule = collections.namedtuple('WatchedModule', ['watcher', 'module_name'])
PathWatcher = None

class LocalSourcesWatcher:

    def __init__(self, main_script_path: str):
        if False:
            i = 10
            return i + 15
        self._main_script_path = os.path.abspath(main_script_path)
        self._script_folder = os.path.dirname(self._main_script_path)
        self._on_file_changed: List[Callable[[str], None]] = []
        self._is_closed = False
        self._cached_sys_modules: Set[str] = set()
        self._folder_black_list = FolderBlackList(config.get_option('server.folderWatchBlacklist'))
        self._watched_modules: Dict[str, WatchedModule] = {}
        for page_info in get_pages(self._main_script_path).values():
            self._register_watcher(page_info['script_path'], module_name=None)

    def register_file_change_callback(self, cb: Callable[[str], None]) -> None:
        if False:
            return 10
        self._on_file_changed.append(cb)

    def on_file_changed(self, filepath):
        if False:
            while True:
                i = 10
        if filepath not in self._watched_modules:
            LOGGER.error('Received event for non-watched file: %s', filepath)
            return
        for wm in self._watched_modules.values():
            if wm.module_name is not None and wm.module_name in sys.modules:
                del sys.modules[wm.module_name]
        for cb in self._on_file_changed:
            cb(filepath)

    def close(self):
        if False:
            print('Hello World!')
        for wm in self._watched_modules.values():
            wm.watcher.close()
        self._watched_modules = {}
        self._is_closed = True

    def _register_watcher(self, filepath, module_name):
        if False:
            return 10
        global PathWatcher
        if PathWatcher is None:
            PathWatcher = get_default_path_watcher_class()
        if PathWatcher is NoOpPathWatcher:
            return
        try:
            wm = WatchedModule(watcher=PathWatcher(filepath, self.on_file_changed), module_name=module_name)
        except PermissionError:
            return
        self._watched_modules[filepath] = wm

    def _deregister_watcher(self, filepath):
        if False:
            while True:
                i = 10
        if filepath not in self._watched_modules:
            return
        if filepath == self._main_script_path:
            return
        wm = self._watched_modules[filepath]
        wm.watcher.close()
        del self._watched_modules[filepath]

    def _file_is_new(self, filepath):
        if False:
            print('Hello World!')
        return filepath not in self._watched_modules

    def _file_should_be_watched(self, filepath):
        if False:
            while True:
                i = 10
        return self._file_is_new(filepath) and (file_util.file_is_in_folder_glob(filepath, self._script_folder) or file_util.file_in_pythonpath(filepath))

    def update_watched_modules(self):
        if False:
            return 10
        if self._is_closed:
            return
        if set(sys.modules) != self._cached_sys_modules:
            modules_paths = {name: self._exclude_blacklisted_paths(get_module_paths(module)) for (name, module) in dict(sys.modules).items()}
            self._cached_sys_modules = set(sys.modules)
            self._register_necessary_watchers(modules_paths)

    def _register_necessary_watchers(self, module_paths: Dict[str, Set[str]]) -> None:
        if False:
            while True:
                i = 10
        for (name, paths) in module_paths.items():
            for path in paths:
                if self._file_should_be_watched(path):
                    self._register_watcher(path, name)

    def _exclude_blacklisted_paths(self, paths: Set[str]) -> Set[str]:
        if False:
            i = 10
            return i + 15
        return {p for p in paths if not self._folder_black_list.is_blacklisted(p)}

def get_module_paths(module: types.ModuleType) -> Set[str]:
    if False:
        i = 10
        return i + 15
    paths_extractors = [lambda m: [m.__file__], lambda m: [m.__spec__.origin], lambda m: [p for p in m.__path__._path]]
    all_paths = set()
    for extract_paths in paths_extractors:
        potential_paths = []
        try:
            potential_paths = extract_paths(module)
        except AttributeError:
            pass
        except Exception as e:
            LOGGER.warning(f'Examining the path of {module.__name__} raised: {e}')
        all_paths.update([os.path.abspath(str(p)) for p in potential_paths if _is_valid_path(p)])
    return all_paths

def _is_valid_path(path: Optional[str]) -> bool:
    if False:
        print('Hello World!')
    return isinstance(path, str) and (os.path.isfile(path) or os.path.isdir(path))