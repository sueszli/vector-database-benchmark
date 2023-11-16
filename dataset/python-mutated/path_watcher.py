from typing import Callable, Optional, Type, Union
import click
import streamlit.watcher
from streamlit import config, env_util
from streamlit.logger import get_logger
from streamlit.watcher.polling_path_watcher import PollingPathWatcher
LOGGER = get_logger(__name__)
try:
    from streamlit.watcher.event_based_path_watcher import EventBasedPathWatcher
    watchdog_available = True
except ImportError:
    watchdog_available = False

    class EventBasedPathWatcher:
        pass

class NoOpPathWatcher:

    def __init__(self, _path_str: str, _on_changed: Callable[[str], None], *, glob_pattern: Optional[str]=None, allow_nonexistent: bool=False):
        if False:
            i = 10
            return i + 15
        pass
PathWatcherType = Union[Type['streamlit.watcher.event_based_path_watcher.EventBasedPathWatcher'], Type[PollingPathWatcher], Type[NoOpPathWatcher]]

def report_watchdog_availability():
    if False:
        while True:
            i = 10
    if not watchdog_available:
        if not config.get_option('global.disableWatchdogWarning'):
            msg = '\n  $ xcode-select --install' if env_util.IS_DARWIN else ''
            click.secho('  %s' % 'For better performance, install the Watchdog module:', fg='blue', bold=True)
            click.secho('%s\n  $ pip install watchdog\n            ' % msg)

def _watch_path(path: str, on_path_changed: Callable[[str], None], watcher_type: Optional[str]=None, *, glob_pattern: Optional[str]=None, allow_nonexistent: bool=False) -> bool:
    if False:
        return 10
    "Create a PathWatcher for the given path if we have a viable\n    PathWatcher class.\n\n    Parameters\n    ----------\n    path\n        Path to watch.\n    on_path_changed\n        Function that's called when the path changes.\n    watcher_type\n        Optional watcher_type string. If None, it will default to the\n        'server.fileWatcherType` config option.\n    glob_pattern\n        Optional glob pattern to use when watching a directory. If set, only\n        files matching the pattern will be counted as being created/deleted\n        within the watched directory.\n    allow_nonexistent\n        If True, allow the file or directory at the given path to be\n        nonexistent.\n\n    Returns\n    -------\n    bool\n        True if the path is being watched, or False if we have no\n        PathWatcher class.\n    "
    if watcher_type is None:
        watcher_type = config.get_option('server.fileWatcherType')
    watcher_class = get_path_watcher_class(watcher_type)
    if watcher_class is NoOpPathWatcher:
        return False
    watcher_class(path, on_path_changed, glob_pattern=glob_pattern, allow_nonexistent=allow_nonexistent)
    return True

def watch_file(path: str, on_file_changed: Callable[[str], None], watcher_type: Optional[str]=None) -> bool:
    if False:
        i = 10
        return i + 15
    return _watch_path(path, on_file_changed, watcher_type)

def watch_dir(path: str, on_dir_changed: Callable[[str], None], watcher_type: Optional[str]=None, *, glob_pattern: Optional[str]=None, allow_nonexistent: bool=False) -> bool:
    if False:
        print('Hello World!')
    return _watch_path(path, on_dir_changed, watcher_type, glob_pattern=glob_pattern, allow_nonexistent=allow_nonexistent)

def get_default_path_watcher_class() -> PathWatcherType:
    if False:
        print('Hello World!')
    'Return the class to use for path changes notifications, based on the\n    server.fileWatcherType config option.\n    '
    return get_path_watcher_class(config.get_option('server.fileWatcherType'))

def get_path_watcher_class(watcher_type: str) -> PathWatcherType:
    if False:
        print('Hello World!')
    "Return the PathWatcher class that corresponds to the given watcher_type\n    string. Acceptable values are 'auto', 'watchdog', 'poll' and 'none'.\n    "
    if watcher_type == 'auto':
        if watchdog_available:
            return EventBasedPathWatcher
        else:
            return PollingPathWatcher
    elif watcher_type == 'watchdog' and watchdog_available:
        return EventBasedPathWatcher
    elif watcher_type == 'poll':
        return PollingPathWatcher
    else:
        return NoOpPathWatcher