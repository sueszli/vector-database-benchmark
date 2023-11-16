from __future__ import annotations
import fnmatch
import os
import subprocess
import sys
import threading
import time
import typing as t
from itertools import chain
from pathlib import PurePath
from ._internal import _log
_ignore_always = tuple({sys.base_prefix, sys.base_exec_prefix})
prefix = {*_ignore_always, sys.prefix, sys.exec_prefix}
if hasattr(sys, 'real_prefix'):
    prefix.add(sys.real_prefix)
_stat_ignore_scan = tuple(prefix)
del prefix
_ignore_common_dirs = {'__pycache__', '.git', '.hg', '.tox', '.nox', '.pytest_cache', '.mypy_cache'}

def _iter_module_paths() -> t.Iterator[str]:
    if False:
        while True:
            i = 10
    'Find the filesystem paths associated with imported modules.'
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name is None or name.startswith(_ignore_always):
            continue
        while not os.path.isfile(name):
            old = name
            name = os.path.dirname(name)
            if name == old:
                break
        else:
            yield name

def _remove_by_pattern(paths: set[str], exclude_patterns: set[str]) -> None:
    if False:
        return 10
    for pattern in exclude_patterns:
        paths.difference_update(fnmatch.filter(paths, pattern))

def _find_stat_paths(extra_files: set[str], exclude_patterns: set[str]) -> t.Iterable[str]:
    if False:
        while True:
            i = 10
    'Find paths for the stat reloader to watch. Returns imported\n    module files, Python files under non-system paths. Extra files and\n    Python files under extra directories can also be scanned.\n\n    System paths have to be excluded for efficiency. Non-system paths,\n    such as a project root or ``sys.path.insert``, should be the paths\n    of interest to the user anyway.\n    '
    paths = set()
    for path in chain(list(sys.path), extra_files):
        path = os.path.abspath(path)
        if os.path.isfile(path):
            paths.add(path)
            continue
        parent_has_py = {os.path.dirname(path): True}
        for (root, dirs, files) in os.walk(path):
            if root.startswith(_stat_ignore_scan) or os.path.basename(root) in _ignore_common_dirs:
                dirs.clear()
                continue
            has_py = False
            for name in files:
                if name.endswith(('.py', '.pyc')):
                    has_py = True
                    paths.add(os.path.join(root, name))
            if not (has_py or parent_has_py[os.path.dirname(root)]):
                dirs.clear()
                continue
            parent_has_py[root] = has_py
    paths.update(_iter_module_paths())
    _remove_by_pattern(paths, exclude_patterns)
    return paths

def _find_watchdog_paths(extra_files: set[str], exclude_patterns: set[str]) -> t.Iterable[str]:
    if False:
        print('Hello World!')
    'Find paths for the stat reloader to watch. Looks at the same\n    sources as the stat reloader, but watches everything under\n    directories instead of individual files.\n    '
    dirs = set()
    for name in chain(list(sys.path), extra_files):
        name = os.path.abspath(name)
        if os.path.isfile(name):
            name = os.path.dirname(name)
        dirs.add(name)
    for name in _iter_module_paths():
        dirs.add(os.path.dirname(name))
    _remove_by_pattern(dirs, exclude_patterns)
    return _find_common_roots(dirs)

def _find_common_roots(paths: t.Iterable[str]) -> t.Iterable[str]:
    if False:
        for i in range(10):
            print('nop')
    root: dict[str, dict] = {}
    for chunks in sorted((PurePath(x).parts for x in paths), key=len, reverse=True):
        node = root
        for chunk in chunks:
            node = node.setdefault(chunk, {})
        node.clear()
    rv = set()

    def _walk(node: t.Mapping[str, dict], path: tuple[str, ...]) -> None:
        if False:
            return 10
        for (prefix, child) in node.items():
            _walk(child, path + (prefix,))
        if not node:
            rv.add(os.path.join(*path))
    _walk(root, ())
    return rv

def _get_args_for_reloading() -> list[str]:
    if False:
        while True:
            i = 10
    'Determine how the script was executed, and return the args needed\n    to execute it again in a new process.\n    '
    if sys.version_info >= (3, 10):
        return [sys.executable, *sys.orig_argv[1:]]
    rv = [sys.executable]
    py_script = sys.argv[0]
    args = sys.argv[1:]
    __main__ = sys.modules['__main__']
    if getattr(__main__, '__package__', None) is None or (os.name == 'nt' and __main__.__package__ == '' and (not os.path.exists(py_script)) and os.path.exists(f'{py_script}.exe')):
        py_script = os.path.abspath(py_script)
        if os.name == 'nt':
            if not os.path.exists(py_script) and os.path.exists(f'{py_script}.exe'):
                py_script += '.exe'
            if os.path.splitext(sys.executable)[1] == '.exe' and os.path.splitext(py_script)[1] == '.exe':
                rv.pop(0)
        rv.append(py_script)
    else:
        if os.path.isfile(py_script):
            py_module = t.cast(str, __main__.__package__)
            name = os.path.splitext(os.path.basename(py_script))[0]
            if name != '__main__':
                py_module += f'.{name}'
        else:
            py_module = py_script
        rv.extend(('-m', py_module.lstrip('.')))
    rv.extend(args)
    return rv

class ReloaderLoop:
    name = ''

    def __init__(self, extra_files: t.Iterable[str] | None=None, exclude_patterns: t.Iterable[str] | None=None, interval: int | float=1) -> None:
        if False:
            return 10
        self.extra_files: set[str] = {os.path.abspath(x) for x in extra_files or ()}
        self.exclude_patterns: set[str] = set(exclude_patterns or ())
        self.interval = interval

    def __enter__(self) -> ReloaderLoop:
        if False:
            print('Hello World!')
        'Do any setup, then run one step of the watch to populate the\n        initial filesystem state.\n        '
        self.run_step()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            for i in range(10):
                print('nop')
        'Clean up any resources associated with the reloader.'
        pass

    def run(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Continually run the watch step, sleeping for the configured\n        interval after each step.\n        '
        while True:
            self.run_step()
            time.sleep(self.interval)

    def run_step(self) -> None:
        if False:
            return 10
        'Run one step for watching the filesystem. Called once to set\n        up initial state, then repeatedly to update it.\n        '
        pass

    def restart_with_reloader(self) -> int:
        if False:
            return 10
        'Spawn a new Python interpreter with the same arguments as the\n        current one, but running the reloader thread.\n        '
        while True:
            _log('info', f' * Restarting with {self.name}')
            args = _get_args_for_reloading()
            new_environ = os.environ.copy()
            new_environ['WERKZEUG_RUN_MAIN'] = 'true'
            exit_code = subprocess.call(args, env=new_environ, close_fds=False)
            if exit_code != 3:
                return exit_code

    def trigger_reload(self, filename: str) -> None:
        if False:
            i = 10
            return i + 15
        self.log_reload(filename)
        sys.exit(3)

    def log_reload(self, filename: str) -> None:
        if False:
            return 10
        filename = os.path.abspath(filename)
        _log('info', f' * Detected change in {filename!r}, reloading')

class StatReloaderLoop(ReloaderLoop):
    name = 'stat'

    def __enter__(self) -> ReloaderLoop:
        if False:
            i = 10
            return i + 15
        self.mtimes: dict[str, float] = {}
        return super().__enter__()

    def run_step(self) -> None:
        if False:
            return 10
        for name in _find_stat_paths(self.extra_files, self.exclude_patterns):
            try:
                mtime = os.stat(name).st_mtime
            except OSError:
                continue
            old_time = self.mtimes.get(name)
            if old_time is None:
                self.mtimes[name] = mtime
                continue
            if mtime > old_time:
                self.trigger_reload(name)

class WatchdogReloaderLoop(ReloaderLoop):

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        if False:
            print('Hello World!')
        from watchdog.observers import Observer
        from watchdog.events import PatternMatchingEventHandler
        from watchdog.events import EVENT_TYPE_OPENED
        from watchdog.events import FileModifiedEvent
        super().__init__(*args, **kwargs)
        trigger_reload = self.trigger_reload

        class EventHandler(PatternMatchingEventHandler):

            def on_any_event(self, event: FileModifiedEvent):
                if False:
                    while True:
                        i = 10
                if event.event_type == EVENT_TYPE_OPENED:
                    return
                trigger_reload(event.src_path)
        reloader_name = Observer.__name__.lower()
        if reloader_name.endswith('observer'):
            reloader_name = reloader_name[:-8]
        self.name = f'watchdog ({reloader_name})'
        self.observer = Observer()
        extra_patterns = [p for p in self.extra_files if not os.path.isdir(p)]
        self.event_handler = EventHandler(patterns=['*.py', '*.pyc', '*.zip', *extra_patterns], ignore_patterns=[*[f'*/{d}/*' for d in _ignore_common_dirs], *self.exclude_patterns])
        self.should_reload = False

    def trigger_reload(self, filename: str) -> None:
        if False:
            while True:
                i = 10
        self.should_reload = True
        self.log_reload(filename)

    def __enter__(self) -> ReloaderLoop:
        if False:
            while True:
                i = 10
        self.watches: dict[str, t.Any] = {}
        self.observer.start()
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            i = 10
            return i + 15
        self.observer.stop()
        self.observer.join()

    def run(self) -> None:
        if False:
            i = 10
            return i + 15
        while not self.should_reload:
            self.run_step()
            time.sleep(self.interval)
        sys.exit(3)

    def run_step(self) -> None:
        if False:
            return 10
        to_delete = set(self.watches)
        for path in _find_watchdog_paths(self.extra_files, self.exclude_patterns):
            if path not in self.watches:
                try:
                    self.watches[path] = self.observer.schedule(self.event_handler, path, recursive=True)
                except OSError:
                    self.watches[path] = None
            to_delete.discard(path)
        for path in to_delete:
            watch = self.watches.pop(path, None)
            if watch is not None:
                self.observer.unschedule(watch)
reloader_loops: dict[str, type[ReloaderLoop]] = {'stat': StatReloaderLoop, 'watchdog': WatchdogReloaderLoop}
try:
    __import__('watchdog.observers')
except ImportError:
    reloader_loops['auto'] = reloader_loops['stat']
else:
    reloader_loops['auto'] = reloader_loops['watchdog']

def ensure_echo_on() -> None:
    if False:
        return 10
    'Ensure that echo mode is enabled. Some tools such as PDB disable\n    it which causes usability issues after a reload.'
    if sys.stdin is None or not sys.stdin.isatty():
        return
    try:
        import termios
    except ImportError:
        return
    attributes = termios.tcgetattr(sys.stdin)
    if not attributes[3] & termios.ECHO:
        attributes[3] |= termios.ECHO
        termios.tcsetattr(sys.stdin, termios.TCSANOW, attributes)

def run_with_reloader(main_func: t.Callable[[], None], extra_files: t.Iterable[str] | None=None, exclude_patterns: t.Iterable[str] | None=None, interval: int | float=1, reloader_type: str='auto') -> None:
    if False:
        for i in range(10):
            print('nop')
    'Run the given function in an independent Python interpreter.'
    import signal
    signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
    reloader = reloader_loops[reloader_type](extra_files=extra_files, exclude_patterns=exclude_patterns, interval=interval)
    try:
        if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
            ensure_echo_on()
            t = threading.Thread(target=main_func, args=())
            t.daemon = True
            with reloader:
                t.start()
                reloader.run()
        else:
            sys.exit(reloader.restart_with_reloader())
    except KeyboardInterrupt:
        pass