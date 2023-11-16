"""Basic introspection of modules."""
from __future__ import annotations
import importlib
import inspect
import os
import pkgutil
import queue
import sys
from multiprocessing import Process, Queue
from types import ModuleType

class ModuleProperties:

    def __init__(self, name: str='', file: str | None=None, path: list[str] | None=None, all: list[str] | None=None, is_c_module: bool=False, subpackages: list[str] | None=None) -> None:
        if False:
            return 10
        self.name = name
        self.file = file
        self.path = path
        self.all = all
        self.is_c_module = is_c_module
        self.subpackages = subpackages or []

def is_c_module(module: ModuleType) -> bool:
    if False:
        while True:
            i = 10
    if module.__dict__.get('__file__') is None:
        return True
    return os.path.splitext(module.__dict__['__file__'])[-1] in ['.so', '.pyd', '.dll']

def is_pyc_only(file: str | None) -> bool:
    if False:
        i = 10
        return i + 15
    return bool(file and file.endswith('.pyc') and (not os.path.exists(file[:-1])))

class InspectError(Exception):
    pass

def get_package_properties(package_id: str) -> ModuleProperties:
    if False:
        while True:
            i = 10
    'Use runtime introspection to get information about a module/package.'
    try:
        package = importlib.import_module(package_id)
    except BaseException as e:
        raise InspectError(str(e)) from e
    name = getattr(package, '__name__', package_id)
    file = getattr(package, '__file__', None)
    path: list[str] | None = getattr(package, '__path__', None)
    if not isinstance(path, list):
        path = None
    pkg_all = getattr(package, '__all__', None)
    if pkg_all is not None:
        try:
            pkg_all = list(pkg_all)
        except Exception:
            pkg_all = None
    is_c = is_c_module(package)
    if path is None:
        if is_c:
            subpackages = [package.__name__ + '.' + name for (name, val) in inspect.getmembers(package) if inspect.ismodule(val) and val.__name__ == package.__name__ + '.' + name]
        else:
            subpackages = []
    else:
        all_packages = pkgutil.walk_packages(path, prefix=package.__name__ + '.', onerror=lambda r: None)
        subpackages = [qualified_name for (importer, qualified_name, ispkg) in all_packages]
    return ModuleProperties(name=name, file=file, path=path, all=pkg_all, is_c_module=is_c, subpackages=subpackages)

def worker(tasks: Queue[str], results: Queue[str | ModuleProperties], sys_path: list[str]) -> None:
    if False:
        while True:
            i = 10
    'The main loop of a worker introspection process.'
    sys.path = sys_path
    while True:
        mod = tasks.get()
        try:
            prop = get_package_properties(mod)
        except InspectError as e:
            results.put(str(e))
            continue
        results.put(prop)

class ModuleInspect:
    """Perform runtime introspection of modules in a separate process.

    Reuse the process for multiple modules for efficiency. However, if there is an
    error, retry using a fresh process to avoid cross-contamination of state between
    modules.

    We use a separate process to isolate us from many side effects. For example, the
    import of a module may kill the current process, and we want to recover from that.

    Always use in a with statement for proper clean-up:

      with ModuleInspect() as m:
          p = m.get_package_properties('urllib.parse')
    """

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self._start()

    def _start(self) -> None:
        if False:
            while True:
                i = 10
        self.tasks: Queue[str] = Queue()
        self.results: Queue[ModuleProperties | str] = Queue()
        self.proc = Process(target=worker, args=(self.tasks, self.results, sys.path))
        self.proc.start()
        self.counter = 0

    def close(self) -> None:
        if False:
            print('Hello World!')
        'Free any resources used.'
        self.proc.terminate()

    def get_package_properties(self, package_id: str) -> ModuleProperties:
        if False:
            return 10
        "Return some properties of a module/package using runtime introspection.\n\n        Raise InspectError if the target couldn't be imported.\n        "
        self.tasks.put(package_id)
        res = self._get_from_queue()
        if res is None:
            self._start()
            raise InspectError(f'Process died when importing {package_id!r}')
        if isinstance(res, str):
            if self.counter > 0:
                self.close()
                self._start()
                return self.get_package_properties(package_id)
            raise InspectError(res)
        self.counter += 1
        return res

    def _get_from_queue(self) -> ModuleProperties | str | None:
        if False:
            i = 10
            return i + 15
        'Get value from the queue.\n\n        Return the value read from the queue, or None if the process unexpectedly died.\n        '
        max_iter = 600
        n = 0
        while True:
            if n == max_iter:
                raise RuntimeError('Timeout waiting for subprocess')
            try:
                return self.results.get(timeout=0.05)
            except queue.Empty:
                if not self.proc.is_alive():
                    return None
            n += 1

    def __enter__(self) -> ModuleInspect:
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, *args: object) -> None:
        if False:
            return 10
        self.close()