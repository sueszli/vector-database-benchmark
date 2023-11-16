"""Base class of a profiler wrapper."""
import inspect
import multiprocessing
import os
import pkgutil
import sys
import zlib

def get_pkg_module_names(package_path):
    if False:
        while True:
            i = 10
    'Returns module filenames from package.\n\n    Args:\n        package_path: Path to Python package.\n    Returns:\n        A set of module filenames.\n    '
    module_names = set()
    for (fobj, modname, _) in pkgutil.iter_modules(path=[package_path]):
        filename = os.path.join(fobj.path, '%s.py' % modname)
        if os.path.exists(filename):
            module_names.add(os.path.abspath(filename))
    return module_names

def hash_name(name):
    if False:
        i = 10
        return i + 15
    'Computes hash of the name.'
    return zlib.adler32(name.encode('utf-8'))

class ProcessWithException(multiprocessing.Process):
    """Process subclass that propagates exceptions to parent process.

    Also handles sending function output to parent process.
    Args:
        parent_conn: Parent end of multiprocessing.Pipe.
        child_conn: Child end of multiprocessing.Pipe.
        result: Result of the child process.
    """

    def __init__(self, result, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        (self.parent_conn, self.child_conn) = multiprocessing.Pipe()
        self.result = result

    def run(self):
        if False:
            print('Hello World!')
        try:
            self.result.update(self._target(*self._args, **self._kwargs))
            self.child_conn.send(None)
        except Exception as exc:
            self.child_conn.send(exc)

    @property
    def exception(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns exception from child process.'
        return self.parent_conn.recv()

    @property
    def output(self):
        if False:
            i = 10
            return i + 15
        'Returns target function output.'
        return self.result._getvalue()

def run_in_separate_process(func, *args, **kwargs):
    if False:
        while True:
            i = 10
    "Runs function in separate process.\n\n    This function is used instead of a decorator, since Python multiprocessing\n    module can't serialize decorated function on all platforms.\n    "
    manager = multiprocessing.Manager()
    manager_dict = manager.dict()
    process = ProcessWithException(manager_dict, target=func, args=args, kwargs=kwargs)
    process.start()
    process.join()
    exc = process.exception
    if exc:
        raise exc
    return process.output

class BaseProfiler:
    """Base class for a profiler wrapper."""

    def __init__(self, run_object):
        if False:
            for i in range(10):
                print('nop')
        'Initializes profiler.\n\n        Args:\n            run_object: object to be profiled.\n        '
        run_obj_type = self.get_run_object_type(run_object)
        if run_obj_type == 'module':
            self.init_module(run_object)
        elif run_obj_type == 'package':
            self.init_package(run_object)
        else:
            self.init_function(run_object)

    @staticmethod
    def get_run_object_type(run_object):
        if False:
            print('Hello World!')
        'Determines run object type.'
        if isinstance(run_object, tuple):
            return 'function'
        (run_object, _, _) = run_object.partition(' ')
        if os.path.isdir(run_object):
            return 'package'
        return 'module'

    def init_module(self, run_object):
        if False:
            print('Hello World!')
        'Initializes profiler with a module.'
        self.profile = self.profile_module
        (self._run_object, _, self._run_args) = run_object.partition(' ')
        self._object_name = '%s (module)' % self._run_object
        self._globs = {'__file__': self._run_object, '__name__': '__main__', '__package__': None}
        program_path = os.path.dirname(self._run_object)
        if sys.path[0] != program_path:
            sys.path.insert(0, program_path)
        self._replace_sysargs()

    def init_package(self, run_object):
        if False:
            while True:
                i = 10
        'Initializes profiler with a package.'
        self.profile = self.profile_package
        (self._run_object, _, self._run_args) = run_object.partition(' ')
        self._object_name = '%s (package)' % self._run_object
        self._replace_sysargs()

    def init_function(self, run_object):
        if False:
            while True:
                i = 10
        'Initializes profiler with a function.'
        self.profile = self.profile_function
        (self._run_object, self._run_args, self._run_kwargs) = run_object
        filename = inspect.getsourcefile(self._run_object)
        self._object_name = '%s @ %s (function)' % (self._run_object.__name__, filename)

    def _replace_sysargs(self):
        if False:
            for i in range(10):
                print('nop')
        'Replaces sys.argv with proper args to pass to script.'
        sys.argv[:] = [self._run_object]
        if self._run_args:
            sys.argv += self._run_args.split()

    def profile_package(self):
        if False:
            while True:
                i = 10
        'Profiles package specified by filesystem path.\n\n        Runs object self._run_object as a package specified by filesystem path.\n        Must be overridden.\n        '
        raise NotImplementedError

    def profile_module(self):
        if False:
            return 10
        'Profiles a module.\n\n        Runs object self._run_object as a Python module.\n        Must be overridden.\n        '
        raise NotImplementedError

    def profile_function(self):
        if False:
            for i in range(10):
                print('nop')
        'Profiles a function.\n\n        Runs object self._run_object as a Python function.\n        Must be overridden.\n        '
        raise NotImplementedError

    def run(self):
        if False:
            while True:
                i = 10
        'Runs a profiler and returns collected stats.'
        return self.profile()