import importlib
import types

class LazyLoader(types.ModuleType):
    """Lazily import a module, mainly to avoid pulling in large dependencies.

    `contrib`, and `ffmpeg` are examples of modules that are large and not always
    needed, and this allows them to only be loaded when they are used.

    Copied from: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/lazy_loader.py
    """

    def __init__(self, local_name, parent_module_globals, name):
        if False:
            while True:
                i = 10
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        super().__init__(name)

    def _load(self):
        if False:
            print('Hello World!')
        module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item):
        if False:
            return 10
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        if False:
            while True:
                i = 10
        module = self._load()
        return dir(module)