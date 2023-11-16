"""TensorFlow root package"""
import sys as _sys
import importlib as _importlib
import types as _types

class _LazyLoader(_types.ModuleType):
    """Lazily import a module so that we can forward it."""

    def __init__(self, local_name, parent_module_globals, name):
        if False:
            i = 10
            return i + 15
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        super(_LazyLoader, self).__init__(name)

    def _load(self):
        if False:
            for i in range(10):
                print('nop')
        "Import the target module and insert it into the parent's namespace."
        module = _importlib.import_module(self.__name__)
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
            print('Hello World!')
        module = self._load()
        return dir(module)

    def __reduce__(self):
        if False:
            return 10
        return (__import__, (self.__name__,))

def _forward_module(old_name):
    if False:
        while True:
            i = 10
    parts = old_name.split('.')
    parts[0] = parts[0] + '_core'
    local_name = parts[-1]
    existing_name = '.'.join(parts)
    _module = _LazyLoader(local_name, globals(), existing_name)
    return _sys.modules.setdefault(old_name, _module)
_top_level_modules = ['tensorflow._api', 'tensorflow.python', 'tensorflow.tools', 'tensorflow.core', 'tensorflow.compiler', 'tensorflow.lite', 'tensorflow.keras', 'tensorflow.compat', 'tensorflow.summary', 'tensorflow.examples']
if 'tensorflow_estimator' not in _sys.modules:
    _root_estimator = False
    _top_level_modules.append('tensorflow.estimator')
else:
    _root_estimator = True
for _m in _top_level_modules:
    _forward_module(_m)
from tensorflow_core import *
_major_api_version = 2
try:
    del core
except NameError:
    pass
try:
    del python
except NameError:
    pass
try:
    del compiler
except NameError:
    pass
try:
    del tools
except NameError:
    pass
try:
    del examples
except NameError:
    pass