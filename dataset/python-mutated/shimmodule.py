"""A shim module for deprecated imports
"""
import importlib.abc
import importlib.util
import sys
import types
from importlib import import_module
from .importstring import import_item

class ShimWarning(Warning):
    """A warning to show when a module has moved, and a shim is in its place."""

class ShimImporter(importlib.abc.MetaPathFinder):
    """Import hook for a shim.

    This ensures that submodule imports return the real target module,
    not a clone that will confuse `is` and `isinstance` checks.
    """

    def __init__(self, src, mirror):
        if False:
            while True:
                i = 10
        self.src = src
        self.mirror = mirror

    def _mirror_name(self, fullname):
        if False:
            return 10
        'get the name of the mirrored module'
        return self.mirror + fullname[len(self.src):]

    def find_spec(self, fullname, path, target=None):
        if False:
            i = 10
            return i + 15
        if fullname.startswith(self.src + '.'):
            mirror_name = self._mirror_name(fullname)
            return importlib.util.find_spec(mirror_name)

class ShimModule(types.ModuleType):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        self._mirror = kwargs.pop('mirror')
        src = kwargs.pop('src', None)
        if src:
            kwargs['name'] = src.rsplit('.', 1)[-1]
        super(ShimModule, self).__init__(*args, **kwargs)
        if src:
            sys.meta_path.append(ShimImporter(src=src, mirror=self._mirror))

    @property
    def __path__(self):
        if False:
            return 10
        return []

    @property
    def __spec__(self):
        if False:
            for i in range(10):
                print('nop')
        "Don't produce __spec__ until requested"
        return import_module(self._mirror).__spec__

    def __dir__(self):
        if False:
            print('Hello World!')
        return dir(import_module(self._mirror))

    @property
    def __all__(self):
        if False:
            while True:
                i = 10
        'Ensure __all__ is always defined'
        mod = import_module(self._mirror)
        try:
            return mod.__all__
        except AttributeError:
            return [name for name in dir(mod) if not name.startswith('_')]

    def __getattr__(self, key):
        if False:
            return 10
        name = '%s.%s' % (self._mirror, key)
        try:
            return import_item(name)
        except ImportError as e:
            raise AttributeError(key) from e

    def __repr__(self):
        if False:
            while True:
                i = 10
        try:
            return self.__getattr__('__repr__')()
        except AttributeError:
            return f'<ShimModule for {self._mirror!r}>'