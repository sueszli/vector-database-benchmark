"""Patches that are applied at runtime to the virtual environment."""
from __future__ import annotations
import os
import sys
from contextlib import suppress
VIRTUALENV_PATCH_FILE = os.path.join(__file__)

def patch_dist(dist):
    if False:
        print('Hello World!')
    "\n    Distutils allows user to configure some arguments via a configuration file:\n    https://docs.python.org/3/install/index.html#distutils-configuration-files.\n\n    Some of this arguments though don't make sense in context of the virtual environment files, let's fix them up.\n    "
    old_parse_config_files = dist.Distribution.parse_config_files

    def parse_config_files(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        result = old_parse_config_files(self, *args, **kwargs)
        install = self.get_option_dict('install')
        if 'prefix' in install:
            install['prefix'] = (VIRTUALENV_PATCH_FILE, os.path.abspath(sys.prefix))
        for base in ('purelib', 'platlib', 'headers', 'scripts', 'data'):
            key = f'install_{base}'
            if key in install:
                install.pop(key, None)
        return result
    dist.Distribution.parse_config_files = parse_config_files
_DISTUTILS_PATCH = ('distutils.dist', 'setuptools.dist')

class _Finder:
    """A meta path finder that allows patching the imported distutils modules."""
    fullname = None
    lock = []

    def find_spec(self, fullname, path, target=None):
        if False:
            for i in range(10):
                print('nop')
        if fullname in _DISTUTILS_PATCH and self.fullname is None:
            if len(self.lock) == 0:
                import threading
                lock = threading.Lock()
                self.lock.append(lock)
            from functools import partial
            from importlib.util import find_spec
            with self.lock[0]:
                self.fullname = fullname
                try:
                    spec = find_spec(fullname, path)
                    if spec is not None:
                        is_new_api = hasattr(spec.loader, 'exec_module')
                        func_name = 'exec_module' if is_new_api else 'load_module'
                        old = getattr(spec.loader, func_name)
                        func = self.exec_module if is_new_api else self.load_module
                        if old is not func:
                            with suppress(AttributeError):
                                setattr(spec.loader, func_name, partial(func, old))
                        return spec
                finally:
                    self.fullname = None
        return None

    @staticmethod
    def exec_module(old, module):
        if False:
            while True:
                i = 10
        old(module)
        if module.__name__ in _DISTUTILS_PATCH:
            patch_dist(module)

    @staticmethod
    def load_module(old, name):
        if False:
            while True:
                i = 10
        module = old(name)
        if module.__name__ in _DISTUTILS_PATCH:
            patch_dist(module)
        return module
sys.meta_path.insert(0, _Finder())