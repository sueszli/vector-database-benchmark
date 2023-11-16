"""
Topic: sample
Desc : 
"""
import importlib
import sys
from collections import defaultdict
_post_import_hooks = defaultdict(list)

class PostImportFinder:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._skip = set()

    def find_module(self, fullname, path=None):
        if False:
            i = 10
            return i + 15
        if fullname in self._skip:
            return None
        self._skip.add(fullname)
        return PostImportLoader(self)

class PostImportLoader:

    def __init__(self, finder):
        if False:
            print('Hello World!')
        self._finder = finder

    def load_module(self, fullname):
        if False:
            for i in range(10):
                print('nop')
        importlib.import_module(fullname)
        module = sys.modules[fullname]
        for func in _post_import_hooks[fullname]:
            func(module)
        self._finder._skip.remove(fullname)
        return module

def when_imported(fullname):
    if False:
        while True:
            i = 10

    def decorate(func):
        if False:
            while True:
                i = 10
        if fullname in sys.modules:
            func(sys.modules[fullname])
        else:
            _post_import_hooks[fullname].append(func)
        return func
    return decorate
sys.meta_path.insert(0, PostImportFinder())
from functools import wraps

def logged(func):
    if False:
        for i in range(10):
            print('nop')

    @wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            while True:
                i = 10
        print('Calling', func.__name__, args, kwargs)
        return func(*args, **kwargs)
    return wrapper

@when_imported('math')
def add_logging(mod):
    if False:
        print('Hello World!')
    mod.cos = logged(mod.cos)
    mod.sin = logged(mod.sin)