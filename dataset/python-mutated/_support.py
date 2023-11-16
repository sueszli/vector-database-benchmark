import contextlib
import functools
import sys
import threading
import unittest
from test.support.import_helper import import_fresh_module
OS_ENV_LOCK = threading.Lock()
TZPATH_LOCK = threading.Lock()
TZPATH_TEST_LOCK = threading.Lock()

def call_once(f):
    if False:
        for i in range(10):
            print('nop')
    'Decorator that ensures a function is only ever called once.'
    lock = threading.Lock()
    cached = functools.lru_cache(None)(f)

    @functools.wraps(f)
    def inner():
        if False:
            return 10
        with lock:
            return cached()
    return inner

@call_once
def get_modules():
    if False:
        print('Hello World!')
    'Retrieve two copies of zoneinfo: pure Python and C accelerated.\n\n    Because this function manipulates the import system in a way that might\n    be fragile or do unexpected things if it is run many times, it uses a\n    `call_once` decorator to ensure that this is only ever called exactly\n    one time — in other words, when using this function you will only ever\n    get one copy of each module rather than a fresh import each time.\n    '
    import zoneinfo as c_module
    py_module = import_fresh_module('zoneinfo', blocked=['_zoneinfo'])
    return (py_module, c_module)

@contextlib.contextmanager
def set_zoneinfo_module(module):
    if False:
        while True:
            i = 10
    'Make sure sys.modules["zoneinfo"] refers to `module`.\n\n    This is necessary because `pickle` will refuse to serialize\n    an type calling itself `zoneinfo.ZoneInfo` unless `zoneinfo.ZoneInfo`\n    refers to the same object.\n    '
    NOT_PRESENT = object()
    old_zoneinfo = sys.modules.get('zoneinfo', NOT_PRESENT)
    sys.modules['zoneinfo'] = module
    yield
    if old_zoneinfo is not NOT_PRESENT:
        sys.modules['zoneinfo'] = old_zoneinfo
    else:
        sys.modules.pop('zoneinfo')

class ZoneInfoTestBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.klass = cls.module.ZoneInfo
        super().setUpClass()

    @contextlib.contextmanager
    def tzpath_context(self, tzpath, block_tzdata=True, lock=TZPATH_LOCK):
        if False:
            while True:
                i = 10

        def pop_tzdata_modules():
            if False:
                i = 10
                return i + 15
            tzdata_modules = {}
            for modname in list(sys.modules):
                if modname.split('.', 1)[0] != 'tzdata':
                    continue
                tzdata_modules[modname] = sys.modules.pop(modname)
            return tzdata_modules
        with lock:
            if block_tzdata:
                tzdata_modules = pop_tzdata_modules()
                sys.modules['tzdata'] = None
            old_path = self.module.TZPATH
            try:
                self.module.reset_tzpath(tzpath)
                yield
            finally:
                if block_tzdata:
                    sys.modules.pop('tzdata')
                    for (modname, module) in tzdata_modules.items():
                        sys.modules[modname] = module
                self.module.reset_tzpath(old_path)