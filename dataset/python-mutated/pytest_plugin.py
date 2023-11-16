"""Pytest plugin for testing xsh files."""
import sys
import importlib
from traceback import format_list, extract_tb
import pytest
from xonsh.main import setup

def pytest_configure(config):
    if False:
        while True:
            i = 10
    setup()

def pytest_collection_modifyitems(items):
    if False:
        print('Hello World!')
    items.sort(key=lambda x: 0 if isinstance(x, XshFunction) else 1)

def _limited_traceback(excinfo):
    if False:
        i = 10
        return i + 15
    ' Return a formatted traceback with all the stack\n        from this frame (i.e __file__) up removed\n    '
    tb = extract_tb(excinfo.tb)
    try:
        idx = [__file__ in e for e in tb].index(True)
        return format_list(tb[idx + 1:])
    except ValueError:
        return format_list(tb)

def pytest_collect_file(parent, path):
    if False:
        while True:
            i = 10
    if path.ext.lower() == '.xsh' and path.basename.startswith('test_'):
        return XshFile(path, parent)

class XshFile(pytest.File):

    def collect(self):
        if False:
            return 10
        sys.path.append(self.fspath.dirname)
        mod = importlib.import_module(self.fspath.purebasename)
        sys.path.pop(0)
        tests = [t for t in dir(mod) if t.startswith('test_')]
        for test_name in tests:
            obj = getattr(mod, test_name)
            if hasattr(obj, '__call__'):
                yield XshFunction(name=test_name, parent=self, test_func=obj, test_module=mod)

class XshFunction(pytest.Item):

    def __init__(self, name, parent, test_func, test_module):
        if False:
            i = 10
            return i + 15
        super().__init__(name, parent)
        self._test_func = test_func
        self._test_module = test_module

    def runtest(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._test_func(*args, **kwargs)

    def repr_failure(self, excinfo):
        if False:
            for i in range(10):
                print('nop')
        ' called when self.runtest() raises an exception. '
        formatted_tb = _limited_traceback(excinfo)
        formatted_tb.insert(0, 'xonsh execution failed\n')
        formatted_tb.append('{}: {}'.format(excinfo.type.__name__, excinfo.value))
        return ''.join(formatted_tb)

    def reportinfo(self):
        if False:
            while True:
                i = 10
        return (self.fspath, 0, 'xonsh test: {}'.format(self.name))