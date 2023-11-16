"""Tests for the User Module Reloader."""
import os
import sys
import pytest
from spyder_kernels.customize.umr import UserModuleReloader

@pytest.fixture
def user_module(tmpdir):
    if False:
        return 10
    'Create a simple module in tmpdir as an example of a user module.'
    if str(tmpdir) not in sys.path:
        sys.path.append(str(tmpdir))

    def create_module(modname):
        if False:
            i = 10
            return i + 15
        modfile = tmpdir.mkdir(modname).join('bar.py')
        code = '\ndef square(x):\n    return x**2\n        '
        modfile.write(code)
        init_file = tmpdir.join(modname).join('__init__.py')
        init_file.write('#')
    return create_module

def test_umr_run(user_module):
    if False:
        for i in range(10):
            print('nop')
    "Test that UMR's run method is working correctly."
    user_module('foo1')
    os.environ['SPY_UMR_VERBOSE'] = 'True'
    umr = UserModuleReloader()
    from foo1.bar import square
    assert umr.run() == ['foo1', 'foo1.bar']

def test_umr_previous_modules(user_module):
    if False:
        for i in range(10):
            print('nop')
    "Test that UMR's previous_modules is working as expected."
    user_module('foo2')
    umr = UserModuleReloader()
    import foo2
    assert 'IPython' in umr.previous_modules
    assert 'foo2' not in umr.previous_modules

def test_umr_namelist():
    if False:
        return 10
    'Test that the UMR skips modules according to its name.'
    umr = UserModuleReloader()
    assert umr.is_module_in_namelist('tensorflow')
    assert umr.is_module_in_namelist('pytorch')
    assert umr.is_module_in_namelist('spyder_kernels')
    assert not umr.is_module_in_namelist('foo')

def test_umr_reload_modules(user_module):
    if False:
        print('Hello World!')
    'Test that the UMR only tries to reload user modules.'
    user_module('foo3')
    umr = UserModuleReloader()
    import xml
    assert not umr.is_module_reloadable(xml, 'xml')
    import numpy
    assert not umr.is_module_reloadable(numpy, 'numpy')
    import foo3
    assert umr.is_module_reloadable(foo3, 'foo3')