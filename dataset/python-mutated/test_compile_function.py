"""See https://github.com/numpy/numpy/pull/11937.

"""
import sys
import os
import uuid
from importlib import import_module
import pytest
import numpy.f2py
from . import util

def setup_module():
    if False:
        print('Hello World!')
    if not util.has_c_compiler():
        pytest.skip('Needs C compiler')
    if not util.has_f77_compiler():
        pytest.skip('Needs FORTRAN 77 compiler')

@pytest.mark.parametrize('extra_args', [['--noopt', '--debug'], '--noopt --debug', ''])
@pytest.mark.leaks_references(reason='Imported module seems never deleted.')
def test_f2py_init_compile(extra_args):
    if False:
        while True:
            i = 10
    fsource = '\n        integer function foo()\n        foo = 10 + 5\n        return\n        end\n    '
    moddir = util.get_module_dir()
    modname = util.get_temp_module_name()
    cwd = os.getcwd()
    target = os.path.join(moddir, str(uuid.uuid4()) + '.f')
    for source_fn in [target, None]:
        with util.switchdir(moddir):
            ret_val = numpy.f2py.compile(fsource, modulename=modname, extra_args=extra_args, source_fn=source_fn)
            assert ret_val == 0
    if sys.platform != 'win32':
        return_check = import_module(modname)
        calc_result = return_check.foo()
        assert calc_result == 15
        del sys.modules[modname]

def test_f2py_init_compile_failure():
    if False:
        while True:
            i = 10
    ret_val = numpy.f2py.compile(b'invalid')
    assert ret_val == 1

def test_f2py_init_compile_bad_cmd():
    if False:
        for i in range(10):
            print('nop')
    try:
        temp = sys.executable
        sys.executable = 'does not exist'
        ret_val = numpy.f2py.compile(b'invalid')
        assert ret_val == 127
    finally:
        sys.executable = temp

@pytest.mark.parametrize('fsource', ['program test_f2py\nend program test_f2py', b'program test_f2py\nend program test_f2py'])
def test_compile_from_strings(tmpdir, fsource):
    if False:
        return 10
    with util.switchdir(tmpdir):
        ret_val = numpy.f2py.compile(fsource, modulename='test_compile_from_strings', extension='.f90')
        assert ret_val == 0