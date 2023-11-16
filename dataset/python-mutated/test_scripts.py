""" Test scripts

Test that we can run executable scripts that have been installed with numpy.
"""
import sys
import os
import pytest
from os.path import join as pathjoin, isfile, dirname
import subprocess
import numpy as np
from numpy.testing import assert_equal, IS_WASM
is_inplace = isfile(pathjoin(dirname(np.__file__), '..', 'setup.py'))

def find_f2py_commands():
    if False:
        i = 10
        return i + 15
    if sys.platform == 'win32':
        exe_dir = dirname(sys.executable)
        if exe_dir.endswith('Scripts'):
            return [os.path.join(exe_dir, 'f2py')]
        else:
            return [os.path.join(exe_dir, 'Scripts', 'f2py')]
    else:
        version = sys.version_info
        major = str(version.major)
        minor = str(version.minor)
        return ['f2py', 'f2py' + major, 'f2py' + major + '.' + minor]

@pytest.mark.skipif(is_inplace, reason='Cannot test f2py command inplace')
@pytest.mark.xfail(reason='Test is unreliable')
@pytest.mark.parametrize('f2py_cmd', find_f2py_commands())
def test_f2py(f2py_cmd):
    if False:
        for i in range(10):
            print('nop')
    stdout = subprocess.check_output([f2py_cmd, '-v'])
    assert_equal(stdout.strip(), np.__version__.encode('ascii'))

@pytest.mark.skipif(IS_WASM, reason='Cannot start subprocess')
def test_pep338():
    if False:
        while True:
            i = 10
    stdout = subprocess.check_output([sys.executable, '-mnumpy.f2py', '-v'])
    assert_equal(stdout.strip(), np.__version__.encode('ascii'))