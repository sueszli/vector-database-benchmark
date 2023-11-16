import sys
import pytest
from astropy.samp import conf
from astropy.samp.hub_script import hub_script

def setup_module(module):
    if False:
        i = 10
        return i + 15
    conf.use_internet = False

def setup_function(function):
    if False:
        print('Hello World!')
    function.sys_argv_orig = sys.argv
    sys.argv = ['samp_hub']

def teardown_function(function):
    if False:
        while True:
            i = 10
    sys.argv = function.sys_argv_orig

@pytest.mark.slow
def test_hub_script():
    if False:
        for i in range(10):
            print('nop')
    sys.argv.append('-m')
    sys.argv.append('-w')
    hub_script(timeout=3)