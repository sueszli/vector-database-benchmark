"""
Tests for the Spyder kernel
"""
import os
import pytest
from spyder.config.manager import CONF
from spyder.plugins.ipythonconsole.utils.kernelspec import SpyderKernelSpec
from spyder.py3compat import to_text_string

@pytest.mark.parametrize('default_interpreter', [True, False])
def test_kernel_pypath(tmpdir, default_interpreter):
    if False:
        i = 10
        return i + 15
    '\n    Test that PYTHONPATH and spyder_pythonpath option are properly handled\n    when an external interpreter is used or not.\n\n    Regression test for spyder-ide/spyder#8681.\n    Regression test for spyder-ide/spyder#17511.\n    '
    CONF.set('main_interpreter', 'default', default_interpreter)
    pypath = to_text_string(tmpdir.mkdir('test-pypath'))
    os.environ['PYTHONPATH'] = pypath
    CONF.set('pythonpath_manager', 'spyder_pythonpath', [pypath])
    kernel_spec = SpyderKernelSpec()
    assert 'PYTHONPATH' not in kernel_spec.env
    assert pypath in kernel_spec.env['SPY_PYTHONPATH']
    CONF.set('main_interpreter', 'default', True)
    CONF.set('pythonpath_manager', 'spyder_pythonpath', [])
    del os.environ['PYTHONPATH']

def test_python_interpreter(tmpdir):
    if False:
        while True:
            i = 10
    'Test the validation of the python interpreter.'
    interpreter = str(tmpdir.mkdir('interpreter').join('python'))
    CONF.set('main_interpreter', 'default', False)
    CONF.set('main_interpreter', 'custom', True)
    CONF.set('main_interpreter', 'executable', interpreter)
    kernel_spec = SpyderKernelSpec()
    assert interpreter not in kernel_spec.argv
    assert CONF.get('main_interpreter', 'default')
    assert not CONF.get('main_interpreter', 'custom')
if __name__ == '__main__':
    pytest.main()