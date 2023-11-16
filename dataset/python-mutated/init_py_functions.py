import pytest
from spack.cmd import CommandNameError, PythonNameError, cmd_name, python_name, require_cmd_name, require_python_name

def test_require_python_name():
    if False:
        while True:
            i = 10
    'Python module names should not contain dashes---ensure that\n    require_python_name() raises the appropriate exception if one is\n    detected.\n    '
    require_python_name('okey_dokey')
    with pytest.raises(PythonNameError):
        require_python_name('okey-dokey')
    require_python_name(python_name('okey-dokey'))

def test_require_cmd_name():
    if False:
        print('Hello World!')
    'By convention, Spack command names should contain dashes rather than\n    underscores---ensure that require_cmd_name() raises the appropriate\n    exception if underscores are detected.\n    '
    require_cmd_name('okey-dokey')
    with pytest.raises(CommandNameError):
        require_cmd_name('okey_dokey')
    require_cmd_name(cmd_name('okey_dokey'))