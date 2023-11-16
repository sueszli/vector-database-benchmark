from __future__ import annotations
import pytest
from ansible.module_utils.common.process import get_bin_path

def test_get_bin_path(mocker):
    if False:
        i = 10
        return i + 15
    path = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
    mocker.patch.dict('os.environ', {'PATH': path})
    mocker.patch('os.pathsep', ':')
    mocker.patch('os.path.isdir', return_value=False)
    mocker.patch('ansible.module_utils.common.process.is_executable', return_value=True)
    mocker.patch('os.path.exists', side_effect=[False, True])
    assert '/usr/local/bin/notacommand' == get_bin_path('notacommand')

def test_get_path_path_raise_valueerror(mocker):
    if False:
        print('Hello World!')
    mocker.patch.dict('os.environ', {'PATH': ''})
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('os.path.isdir', return_value=False)
    mocker.patch('ansible.module_utils.common.process.is_executable', return_value=True)
    with pytest.raises(ValueError, match='Failed to find required executable "notacommand"'):
        get_bin_path('notacommand')