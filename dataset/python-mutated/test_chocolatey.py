"""
Test for the chocolatey module
"""
import os
import pytest
import salt.modules.chocolatey as chocolatey
import salt.utils
import salt.utils.platform
from tests.support.mock import MagicMock, patch
pytestmark = [pytest.mark.skipif(not salt.utils.platform.is_windows(), reason='Not a Windows system')]

@pytest.fixture(scope='module')
def choco_path():
    if False:
        for i in range(10):
            print('nop')
    return 'C:\\path\\to\\chocolatey.exe'

@pytest.fixture(scope='module')
def chocolatey_path_pd():
    if False:
        print('Hello World!')
    return os.path.join(os.environ.get('ProgramData'), 'Chocolatey', 'bin', 'chocolatey.exe')

@pytest.fixture(scope='module')
def choco_path_pd():
    if False:
        print('Hello World!')
    return os.path.join(os.environ.get('ProgramData'), 'Chocolatey', 'bin', 'choco.exe')

@pytest.fixture(scope='module')
def choco_path_sd():
    if False:
        print('Hello World!')
    return os.path.join(os.environ.get('SystemDrive'), 'Chocolatey', 'bin', 'chocolatey.bat')

@pytest.fixture(scope='module')
def mock_false():
    if False:
        return 10
    return MagicMock(return_value=False)

@pytest.fixture(scope='module')
def mock_true():
    if False:
        while True:
            i = 10
    return MagicMock(return_value=True)

@pytest.fixture()
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {chocolatey: {'__context__': {}, '__salt__': {}}}

def test__clear_context(choco_path):
    if False:
        while True:
            i = 10
    '\n    Tests _clear_context function\n    '
    context = {'chocolatey._yes': ['--yes'], 'chocolatey._path': choco_path, 'chocolatey._version': '0.9.9'}
    with patch.dict(chocolatey.__context__, context):
        chocolatey._clear_context()
        assert chocolatey.__context__ == {}

def test__yes_context():
    if False:
        i = 10
        return i + 15
    '\n    Tests _yes function when it exists in __context__\n    '
    with patch.dict(chocolatey.__context__, {'chocolatey._yes': ['--yes']}):
        result = chocolatey._yes()
        expected = ['--yes']
        assert result == expected
        assert chocolatey.__context__['chocolatey._yes'] == expected

def test__yes_version_greater():
    if False:
        print('Hello World!')
    '\n    Test _yes when Chocolatey version is greater than 0.9.9\n    '
    mock_version = MagicMock(return_value='10.0.0')
    with patch('salt.modules.chocolatey.chocolatey_version', mock_version):
        result = chocolatey._yes()
        expected = ['--yes']
        assert result == expected
        assert chocolatey.__context__['chocolatey._yes'] == expected

def test__yes_version_less_than():
    if False:
        print('Hello World!')
    '\n    Test _yes when Chocolatey version is less than 0.9.9\n    '
    mock_version = MagicMock(return_value='0.9.0')
    with patch('salt.modules.chocolatey.chocolatey_version', mock_version):
        result = chocolatey._yes()
        expected = []
        assert result == expected
        assert chocolatey.__context__['chocolatey._yes'] == expected

def test__find_chocolatey_context(choco_path):
    if False:
        while True:
            i = 10
    '\n    Test _find_chocolatey when it exists in __context__\n    '
    with patch.dict(chocolatey.__context__, {'chocolatey._path': choco_path}):
        result = chocolatey._find_chocolatey()
        expected = choco_path
        assert result == expected

def test__find_chocolatey_which(choco_path):
    if False:
        return 10
    '\n    Test _find_chocolatey when found with `cmd.which`\n    '
    mock_which = MagicMock(return_value=choco_path)
    with patch.dict(chocolatey.__salt__, {'cmd.which': mock_which}):
        result = chocolatey._find_chocolatey()
        expected = choco_path
        assert result == expected
        assert chocolatey.__context__['chocolatey._path'] == expected

def test__find_chocolatey_programdata(mock_false, mock_true, chocolatey_path_pd):
    if False:
        i = 10
        return i + 15
    '\n    Test _find_chocolatey when found in ProgramData and named chocolatey.exe\n    '
    with patch.dict(chocolatey.__salt__, {'cmd.which': mock_false}), patch('os.path.isfile', mock_true):
        result = chocolatey._find_chocolatey()
        expected = chocolatey_path_pd
        assert result == expected
        assert chocolatey.__context__['chocolatey._path'] == expected

def test__find_choco_programdata(mock_false, choco_path_pd):
    if False:
        while True:
            i = 10
    '\n    Test _find_chocolatey when found in ProgramData and named choco.exe\n    '
    mock_is_file = MagicMock(side_effect=[False, True])
    with patch.dict(chocolatey.__salt__, {'cmd.which': mock_false}), patch('os.path.isfile', mock_is_file):
        result = chocolatey._find_chocolatey()
        expected = choco_path_pd
        assert result == expected
        assert chocolatey.__context__['chocolatey._path'] == expected

def test__find_chocolatey_systemdrive(mock_false, choco_path_sd):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test _find_chocolatey when found on SystemDrive (older versions)\n    '
    with patch.dict(chocolatey.__salt__, {'cmd.which': mock_false}), patch('os.path.isfile', MagicMock(side_effect=[False, False, True])):
        result = chocolatey._find_chocolatey()
        expected = choco_path_sd
        assert result == expected
        assert chocolatey.__context__['chocolatey._path'] == expected

def test_version_check_remote_false():
    if False:
        while True:
            i = 10
    '\n    Test version when remote is False\n    '
    list_return_value = {'ack': ['3.1.1']}
    with patch.object(chocolatey, 'list_', return_value=list_return_value):
        expected = {'ack': {'installed': ['3.1.1']}}
        result = chocolatey.version('ack', check_remote=False)
        assert result == expected

def test_version_check_remote_true():
    if False:
        return 10
    '\n    Test version when remote is True\n    '
    list_side_effect = [{'ack': ['3.1.1']}, {'ack': ['3.1.1'], 'Wolfpack': ['3.0.17'], 'blackbird': ['1.0.79.3']}]
    with patch.object(chocolatey, 'list_', side_effect=list_side_effect):
        expected = {'ack': {'available': ['3.1.1'], 'installed': ['3.1.1']}}
        result = chocolatey.version('ack', check_remote=True)
        assert result == expected

def test_version_check_remote_true_not_available():
    if False:
        while True:
            i = 10
    '\n    Test version when remote is True but remote version is unavailable\n    '
    list_side_effect = [{'ack': ['3.1.1']}, {'Wolfpack': ['3.0.17'], 'blackbird': ['1.0.79.3']}]
    with patch.object(chocolatey, 'list_', side_effect=list_side_effect):
        expected = {'ack': {'installed': ['3.1.1']}}
        result = chocolatey.version('ack', check_remote=True)
        assert result == expected

def test_add_source(choco_path):
    if False:
        while True:
            i = 10
    '\n    Test add_source when remote is False\n    '
    cmd_run_all_mock = MagicMock(return_value={'retcode': 0, 'stdout': 'data'})
    cmd_run_which_mock = MagicMock(return_value=choco_path)
    with patch.dict(chocolatey.__salt__, {'cmd.which': cmd_run_which_mock, 'cmd.run_all': cmd_run_all_mock}):
        expected_call = [choco_path, 'sources', 'add', '--name', 'source_name', '--source', 'source_location']
        result = chocolatey.add_source('source_name', 'source_location')
        cmd_run_all_mock.assert_called_with(expected_call, python_shell=False)
        expected_call = [choco_path, 'sources', 'add', '--name', 'source_name', '--source', 'source_location', '--priority', 'priority']
        result = chocolatey.add_source('source_name', 'source_location', priority='priority')
        cmd_run_all_mock.assert_called_with(expected_call, python_shell=False)

def test_list_pre_2_0_0():
    if False:
        for i in range(10):
            print('nop')
    mock_version = MagicMock(return_value='1.2.1')
    mock_find = MagicMock(return_value=choco_path)
    mock_run = MagicMock(return_value={'stdout': 'No packages', 'retcode': 0})
    with patch.object(chocolatey, 'chocolatey_version', mock_version), patch.object(chocolatey, '_find_chocolatey', mock_find), patch.dict(chocolatey.__salt__, {'cmd.run_all': mock_run}):
        chocolatey.list_()
        expected_call = [choco_path, 'list', '--limit-output']
        mock_run.assert_called_with(expected_call, python_shell=False)

def test_list_post_2_0_0():
    if False:
        return 10
    mock_version = MagicMock(return_value='2.0.1')
    mock_find = MagicMock(return_value=choco_path)
    mock_run = MagicMock(return_value={'stdout': 'No packages', 'retcode': 0})
    with patch.object(chocolatey, 'chocolatey_version', mock_version), patch.object(chocolatey, '_find_chocolatey', mock_find), patch.dict(chocolatey.__salt__, {'cmd.run_all': mock_run}):
        chocolatey.list_()
        expected_call = [choco_path, 'search', '--limit-output']
        mock_run.assert_called_with(expected_call, python_shell=False)

def test_list_webpi_pre_2_0_0():
    if False:
        while True:
            i = 10
    mock_version = MagicMock(return_value='1.2.1')
    mock_find = MagicMock(return_value=choco_path)
    mock_run = MagicMock(return_value={'stdout': 'No packages', 'retcode': 0})
    with patch.object(chocolatey, 'chocolatey_version', mock_version), patch.object(chocolatey, '_find_chocolatey', mock_find), patch.dict(chocolatey.__salt__, {'cmd.run_all': mock_run}):
        chocolatey.list_webpi()
        expected_call = [choco_path, 'list', '--source', 'webpi']
        mock_run.assert_called_with(expected_call, python_shell=False)

def test_list_webpi_post_2_0_0():
    if False:
        while True:
            i = 10
    mock_version = MagicMock(return_value='2.0.1')
    mock_find = MagicMock(return_value=choco_path)
    mock_run = MagicMock(return_value={'stdout': 'No packages', 'retcode': 0})
    with patch.object(chocolatey, 'chocolatey_version', mock_version), patch.object(chocolatey, '_find_chocolatey', mock_find), patch.dict(chocolatey.__salt__, {'cmd.run_all': mock_run}):
        chocolatey.list_webpi()
        expected_call = [choco_path, 'search', '--source', 'webpi']
        mock_run.assert_called_with(expected_call, python_shell=False)

def test_list_windowsfeatures_pre_2_0_0():
    if False:
        i = 10
        return i + 15
    mock_version = MagicMock(return_value='1.2.1')
    mock_find = MagicMock(return_value=choco_path)
    mock_run = MagicMock(return_value={'stdout': 'No packages', 'retcode': 0})
    with patch.object(chocolatey, 'chocolatey_version', mock_version), patch.object(chocolatey, '_find_chocolatey', mock_find), patch.dict(chocolatey.__salt__, {'cmd.run_all': mock_run}):
        chocolatey.list_windowsfeatures()
        expected_call = [choco_path, 'list', '--source', 'windowsfeatures']
        mock_run.assert_called_with(expected_call, python_shell=False)

def test_list_windowsfeatures_post_2_0_0():
    if False:
        return 10
    mock_version = MagicMock(return_value='2.0.1')
    mock_find = MagicMock(return_value=choco_path)
    mock_run = MagicMock(return_value={'stdout': 'No packages', 'retcode': 0})
    with patch.object(chocolatey, 'chocolatey_version', mock_version), patch.object(chocolatey, '_find_chocolatey', mock_find), patch.dict(chocolatey.__salt__, {'cmd.run_all': mock_run}):
        chocolatey.list_windowsfeatures()
        expected_call = [choco_path, 'search', '--source', 'windowsfeatures']
        mock_run.assert_called_with(expected_call, python_shell=False)