"""
:codeauthor: Rahul Handay <rahulha@saltstack.com>
"""
import os
import pytest
import salt.modules.win_path as win_path
import salt.utils.stringutils
import salt.utils.win_reg as reg_util
from tests.support.mock import MagicMock, patch
pytestmark = [pytest.mark.windows_whitelisted, pytest.mark.skip_unless_on_windows]
'\nTest cases for salt.modules.win_path.\n'

@pytest.fixture()
def pathsep():
    if False:
        i = 10
        return i + 15
    return ';'

@pytest.fixture
def configure_loader_modules():
    if False:
        for i in range(10):
            print('nop')
    return {win_path: {'__opts__': {'test': False}, '__salt__': {}, '__utils__': {'reg.read_value': reg_util.read_value}}}

def test_get_path():
    if False:
        print('Hello World!')
    '\n    Test to return the system path\n    '
    mock = MagicMock(return_value={'vdata': 'C:\\Salt'})
    with patch.dict(win_path.__utils__, {'reg.read_value': mock}):
        assert win_path.get_path() == ['C:\\Salt']

def test_exists():
    if False:
        i = 10
        return i + 15
    '\n    Test to check if the directory is configured\n    '
    mock = MagicMock(return_value=['C:\\Foo', 'C:\\Bar'])
    with patch.object(win_path, 'get_path', mock):
        assert win_path.exists('C:\\FOO') is True
        assert win_path.exists('c:\\foo') is True
        assert win_path.exists('c:\\mystuff') is False

def test_util_reg():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test to check if registry comes back clean when get_path is called\n    '
    mock = MagicMock(return_value={'vdata': ''})
    with patch.dict(win_path.__utils__, {'reg.read_value': mock}):
        assert win_path.get_path() == []

def test_add(pathsep):
    if False:
        return 10
    '\n    Test to add the directory to the SYSTEM path\n    '
    orig_path = ('C:\\Foo', 'C:\\Bar')

    def _env(path):
        if False:
            i = 10
            return i + 15
        return {'PATH': salt.utils.stringutils.to_str(pathsep.join(path))}

    def _run(name, index=None, retval=True, path=None):
        if False:
            i = 10
            return i + 15
        if path is None:
            path = orig_path
        env = _env(path)
        mock_get = MagicMock(return_value=list(path))
        mock_set = MagicMock(return_value=retval)
        patch_sep = patch.object(win_path, 'PATHSEP', pathsep)
        patch_path = patch.object(win_path, 'get_path', mock_get)
        patch_env = patch.object(os, 'environ', env)
        patch_dict = patch.dict(win_path.__utils__, {'reg.set_value': mock_set})
        patch_rehash = patch.object(win_path, 'rehash', MagicMock(return_value=True))
        with patch_sep, patch_path, patch_env, patch_dict, patch_rehash:
            return (win_path.add(name, index), env, mock_set)

    def _path_matches(path):
        if False:
            while True:
                i = 10
        return salt.utils.stringutils.to_str(pathsep.join(path))
    (ret, env, mock_set) = _run('')
    assert ret is False
    (ret, env, mock_set) = _run('c:\\salt', retval=True)
    new_path = ('C:\\Foo', 'C:\\Bar', 'c:\\salt')
    assert ret is True
    assert env['PATH'] == _path_matches(new_path)
    (ret, env, mock_set) = _run('c:\\salt', retval=False)
    new_path = ('C:\\Foo', 'C:\\Bar', 'c:\\salt')
    assert ret is False
    assert env['PATH'] == _path_matches(new_path)
    (ret, env, mock_set) = _run('c:\\salt', index=1, retval=True)
    new_path = ('C:\\Foo', 'c:\\salt', 'C:\\Bar')
    assert ret is True
    assert env['PATH'] == _path_matches(new_path)
    (ret, env, mock_set) = _run('c:\\salt', index=0, retval=True)
    new_path = ('c:\\salt', 'C:\\Foo', 'C:\\Bar')
    assert ret is True
    assert env['PATH'] == _path_matches(new_path)
    (ret, env, mock_set) = _run('c:\\foo', retval=True)
    assert ret is True
    assert env['PATH'] == _path_matches(orig_path)
    (ret, env, mock_set) = _run('c:\\foo', index=-1, retval=True)
    new_path = ('C:\\Bar', 'c:\\foo')
    assert ret is True
    assert env['PATH'] == _path_matches(new_path)
    (ret, env, mock_set) = _run('c:\\foo', index=-2, retval=True)
    assert ret is True
    assert env['PATH'] == _path_matches(orig_path)
    (ret, env, mock_set) = _run('c:\\foo', index=-5, retval=True)
    assert ret is True
    assert env['PATH'] == _path_matches(orig_path)
    (ret, env, mock_set) = _run('c:\\bar', index=-5, retval=True)
    new_path = ('c:\\bar', 'C:\\Foo')
    assert ret is True
    assert env['PATH'] == _path_matches(new_path)
    (ret, env, mock_set) = _run('c:\\bar', index=-1, retval=True)
    assert ret is True
    assert env['PATH'] == _path_matches(orig_path)
    (ret, env, mock_set) = _run('c:\\foo', index=5, retval=True)
    new_path = ('C:\\Bar', 'c:\\foo')
    assert ret is True
    assert env['PATH'] == _path_matches(new_path)

def test_remove(pathsep):
    if False:
        while True:
            i = 10
    '\n    Test win_path.remove\n    '
    orig_path = ('C:\\Foo', 'C:\\Bar', 'C:\\Baz')

    def _env(path):
        if False:
            return 10
        return {'PATH': salt.utils.stringutils.to_str(pathsep.join(path))}

    def _run(name='c:\\salt', retval=True, path=None):
        if False:
            for i in range(10):
                print('nop')
        if path is None:
            path = orig_path
        env = _env(path)
        mock_get = MagicMock(return_value=list(path))
        mock_set = MagicMock(return_value=retval)
        patch_path_sep = patch.object(win_path, 'PATHSEP', pathsep)
        patch_path = patch.object(win_path, 'get_path', mock_get)
        patch_env = patch.object(os, 'environ', env)
        patch_dict = patch.dict(win_path.__utils__, {'reg.set_value': mock_set})
        patch_rehash = patch.object(win_path, 'rehash', MagicMock(return_value=True))
        with patch_path_sep, patch_path, patch_env, patch_dict, patch_rehash:
            return (win_path.remove(name), env, mock_set)

    def _path_matches(path):
        if False:
            for i in range(10):
                print('nop')
        return salt.utils.stringutils.to_str(pathsep.join(path))
    (ret, env, mock_set) = _run('C:\\Bar', retval=True)
    new_path = ('C:\\Foo', 'C:\\Baz')
    assert ret is True
    assert env['PATH'] == _path_matches(new_path)
    (ret, env, mock_set) = _run('c:\\bar', retval=True)
    new_path = ('C:\\Foo', 'C:\\Baz')
    assert ret is True
    assert env['PATH'] == _path_matches(new_path)
    old_path = orig_path + ('C:\\BAR',)
    (ret, env, mock_set) = _run('c:\\bar', retval=True)
    new_path = ('C:\\Foo', 'C:\\Baz')
    assert ret is True
    assert env['PATH'] == _path_matches(new_path)
    (ret, env, mock_set) = _run('c:\\bar', retval=False)
    new_path = ('C:\\Foo', 'C:\\Baz')
    assert ret is False
    assert env['PATH'] == _path_matches(new_path)
    (ret, env, mock_set) = _run('C:\\NotThere', retval=True)
    assert ret is True
    assert env['PATH'] == _path_matches(orig_path)