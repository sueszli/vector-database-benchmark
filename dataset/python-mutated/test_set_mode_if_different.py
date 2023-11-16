from __future__ import annotations
import errno
import os
from itertools import product
try:
    import builtins
except ImportError:
    import __builtin__ as builtins
import pytest
SYNONYMS_0660 = (432, '0o660', '660', 'u+rw-x,g+rw-x,o-rwx', 'u=rw,g=rw,o-rwx')

@pytest.fixture
def mock_stats(mocker):
    if False:
        while True:
            i = 10
    mock_stat1 = mocker.MagicMock()
    mock_stat1.st_mode = 292
    mock_stat2 = mocker.MagicMock()
    mock_stat2.st_mode = 432
    yield {'before': mock_stat1, 'after': mock_stat2}

@pytest.fixture
def am_check_mode(am):
    if False:
        return 10
    am.check_mode = True
    yield am
    am.check_mode = False

@pytest.fixture
def mock_lchmod(mocker):
    if False:
        while True:
            i = 10
    m_lchmod = mocker.patch('ansible.module_utils.basic.os.lchmod', return_value=None, create=True)
    yield m_lchmod

@pytest.mark.parametrize('previous_changes, check_mode, exists, stdin', product((True, False), (True, False), (True, False), ({},)), indirect=['stdin'])
def test_no_mode_given_returns_previous_changes(am, mock_stats, mock_lchmod, mocker, previous_changes, check_mode, exists):
    if False:
        while True:
            i = 10
    am.check_mode = check_mode
    mocker.patch('os.lstat', side_effect=[mock_stats['before']])
    m_lchmod = mocker.patch('os.lchmod', return_value=None, create=True)
    m_path_exists = mocker.patch('os.path.exists', return_value=exists)
    assert am.set_mode_if_different('/path/to/file', None, previous_changes) == previous_changes
    assert not m_lchmod.called
    assert not m_path_exists.called

@pytest.mark.parametrize('mode, check_mode, stdin', product(SYNONYMS_0660, (True, False), ({},)), indirect=['stdin'])
def test_mode_changed_to_0660(am, mock_stats, mocker, mode, check_mode):
    if False:
        while True:
            i = 10
    am.check_mode = check_mode
    mocker.patch('os.lstat', side_effect=[mock_stats['before'], mock_stats['after'], mock_stats['after']])
    m_lchmod = mocker.patch('os.lchmod', return_value=None, create=True)
    mocker.patch('os.path.exists', return_value=True)
    assert am.set_mode_if_different('/path/to/file', mode, False)
    if check_mode:
        assert not m_lchmod.called
    else:
        m_lchmod.assert_called_with(b'/path/to/file', 432)

@pytest.mark.parametrize('mode, check_mode, stdin', product(SYNONYMS_0660, (True, False), ({},)), indirect=['stdin'])
def test_mode_unchanged_when_already_0660(am, mock_stats, mocker, mode, check_mode):
    if False:
        for i in range(10):
            print('nop')
    am.check_mode = check_mode
    mocker.patch('os.lstat', side_effect=[mock_stats['after'], mock_stats['after'], mock_stats['after']])
    m_lchmod = mocker.patch('os.lchmod', return_value=None, create=True)
    mocker.patch('os.path.exists', return_value=True)
    assert not am.set_mode_if_different('/path/to/file', mode, False)
    assert not m_lchmod.called

@pytest.mark.parametrize('mode, stdin', product(SYNONYMS_0660, ({},)), indirect=['stdin'])
def test_mode_changed_to_0660_check_mode_no_file(am, mocker, mode):
    if False:
        for i in range(10):
            print('nop')
    am.check_mode = True
    mocker.patch('os.path.exists', return_value=False)
    assert am.set_mode_if_different('/path/to/file', mode, False)

@pytest.mark.parametrize('check_mode, stdin', product((True, False), ({},)), indirect=['stdin'])
def test_missing_lchmod_is_not_link(am, mock_stats, mocker, monkeypatch, check_mode):
    if False:
        print('Hello World!')
    'Some platforms have lchmod (*BSD) others do not (Linux)'
    am.check_mode = check_mode
    original_hasattr = hasattr
    monkeypatch.delattr(os, 'lchmod', raising=False)
    mocker.patch('os.lstat', side_effect=[mock_stats['before'], mock_stats['after']])
    mocker.patch('os.path.islink', return_value=False)
    mocker.patch('os.path.exists', return_value=True)
    m_chmod = mocker.patch('os.chmod', return_value=None)
    assert am.set_mode_if_different('/path/to/file/no_lchmod', 432, False)
    if check_mode:
        assert not m_chmod.called
    else:
        m_chmod.assert_called_with(b'/path/to/file/no_lchmod', 432)

@pytest.mark.parametrize('check_mode, stdin', product((True, False), ({},)), indirect=['stdin'])
def test_missing_lchmod_is_link(am, mock_stats, mocker, monkeypatch, check_mode):
    if False:
        i = 10
        return i + 15
    'Some platforms have lchmod (*BSD) others do not (Linux)'
    am.check_mode = check_mode
    original_hasattr = hasattr
    monkeypatch.delattr(os, 'lchmod', raising=False)
    mocker.patch('os.lstat', side_effect=[mock_stats['before'], mock_stats['after']])
    mocker.patch('os.path.islink', return_value=True)
    mocker.patch('os.path.exists', return_value=True)
    m_chmod = mocker.patch('os.chmod', return_value=None)
    mocker.patch('os.stat', return_value=mock_stats['after'])
    assert am.set_mode_if_different('/path/to/file/no_lchmod', 432, False)
    if check_mode:
        assert not m_chmod.called
    else:
        m_chmod.assert_called_with(b'/path/to/file/no_lchmod', 432)
    mocker.resetall()
    mocker.stopall()

@pytest.mark.parametrize('stdin,', ({},), indirect=['stdin'])
def test_missing_lchmod_is_link_in_sticky_dir(am, mock_stats, mocker):
    if False:
        return 10
    'Some platforms have lchmod (*BSD) others do not (Linux)'
    am.check_mode = False
    original_hasattr = hasattr

    def _hasattr(obj, name):
        if False:
            for i in range(10):
                print('nop')
        if obj == os and name == 'lchmod':
            return False
        return original_hasattr(obj, name)
    mock_lstat = mocker.MagicMock()
    mock_lstat.st_mode = 511
    mocker.patch('os.lstat', side_effect=[mock_lstat, mock_lstat])
    mocker.patch.object(builtins, 'hasattr', side_effect=_hasattr)
    mocker.patch('os.path.islink', return_value=True)
    mocker.patch('os.path.exists', return_value=True)
    m_stat = mocker.patch('os.stat', side_effect=OSError(errno.EACCES, 'Permission denied'))
    m_chmod = mocker.patch('os.chmod', return_value=None)
    assert not am.set_mode_if_different('/path/to/file/no_lchmod', 432, False)
    m_stat.assert_called_with(b'/path/to/file/no_lchmod')
    m_chmod.assert_not_called()
    mocker.resetall()
    mocker.stopall()