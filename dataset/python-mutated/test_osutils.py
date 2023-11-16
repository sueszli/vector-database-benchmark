import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
from tribler.core.utilities.osutils import dir_copy, fix_filebasename, get_appstate_dir, get_desktop_dir, get_home_dir, get_picture_dir, get_root_state_directory, is_android

def test_fix_filebasename():
    if False:
        i = 10
        return i + 15
    default_name = '_'
    win_name_table = {'abcdef': 'abcdef', '.': default_name, '..': default_name, '': default_name, ' ': default_name, '   ': default_name, os.path.join('a', 'b'): 'a_b', '\\a': '_a', '\x92\x97': '\x92\x97', '\\\\': '__', '\\a\\': '_a_', '/a': '_a', '//': '__', '/a/': '_a_', 'a' * 300: 'a' * 255}
    for c in '"*/:<>?\\|':
        win_name_table[c] = default_name
    linux_name_table = {'abcdef': 'abcdef', '.': default_name, '..': default_name, '': default_name, ' ': default_name, '   ': default_name, os.path.join('a', 'b'): 'a_b', '/a': '_a', '\x92\x97': '\x92\x97', '//': '__', '/a/': '_a_', 'a' * 300: 'a' * 255}
    if sys.platform.startswith('win'):
        name_table = win_name_table
    else:
        name_table = linux_name_table
    for name in name_table:
        fixedname = fix_filebasename(name)
        assert fixedname == name_table[name]

def test_is_android():
    if False:
        return 10
    if sys.platform.startswith('linux') and 'ANDROID_PRIVATE' in os.environ:
        assert is_android()
    else:
        assert not is_android()

def test_home_dir():
    if False:
        for i in range(10):
            print('nop')
    home_dir = get_home_dir()
    assert isinstance(home_dir, Path)
    assert home_dir.is_dir()

def test_appstate_dir():
    if False:
        i = 10
        return i + 15
    appstate_dir = get_appstate_dir()
    assert isinstance(appstate_dir, Path)
    assert appstate_dir.is_dir()

def test_picture_dir():
    if False:
        i = 10
        return i + 15
    picture_dir = get_picture_dir()
    assert isinstance(picture_dir, Path)
    assert picture_dir.is_dir()

def test_desktop_dir():
    if False:
        print('Hello World!')
    desktop_dir = get_desktop_dir()
    assert isinstance(desktop_dir, Path)
    assert desktop_dir.is_dir()

def test_dir_copy(tmpdir):
    if False:
        print('Hello World!')
    '\n    Tests copying a source directory to destination directory.\n    '
    src_dir = os.path.join(tmpdir, 'src')
    src_sub_dirs = ['dir1', 'dir2', 'dir3']
    os.makedirs(src_dir)
    for sub_dir in src_sub_dirs:
        os.makedirs(os.path.join(src_dir, sub_dir))
    dummy_file = 'dummy.txt'
    Path(src_dir, dummy_file).write_text('source: hello world')
    assert len(os.listdir(src_dir)) > 1
    dest_dir1 = os.path.join(tmpdir, 'dest1')
    dest_dir2 = os.path.join(tmpdir, 'dest2')
    os.makedirs(dest_dir2)
    Path(dest_dir2, dummy_file).write_text('dest: hello world')
    assert len(os.listdir(dest_dir2)) == 1
    dir_copy(src_dir, dest_dir1)
    assert len(os.listdir(dest_dir1)) == len(os.listdir(src_dir))
    dir_copy(src_dir, dest_dir2, merge_if_exists=False)
    assert len(os.listdir(dest_dir2)) == 1
    dir_copy(src_dir, dest_dir2, merge_if_exists=True)
    assert len(os.listdir(src_dir)) == len(os.listdir(dest_dir2))
    assert Path(dest_dir2, dummy_file).read_text() == 'source: hello world'

@patch.dict(os.environ, {'TSTATEDIR': '/absolute/path'})
def test_get_root_state_directory_env(tmp_path):
    if False:
        while True:
            i = 10
    (tmp_path / '.Tribler').mkdir()
    with patch.dict(os.environ, {'TSTATEDIR': str(tmp_path)}):
        path = get_root_state_directory()
    assert path == tmp_path

@patch('tribler.core.utilities.osutils.get_appstate_dir')
def test_get_root_state_directory(get_appstate_dir_mock: Mock, tmp_path):
    if False:
        return 10
    get_appstate_dir_mock.return_value = tmp_path
    (tmp_path / '.Tribler').mkdir()
    path = get_root_state_directory()
    assert path.name == '.Tribler'

@patch('tribler.core.utilities.osutils.get_appstate_dir')
def test_get_root_state_directory_does_not_exist(get_appstate_dir_mock: Mock, tmp_path):
    if False:
        i = 10
        return i + 15
    get_appstate_dir_mock.return_value = tmp_path
    with pytest.raises(FileNotFoundError, match='^\\[Errno 2\\] Root directory does not exist:'):
        get_root_state_directory()

@patch('tribler.core.utilities.osutils.get_appstate_dir')
def test_get_root_state_directory_not_a_dir(get_appstate_dir_mock: Mock, tmp_path):
    if False:
        print('Hello World!')
    (tmp_path / 'some_file').write_text('')
    get_appstate_dir_mock.return_value = tmp_path
    with pytest.raises(NotADirectoryError, match='^\\[Errno 20\\] Root state path is not a directory:'):
        get_root_state_directory(home_dir_postfix='some_file')

@patch('tribler.core.utilities.osutils.get_appstate_dir')
def test_get_root_state_directory_create(get_appstate_dir_mock: Mock, tmp_path):
    if False:
        while True:
            i = 10
    get_appstate_dir_mock.return_value = tmp_path
    path = get_root_state_directory(create=True)
    assert path.name == '.Tribler'
    assert path.exists()