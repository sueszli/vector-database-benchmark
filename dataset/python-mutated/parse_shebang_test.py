from __future__ import annotations
import contextlib
import os.path
import shutil
import sys
import pytest
from pre_commit import parse_shebang
from pre_commit.envcontext import envcontext
from pre_commit.envcontext import Var
from pre_commit.util import make_executable

def _echo_exe() -> str:
    if False:
        while True:
            i = 10
    exe = shutil.which('echo')
    assert exe is not None
    return exe

def test_file_doesnt_exist():
    if False:
        while True:
            i = 10
    assert parse_shebang.parse_filename('herp derp derp') == ()

def test_simple_case(tmpdir):
    if False:
        return 10
    x = tmpdir.join('f')
    x.write('#!/usr/bin/env echo')
    make_executable(x.strpath)
    assert parse_shebang.parse_filename(x.strpath) == ('echo',)

def test_find_executable_full_path():
    if False:
        for i in range(10):
            print('nop')
    assert parse_shebang.find_executable(sys.executable) == sys.executable

def test_find_executable_on_path():
    if False:
        for i in range(10):
            print('nop')
    assert parse_shebang.find_executable('echo') == _echo_exe()

def test_find_executable_not_found_none():
    if False:
        return 10
    assert parse_shebang.find_executable('not-a-real-executable') is None

def write_executable(shebang, filename='run'):
    if False:
        for i in range(10):
            print('nop')
    os.mkdir('bin')
    path = os.path.join('bin', filename)
    with open(path, 'w') as f:
        f.write(f'#!{shebang}')
    make_executable(path)
    return path

@contextlib.contextmanager
def bin_on_path():
    if False:
        print('Hello World!')
    bindir = os.path.join(os.getcwd(), 'bin')
    with envcontext((('PATH', (bindir, os.pathsep, Var('PATH'))),)):
        yield

def test_find_executable_path_added(in_tmpdir):
    if False:
        print('Hello World!')
    path = os.path.abspath(write_executable('/usr/bin/env sh'))
    assert parse_shebang.find_executable('run') is None
    with bin_on_path():
        assert parse_shebang.find_executable('run') == path

def test_find_executable_path_ext(in_tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Windows exports PATHEXT as a list of extensions to automatically add\n    to executables when doing PATH searching.\n    '
    exe_path = os.path.abspath(write_executable('/usr/bin/env sh', filename='run.myext'))
    env_path = {'PATH': os.path.dirname(exe_path)}
    env_path_ext = dict(env_path, PATHEXT=os.pathsep.join(('.exe', '.myext')))
    assert parse_shebang.find_executable('run') is None
    assert parse_shebang.find_executable('run', env=env_path) is None
    ret = parse_shebang.find_executable('run.myext', env=env_path)
    assert ret == exe_path
    ret = parse_shebang.find_executable('run', env=env_path_ext)
    assert ret == exe_path

def test_normexe_does_not_exist():
    if False:
        return 10
    with pytest.raises(OSError) as excinfo:
        parse_shebang.normexe('i-dont-exist-lol')
    assert excinfo.value.args == ('Executable `i-dont-exist-lol` not found',)

def test_normexe_does_not_exist_sep():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(OSError) as excinfo:
        parse_shebang.normexe('./i-dont-exist-lol')
    assert excinfo.value.args == ('Executable `./i-dont-exist-lol` not found',)

@pytest.mark.xfail(sys.platform == 'win32', reason='posix only')
def test_normexe_not_executable(tmpdir):
    if False:
        print('Hello World!')
    tmpdir.join('exe').ensure()
    with tmpdir.as_cwd(), pytest.raises(OSError) as excinfo:
        parse_shebang.normexe('./exe')
    assert excinfo.value.args == ('Executable `./exe` is not executable',)

def test_normexe_is_a_directory(tmpdir):
    if False:
        print('Hello World!')
    with tmpdir.as_cwd():
        tmpdir.join('exe').ensure_dir()
        exe = os.path.join('.', 'exe')
        with pytest.raises(OSError) as excinfo:
            parse_shebang.normexe(exe)
        (msg,) = excinfo.value.args
        assert msg == f'Executable `{exe}` is a directory'

def test_normexe_already_full_path():
    if False:
        for i in range(10):
            print('nop')
    assert parse_shebang.normexe(sys.executable) == sys.executable

def test_normexe_gives_full_path():
    if False:
        print('Hello World!')
    assert parse_shebang.normexe('echo') == _echo_exe()
    assert os.sep in _echo_exe()

def test_normalize_cmd_trivial():
    if False:
        i = 10
        return i + 15
    cmd = (_echo_exe(), 'hi')
    assert parse_shebang.normalize_cmd(cmd) == cmd

def test_normalize_cmd_PATH():
    if False:
        print('Hello World!')
    cmd = ('echo', '--version')
    expected = (_echo_exe(), '--version')
    assert parse_shebang.normalize_cmd(cmd) == expected

def test_normalize_cmd_shebang(in_tmpdir):
    if False:
        for i in range(10):
            print('nop')
    us = sys.executable.replace(os.sep, '/')
    path = write_executable(us)
    assert parse_shebang.normalize_cmd((path,)) == (us, path)

def test_normalize_cmd_PATH_shebang_full_path(in_tmpdir):
    if False:
        print('Hello World!')
    us = sys.executable.replace(os.sep, '/')
    path = write_executable(us)
    with bin_on_path():
        ret = parse_shebang.normalize_cmd(('run',))
        assert ret == (us, os.path.abspath(path))

def test_normalize_cmd_PATH_shebang_PATH(in_tmpdir):
    if False:
        return 10
    echo = _echo_exe()
    path = write_executable('/usr/bin/env echo')
    with bin_on_path():
        ret = parse_shebang.normalize_cmd(('run',))
        assert ret == (echo, os.path.abspath(path))