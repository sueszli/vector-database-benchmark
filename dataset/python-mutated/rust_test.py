from __future__ import annotations
from unittest import mock
import pytest
import pre_commit.constants as C
from pre_commit import parse_shebang
from pre_commit.languages import rust
from pre_commit.store import _make_local_repo
from testing.language_helpers import run_language
ACTUAL_GET_DEFAULT_VERSION = rust.get_default_version.__wrapped__

@pytest.fixture
def cmd_output_b_mck():
    if False:
        return 10
    with mock.patch.object(rust, 'cmd_output_b') as mck:
        yield mck

def test_sets_system_when_rust_is_available(cmd_output_b_mck):
    if False:
        while True:
            i = 10
    cmd_output_b_mck.return_value = (0, b'', b'')
    assert ACTUAL_GET_DEFAULT_VERSION() == 'system'

def test_uses_default_when_rust_is_not_available(cmd_output_b_mck):
    if False:
        return 10
    cmd_output_b_mck.return_value = (127, b'', b'error: not found')
    assert ACTUAL_GET_DEFAULT_VERSION() == C.DEFAULT

def _make_hello_world(tmp_path):
    if False:
        return 10
    src_dir = tmp_path.joinpath('src')
    src_dir.mkdir()
    src_dir.joinpath('main.rs').write_text('fn main() {\n    println!("Hello, world!");\n}\n')
    tmp_path.joinpath('Cargo.toml').write_text('[package]\nname = "hello_world"\nversion = "0.1.0"\nedition = "2021"\n')

def test_installs_rust_missing_rustup(tmp_path):
    if False:
        i = 10
        return i + 15
    _make_hello_world(tmp_path)
    calls = []
    orig = parse_shebang.find_executable

    def mck(exe, env=None):
        if False:
            return 10
        calls.append(exe)
        if len(calls) == 1:
            assert exe == 'rustup'
            return None
        return orig(exe, env=env)
    with mock.patch.object(parse_shebang, 'find_executable', side_effect=mck):
        ret = run_language(tmp_path, rust, 'hello_world', version='1.56.0')
    assert calls == ['rustup', 'rustup', 'cargo', 'hello_world']
    assert ret == (0, b'Hello, world!\n')

@pytest.mark.parametrize('version', (C.DEFAULT, '1.56.0'))
def test_language_version_with_rustup(tmp_path, version):
    if False:
        return 10
    assert parse_shebang.find_executable('rustup') is not None
    _make_hello_world(tmp_path)
    ret = run_language(tmp_path, rust, 'hello_world', version=version)
    assert ret == (0, b'Hello, world!\n')

@pytest.mark.parametrize('dep', ('cli:shellharden:4.2.0', 'cli:shellharden'))
def test_rust_cli_additional_dependencies(tmp_path, dep):
    if False:
        return 10
    _make_local_repo(str(tmp_path))
    t_sh = tmp_path.joinpath('t.sh')
    t_sh.write_text('echo $hi\n')
    assert rust.get_default_version() == 'system'
    ret = run_language(tmp_path, rust, 'shellharden --transform', deps=(dep,), args=(str(t_sh),))
    assert ret == (0, b'echo "$hi"\n')

def test_run_lib_additional_dependencies(tmp_path):
    if False:
        i = 10
        return i + 15
    _make_hello_world(tmp_path)
    deps = ('shellharden:4.2.0', 'git-version')
    ret = run_language(tmp_path, rust, 'hello_world', deps=deps)
    assert ret == (0, b'Hello, world!\n')
    bin_dir = tmp_path.joinpath('rustenv-system', 'bin')
    assert bin_dir.is_dir()
    assert not bin_dir.joinpath('shellharden').exists()
    assert not bin_dir.joinpath('shellharden.exe').exists()