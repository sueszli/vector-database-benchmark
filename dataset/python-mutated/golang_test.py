from __future__ import annotations
from unittest import mock
import pytest
import re_assert
import pre_commit.constants as C
from pre_commit import lang_base
from pre_commit.envcontext import envcontext
from pre_commit.languages import golang
from pre_commit.store import _make_local_repo
from testing.language_helpers import run_language
ACTUAL_GET_DEFAULT_VERSION = golang.get_default_version.__wrapped__

@pytest.fixture
def exe_exists_mck():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(lang_base, 'exe_exists') as mck:
        yield mck

def test_golang_default_version_system_available(exe_exists_mck):
    if False:
        print('Hello World!')
    exe_exists_mck.return_value = True
    assert ACTUAL_GET_DEFAULT_VERSION() == 'system'

def test_golang_default_version_system_not_available(exe_exists_mck):
    if False:
        print('Hello World!')
    exe_exists_mck.return_value = False
    assert ACTUAL_GET_DEFAULT_VERSION() == C.DEFAULT
ACTUAL_INFER_GO_VERSION = golang._infer_go_version.__wrapped__

def test_golang_infer_go_version_not_default():
    if False:
        print('Hello World!')
    assert ACTUAL_INFER_GO_VERSION('1.19.4') == '1.19.4'

def test_golang_infer_go_version_default():
    if False:
        while True:
            i = 10
    version = ACTUAL_INFER_GO_VERSION(C.DEFAULT)
    assert version != C.DEFAULT
    re_assert.Matches('^\\d+\\.\\d+(?:\\.\\d+)?$').assert_matches(version)

def _make_hello_world(tmp_path):
    if False:
        i = 10
        return i + 15
    go_mod = 'module golang-hello-world\n\ngo 1.18\n\nrequire github.com/BurntSushi/toml v1.1.0\n'
    go_sum = 'github.com/BurntSushi/toml v1.1.0 h1:ksErzDEI1khOiGPgpwuI7x2ebx/uXQNw7xJpn9Eq1+I=\ngithub.com/BurntSushi/toml v1.1.0/go.mod h1:CxXYINrC8qIiEnFrOxCa7Jy5BFHlXnUU2pbicEuybxQ=\n'
    hello_world_go = 'package main\n\n\nimport (\n        "fmt"\n        "github.com/BurntSushi/toml"\n)\n\ntype Config struct {\n        What string\n}\n\nfunc main() {\n        var conf Config\n        toml.Decode("What = \'world\'\\n", &conf)\n        fmt.Printf("hello %v\\n", conf.What)\n}\n'
    tmp_path.joinpath('go.mod').write_text(go_mod)
    tmp_path.joinpath('go.sum').write_text(go_sum)
    mod_dir = tmp_path.joinpath('golang-hello-world')
    mod_dir.mkdir()
    main_file = mod_dir.joinpath('main.go')
    main_file.write_text(hello_world_go)

def test_golang_system(tmp_path):
    if False:
        return 10
    _make_hello_world(tmp_path)
    ret = run_language(tmp_path, golang, 'golang-hello-world')
    assert ret == (0, b'hello world\n')

def test_golang_default_version(tmp_path):
    if False:
        return 10
    _make_hello_world(tmp_path)
    ret = run_language(tmp_path, golang, 'golang-hello-world', version=C.DEFAULT)
    assert ret == (0, b'hello world\n')

def test_golang_versioned(tmp_path):
    if False:
        return 10
    _make_local_repo(str(tmp_path))
    (ret, out) = run_language(tmp_path, golang, 'go version', version='1.21.1')
    assert ret == 0
    assert out.startswith(b'go version go1.21.1')

def test_local_golang_additional_deps(tmp_path):
    if False:
        while True:
            i = 10
    _make_local_repo(str(tmp_path))
    ret = run_language(tmp_path, golang, 'hello', deps=('golang.org/x/example/hello@latest',))
    assert ret == (0, b'Hello, world!\n')

def test_golang_hook_still_works_when_gobin_is_set(tmp_path):
    if False:
        while True:
            i = 10
    with envcontext((('GOBIN', str(tmp_path.joinpath('gobin'))),)):
        test_golang_system(tmp_path)