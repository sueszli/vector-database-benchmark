from __future__ import annotations
import sys
import pytest
from pre_commit.languages import lua
from pre_commit.util import make_executable
from testing.language_helpers import run_language
pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='lua is not supported on windows')

def test_lua(tmp_path):
    if False:
        i = 10
        return i + 15
    rockspec = 'package = "hello"\nversion = "dev-1"\n\nsource = {\n   url = "git+ssh://git@github.com/pre-commit/pre-commit.git"\n}\ndescription = {}\ndependencies = {}\nbuild = {\n    type = "builtin",\n    modules = {},\n    install = {\n        bin = {"bin/hello-world-lua"}\n    },\n}\n'
    hello_world_lua = "#!/usr/bin/env lua\nprint('hello world')\n"
    tmp_path.joinpath('hello-dev-1.rockspec').write_text(rockspec)
    bin_dir = tmp_path.joinpath('bin')
    bin_dir.mkdir()
    bin_file = bin_dir.joinpath('hello-world-lua')
    bin_file.write_text(hello_world_lua)
    make_executable(bin_file)
    expected = (0, b'hello world\n')
    assert run_language(tmp_path, lua, 'hello-world-lua') == expected

def test_lua_additional_dependencies(tmp_path):
    if False:
        while True:
            i = 10
    (ret, out) = run_language(tmp_path, lua, 'luacheck --version', deps=('luacheck',))
    assert ret == 0
    assert out.startswith(b'Luacheck: ')