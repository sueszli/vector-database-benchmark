import os
import pytest
from vyper.cli.vyper_compile import _parse_args

@pytest.fixture
def chdir_path(tmp_path):
    if False:
        while True:
            i = 10
    orig_path = os.getcwd()
    yield tmp_path
    os.chdir(orig_path)

def test_paths(chdir_path):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo() -> bool:\n    return True\n'
    bar_path = chdir_path.joinpath('bar.vy')
    with bar_path.open('w') as fp:
        fp.write(code)
    _parse_args([str(bar_path)])
    os.chdir(chdir_path.parent)
    _parse_args([str(bar_path)])
    _parse_args([str(bar_path.relative_to(chdir_path.parent))])