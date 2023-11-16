import pytest
from vyper import compiler
from vyper.exceptions import NamespaceCollision
fail_list = ['\n@external\ndef foo(int128: int128):\n    pass\n    ', '\n@external\ndef foo():\n    x: int128 = 12\n@external\ndef foo():\n    y: int128 = 12\n    ', '\nfoo: int128\n\n@external\ndef foo():\n    pass\n    ', '\nx: int128\nx: int128\n    ', '\n@external\ndef foo():\n    x: int128 = 0\n    x: int128 = 0\n    ', '\n@external\ndef foo():\n    msg: bool = True\n    ', '\nint128: Bytes[3]\n    ']

@pytest.mark.parametrize('bad_code', fail_list)
def test_insufficient_arguments(bad_code):
    if False:
        i = 10
        return i + 15
    with pytest.raises(NamespaceCollision):
        compiler.compile_code(bad_code)
pass_list = ['\nx: int128\n\n@external\ndef foo(x: int128): pass\n    ', '\nx: int128\n\n@external\ndef foo():\n    x: int128 = 1234\n    ']

@pytest.mark.parametrize('code', pass_list)
def test_valid(code):
    if False:
        return 10
    compiler.compile_code(code)