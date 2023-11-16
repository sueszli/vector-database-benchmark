import pytest
from pytest import raises
from vyper import compiler
from vyper.exceptions import TypeMismatch
fail_list = ['\n@external\ndef foo():\n    a: uint256 = 3\n    b: int128 = 4\n    c: uint256 = min(a, b)\n    ', '\n@external\ndef broken():\n    a : uint256 = 3\n    b : int128 = 4\n    c : uint256 = unsafe_add(a, b)\n    ', '\n@external\ndef foo():\n    b: Bytes[1] = b"\x05"\n    x: uint256 = as_wei_value(b, "babbage")\n    ']

@pytest.mark.parametrize('bad_code', fail_list)
def test_type_mismatch_exception(bad_code):
    if False:
        while True:
            i = 10
    with raises(TypeMismatch):
        compiler.compile_code(bad_code)