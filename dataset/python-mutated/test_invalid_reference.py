import pytest
from vyper import compiler
from vyper.exceptions import InvalidReference
fail_list = ['\nx: uint256\n\n@external\ndef foo():\n    send(0x1234567890123456789012345678901234567890, x)\n    ', '\n@external\ndef bar(x: int128) -> int128:\n    return 3 * x\n\n@external\ndef foo() -> int128:\n    return bar(20)\n    ', '\nb: int128\n@external\ndef foo():\n    b = 7\n    ', '\nx: int128\n@external\ndef foo():\n    x = 5\n    ', '\n@external\ndef foo():\n    int128 = 5\n    ', '\na: public(constant(uint256)) = 1\n\n@external\ndef foo():\n    b: uint256 = self.a\n    ', '\na: public(immutable(uint256))\n\n@external\ndef __init__():\n    a = 123\n\n@external\ndef foo():\n    b: uint256 = self.a\n    ']

@pytest.mark.parametrize('bad_code', fail_list)
def test_invalid_reference_exception(bad_code):
    if False:
        while True:
            i = 10
    with pytest.raises(InvalidReference):
        compiler.compile_code(bad_code)