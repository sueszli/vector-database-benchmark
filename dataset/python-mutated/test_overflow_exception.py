import pytest
from vyper import compiler
from vyper.exceptions import OverflowException
fail_list = ['\n@external\ndef foo():\n    x: int256 = -57896044618658097711785492504343953926634992332820282019728792003956564819969 # -2**255 - 1  # noqa: E501\n    ', '\n@external\ndef foo():\n    x: decimal = 18707220957835557353007165858768422651595.9365500928\n    ', '\n@external\ndef foo():\n    x: decimal = -18707220957835557353007165858768422651595.9365500929\n    ', '\n@external\ndef foo():\n    x: uint256 = convert(821649876217461872458712528745872158745214187264875632587324658732648753245328764872135671285218762145, uint256)  # noqa: E501\n    ', '\n@external\ndef overflow2() -> uint256:\n    a: uint256 = 2**256\n    return a\n    ']

@pytest.mark.parametrize('bad_code', fail_list)
def test_invalid_literal_exception(bad_code):
    if False:
        i = 10
        return i + 15
    with pytest.raises(OverflowException):
        compiler.compile_code(bad_code)