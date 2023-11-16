import pytest
from vyper import compiler
from vyper.exceptions import UndeclaredDefinition
fail_list = ['\n@external\ndef test1(b: uint256) -> uint256:\n    a: uint256 = a + b\n    return a\n    ', '\n@external\ndef test2(b: uint256, c: uint256) -> uint256:\n    a: uint256 = a + b + c\n    return a\n    ', '\n@external\ndef test3(b: int128, c: int128) -> int128:\n    a: int128 = - a\n    return a\n    ', '\n@external\ndef test4(b: bool) -> bool:\n    a: bool = b or a\n    return a\n    ', '\n@external\ndef test5(b: bool) -> bool:\n    a: bool = a != b\n    return a\n    ', '\n@external\ndef test6(b:bool, c: bool) -> bool:\n    a: bool = (a and b) and c\n    return a\n    ', '\n@external\ndef foo():\n    throe = 2\n    ', '\n@external\ndef foo():\n    x: int128 = bar(55)\n    ', '\n@external\ndef foo():\n    x = 5\n    x: int128 = 0\n    ', '\n@external\ndef foo():\n    bork = zork\n    ']

@pytest.mark.parametrize('bad_code', fail_list)
def test_undeclared_def_exception(bad_code):
    if False:
        return 10
    with pytest.raises(UndeclaredDefinition):
        compiler.compile_code(bad_code)