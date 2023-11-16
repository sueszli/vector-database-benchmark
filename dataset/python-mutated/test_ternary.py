import pytest
from vyper.compiler import compile_code
from vyper.exceptions import InvalidType, TypeMismatch
good_list = ['\n@external\ndef foo(a: uint256, b: uint256) -> uint256:\n    return a if a > b else b\n    ', '\n@external\ndef foo():\n    a: bool = (True if True else True) or True\n    ', '\nb: uint256\n\n@external\ndef foo(x: uint256) -> uint256:\n    return x if x > self.b else self.b\n    ', '\n@external\ndef foo(x: uint256, t: bool) -> uint256:\n    return x if t else 1\n    ', '\n@external\ndef foo(x: uint256) -> uint256:\n    return x if True else 1\n    ', '\n@external\ndef foo(x: uint256) -> uint256:\n    return x if False else 1\n    ', '\n@external\ndef foo(t: bool) -> DynArray[uint256, 1]:\n    return [2] if t else [1]\n    ', '\n@external\ndef foo(t: bool) -> (uint256, uint256):\n    a: uint256 = 0\n    b: uint256 = 1\n    return (a, b) if t else (b, a)\n    ']

@pytest.mark.parametrize('code', good_list)
def test_ternary_good(code):
    if False:
        for i in range(10):
            print('nop')
    assert compile_code(code) is not None
fail_list = [('\n@external\ndef foo() -> uint256:\n    return 1 if 1 else 2\n    ', InvalidType), ('\nTEST: constant(uint256) = 1\n@external\ndef foo() -> uint256:\n    return 1 if TEST else 2\n    ', InvalidType), ('\nTEST: constant(uint256) = 1\n@external\ndef foo(t: uint256) -> uint256:\n    return 1 if t else 2\n    ', TypeMismatch), ('\n@external\ndef foo() -> uint256:\n    return 1 if True else 2.0\n    ', TypeMismatch), ('\nT: constant(uint256) = 1\n@external\ndef foo() -> uint256:\n    return T if True else 2.0\n    ', TypeMismatch), ('\n@external\ndef foo(x: uint256, y: uint8) -> uint256:\n    return x if True else y\n    ', TypeMismatch), ('\n@external\ndef foo(a: uint256, b: uint256, c: uint256) -> (uint256, uint256):\n    return (a, b) if True else (a, b, c)\n    ', TypeMismatch), ('\n@external\ndef foo(a: uint256, b: uint256, c: uint256) -> (uint256, uint256):\n    return (a, b, c) if True else (a, b)\n    ', TypeMismatch)]

@pytest.mark.parametrize('code,exc', fail_list)
def test_functions_call_fail(code, exc):
    if False:
        return 10
    with pytest.raises(exc):
        compile_code(code)