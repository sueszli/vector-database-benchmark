import pytest
from pytest import raises
from vyper import compiler
from vyper.exceptions import ImmutableViolation, StateAccessViolation, StructureException, TypeMismatch
fail_list = [('\n@external\ndef test():\n    a: int128 = 0\n    b: int128 = 0\n    c: int128 = 0\n    a, b, c = 1, 2, 3\n    ', StructureException), '\n@internal\ndef out_literals() -> (int128, int128, Bytes[10]):\n    return 1, 2, b"3333"\n\n@external\ndef test() -> (int128, address, Bytes[10]):\n    a: int128 = 0\n    b: int128 = 0\n    a, b, b = self.out_literals()  # incorrect bytes type\n    return a, b, c\n    ', '\n@internal\ndef out_literals() -> (int128, int128, Bytes[10]):\n    return 1, 2, b"3333"\n\n@external\ndef test() -> (int128, address, Bytes[10]):\n    a: int128 = 0\n    b: address = empty(address)\n    a, b = self.out_literals()  # tuple count mismatch\n    return\n    ', '\n@internal\ndef out_literals() -> (int128, int128, int128):\n    return 1, 2, 3\n\n@external\ndef test() -> (int128, int128, Bytes[10]):\n    a: int128 = 0\n    b: int128 = 0\n    c: Bytes[10] = b""\n    a, b, c = self.out_literals()\n    return a, b, c\n    ', '\n@internal\ndef out_literals() -> (int128, int128, Bytes[100]):\n    return 1, 2, b"test"\n\n@external\ndef test():\n    a: int128 = 0\n    b: int128 = 0\n    c: Bytes[1] = b""\n    a, b, c = self.out_literals()\n    ', ('\n@internal\ndef _test(a: bytes32) -> (bytes32, uint256, int128):\n    b: uint256 = 1000\n    return a, b, -1200\n\n@external\ndef test(a: bytes32) -> (bytes32, uint256, int128):\n    b: uint256 = 1\n    c: int128 = 1\n    d: int128 = 123\n    a, b, c = self._test(a)\n    assert d == 123\n    return a, b, c\n    ', ImmutableViolation), ('\nB: immutable(uint256)\n\n@external\ndef __init__(b: uint256):\n    B = b\n\n@internal\ndef foo() -> (uint256, uint256):\n    return (1, 2)\n\n@external\ndef bar():\n    a: uint256 = 1\n    a, B = self.foo()\n    ', ImmutableViolation), ('\nx: public(uint256)\n\n@internal\n@view\ndef return_two() -> (uint256, uint256):\n    return 1, 2\n\n@external\n@view\ndef foo():\n    a: uint256 = 0\n    a, self.x = self.return_two()\n     ', StateAccessViolation)]

@pytest.mark.parametrize('bad_code', fail_list)
def test_tuple_assign_fail(bad_code):
    if False:
        i = 10
        return i + 15
    if isinstance(bad_code, tuple):
        with raises(bad_code[1]):
            compiler.compile_code(bad_code[0])
    else:
        with raises(TypeMismatch):
            compiler.compile_code(bad_code)