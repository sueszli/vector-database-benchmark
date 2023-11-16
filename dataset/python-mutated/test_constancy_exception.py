import pytest
from pytest import raises
from vyper import compiler
from vyper.exceptions import ImmutableViolation, StateAccessViolation

@pytest.mark.parametrize('bad_code', ['\nx: int128\n@external\n@view\ndef foo() -> int128:\n    self.x = 5\n    return 1', '\n@external\n@view\ndef foo() -> int128:\n    send(0x1234567890123456789012345678901234567890, 5)\n    return 1', '\n@external\n@view\ndef foo():\n    selfdestruct(0x1234567890123456789012345678901234567890)', '\nx: int128\ny: int128\n@external\n@view\ndef foo() -> int128:\n    self.y = 9\n    return 5', '\n@external\n@view\ndef foo() -> int128:\n    x: Bytes[4] = raw_call(\n        0x1234567890123456789012345678901234567890, b"cow", max_outsize=4, gas=595757, value=9\n    )\n    return 5', '\n@external\n@view\ndef foo() -> int128:\n    x: address = create_minimal_proxy_to(0x1234567890123456789012345678901234567890, value=9)\n    return 5', '\nglob: int128\n@internal\ndef foo() -> int128:\n    self.glob += 1\n    return 5\n@external\ndef bar():\n    for i in range(self.foo(), self.foo() + 1):\n        pass', '\nglob: int128\n@internal\ndef foo() -> int128:\n    self.glob += 1\n    return 5\n@external\ndef bar():\n    for i in [1,2,3,4,self.foo()]:\n        pass', '\n@external\ndef foo():\n    x: int128 = 5\n    for i in range(x):\n        pass', '\nf:int128\n\n@external\ndef a (x:int128):\n    self.f = 100\n\n@view\n@external\ndef b():\n    self.a(10)'])
def test_statefulness_violations(bad_code):
    if False:
        while True:
            i = 10
    with raises(StateAccessViolation):
        compiler.compile_code(bad_code)

@pytest.mark.parametrize('bad_code', ['\n@external\ndef foo(x: int128):\n    x = 5\n        ', '\n@external\ndef test(a: uint256[4]):\n    a[0] = 1\n        ', '\n@external\ndef test(a: uint256[4][4]):\n    a[0][1] = 1\n        ', '\nstruct Foo:\n    a: DynArray[DynArray[uint256, 2], 2]\n\n@external\ndef foo(f: Foo) -> Foo:\n    f.a[1] = [0, 1]\n    return f\n        '])
def test_immutability_violations(bad_code):
    if False:
        i = 10
        return i + 15
    with raises(ImmutableViolation):
        compiler.compile_code(bad_code)