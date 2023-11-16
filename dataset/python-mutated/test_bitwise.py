import pytest
from vyper.compiler import compile_code
from vyper.exceptions import InvalidLiteral, InvalidOperation, TypeMismatch
from vyper.utils import unsigned_to_signed
code = '\n@external\ndef _bitwise_and(x: uint256, y: uint256) -> uint256:\n    return x & y\n\n@external\ndef _bitwise_or(x: uint256, y: uint256) -> uint256:\n    return x | y\n\n@external\ndef _bitwise_xor(x: uint256, y: uint256) -> uint256:\n    return x ^ y\n\n@external\ndef _bitwise_not(x: uint256) -> uint256:\n    return ~x\n\n@external\ndef _shl(x: uint256, y: uint256) -> uint256:\n    return x << y\n\n@external\ndef _shr(x: uint256, y: uint256) -> uint256:\n    return x >> y\n    '

def test_bitwise_opcodes():
    if False:
        for i in range(10):
            print('nop')
    opcodes = compile_code(code, output_formats=['opcodes'])['opcodes']
    assert 'SHL' in opcodes
    assert 'SHR' in opcodes

def test_test_bitwise(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    c = get_contract_with_gas_estimation(code)
    x = 126416208461208640982146408124
    y = 7128468721412412459
    assert c._bitwise_and(x, y) == x & y
    assert c._bitwise_or(x, y) == x | y
    assert c._bitwise_xor(x, y) == x ^ y
    assert c._bitwise_not(x) == 2 ** 256 - 1 - x
    for t in (x, y):
        for s in (0, 1, 3, 255, 256):
            assert c._shr(t, s) == t >> s
            assert c._shl(t, s) == (t << s) % 2 ** 256

def test_signed_shift(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\ndef _sar(x: int256, y: uint256) -> int256:\n    return x >> y\n\n@external\ndef _shl(x: int256, y: uint256) -> int256:\n    return x << y\n    '
    c = get_contract_with_gas_estimation(code)
    x = 126416208461208640982146408124
    y = 7128468721412412459
    cases = [x, y, -x, -y]
    for t in cases:
        for s in (0, 1, 3, 255, 256):
            assert c._sar(t, s) == t >> s
            assert c._shl(t, s) == unsigned_to_signed((t << s) % 2 ** 256, 256)

def test_precedence(get_contract):
    if False:
        return 10
    code = '\n@external\ndef foo(a: uint256, b: uint256, c: uint256) -> (uint256, uint256):\n    return (a | b & c, (a | b) & c)\n\n@external\ndef bar(a: uint256, b: uint256, c: uint256) -> (uint256, uint256):\n    return (a | ~b & c, (a | ~b) & c)\n\n@external\ndef baz(a: uint256, b: uint256, c: uint256) -> (uint256, uint256):\n    return (a + 8 | ~b & c * 2, (a  + 8 | ~b) & c * 2)\n    '
    c = get_contract(code)
    assert tuple(c.foo(1, 6, 14)) == (1 | 6 & 14, (1 | 6) & 14) == (7, 6)
    assert tuple(c.bar(1, 6, 14)) == (1 | ~6 & 14, (1 | ~6) & 14) == (9, 8)
    assert tuple(c.baz(1, 6, 14)) == (1 + 8 | ~6 & 14 * 2, (1 + 8 | ~6) & 14 * 2) == (25, 24)

def test_literals(get_contract):
    if False:
        print('Hello World!')
    code = '\n@external\ndef _shr(x: uint256) -> uint256:\n    return x >> 3\n\n@external\ndef _shl(x: uint256) -> uint256:\n    return x << 3\n    '
    c = get_contract(code)
    assert c._shr(80) == 10
    assert c._shl(80) == 640
fail_list = [('\n@external\ndef foo(x: uint8, y: uint8) -> uint8:\n    return x << y\n    ', InvalidOperation), ('\n@external\ndef foo(x: int8, y: uint8) -> int8:\n    return x << y\n    ', InvalidOperation), ('\n@external\ndef foo(x: uint256, y: int128) -> uint256:\n    return x << y\n    ', TypeMismatch), ('\n@external\ndef foo() -> uint256:\n    return 2 << 257\n    ', InvalidLiteral), ('\n@external\ndef foo() -> uint256:\n    return 2 << -1\n    ', InvalidLiteral), ('\n@external\ndef foo() -> uint256:\n    return 2 << -1\n    ', InvalidLiteral)]

@pytest.mark.parametrize('bad_code,exc', fail_list)
def test_shift_fail(get_contract_with_gas_estimation, bad_code, exc, assert_compile_failed):
    if False:
        while True:
            i = 10
    assert_compile_failed(lambda : get_contract_with_gas_estimation(bad_code), exc)