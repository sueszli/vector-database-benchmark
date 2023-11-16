import pytest
from pytest import raises
from vyper import compiler
from vyper.exceptions import InvalidOperation, InvalidType, SyntaxException, TypeMismatch
fail_list = [('\n@external\ndef foo():\n    x: bool = True\n    x = 5\n    ', InvalidType), ('\n@external\ndef foo():\n    True = 3\n    ', SyntaxException), ('\n@external\ndef foo():\n    x: bool = True\n    x = 129\n    ', InvalidType), ('\n@external\ndef foo() -> bool:\n    return (1 == 2) <= (1 == 1)\n    ', TypeMismatch), '\n@external\ndef foo() -> bool:\n    return (1 == 2) or 3\n    ', '\n@external\ndef foo() -> bool:\n    return 1.0 == 1\n    ', '\n@external\ndef foo() -> bool:\n    a: address = empty(address)\n    return a == 1\n    ', ('\n@external\ndef foo(a: address) -> bool:\n    return not a\n    ', InvalidOperation), ('\n@external\ndef test(a: address) -> bool:\n    assert(a)\n    return True\n    ', TypeMismatch)]

@pytest.mark.parametrize('bad_code', fail_list)
def test_bool_fail(bad_code):
    if False:
        while True:
            i = 10
    if isinstance(bad_code, tuple):
        with raises(bad_code[1]):
            compiler.compile_code(bad_code[0])
    else:
        with raises(TypeMismatch):
            compiler.compile_code(bad_code)
valid_list = ['\n@external\ndef foo():\n    x: bool = True\n    z: bool = x and False\n    ', '\n@external\ndef foo():\n    x: bool = True\n    z: bool = x and False\n    ', '\n@external\ndef foo():\n    x: bool = True\n    x = False\n    ', '\n@external\ndef foo() -> bool:\n    return 1 == 1\n    ', '\n@external\ndef foo() -> bool:\n    return 1 != 1\n    ', '\n@external\ndef foo() -> bool:\n    return 1 > 1\n    ', '\n@external\ndef foo() -> bool:\n    return 2 >= 1\n    ', '\n@external\ndef foo() -> bool:\n    return 1 < 1\n    ', '\n@external\ndef foo() -> bool:\n    return 1 <= 1\n    ', '\n@external\ndef foo2(a: address) -> bool:\n    return a != empty(address)\n    ']

@pytest.mark.parametrize('good_code', valid_list)
def test_bool_success(good_code):
    if False:
        i = 10
        return i + 15
    assert compiler.compile_code(good_code) is not None

@pytest.mark.parametrize('length,value,result', [(1, 'a', False), (1, '', True), (8, 'helloooo', False), (8, 'hello', False), (8, '', True), (40, 'a', False), (40, 'hellohellohellohellohellohellohellohello', False), (40, '', True)])
@pytest.mark.parametrize('op', ['==', '!='])
def test_empty_string_comparison(get_contract_with_gas_estimation, length, value, result, op):
    if False:
        return 10
    contract = f'\n@external\ndef foo(xs: String[{length}]) -> bool:\n    return xs {op} ""\n    '
    c = get_contract_with_gas_estimation(contract)
    if op == '==':
        assert c.foo(value) == result
    elif op == '!=':
        assert c.foo(value) != result

@pytest.mark.parametrize('length,value,result', [(1, b'a', False), (1, b'', True), (8, b'helloooo', False), (8, b'hello', False), (8, b'', True), (40, b'a', False), (40, b'hellohellohellohellohellohellohellohello', False), (40, b'', True)])
@pytest.mark.parametrize('op', ['==', '!='])
def test_empty_bytes_comparison(get_contract_with_gas_estimation, length, value, result, op):
    if False:
        return 10
    contract = f'\n@external\ndef foo(xs: Bytes[{length}]) -> bool:\n    return b"" {op} xs\n    '
    c = get_contract_with_gas_estimation(contract)
    if op == '==':
        assert c.foo(value) == result
    elif op == '!=':
        assert c.foo(value) != result