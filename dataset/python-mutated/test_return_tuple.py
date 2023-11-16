import pytest
from vyper import compiler
from vyper.exceptions import FunctionDeclarationException
pytestmark = pytest.mark.usefixtures('memory_mocker')
fail_list = ['\n@external\ndef unmatched_tupl_length() -> (Bytes[8], int128, Bytes[8]):\n    return "test", 123\n    ']

@pytest.mark.parametrize('bad_code', fail_list)
def test_tuple_return_fail(bad_code):
    if False:
        return 10
    with pytest.raises(FunctionDeclarationException):
        compiler.compile_code(bad_code)

def test_self_call_in_return_tuple(get_contract):
    if False:
        return 10
    code = '\n@internal\ndef _foo() -> uint256:\n    a: uint256[10] = [6,7,8,9,10,11,12,13,14,15]\n    return 3\n\n@external\ndef foo() -> (uint256, uint256, uint256, uint256, uint256):\n    return 1, 2, self._foo(), 4, 5\n    '
    c = get_contract(code)
    assert c.foo() == [1, 2, 3, 4, 5]

def test_call_in_call(get_contract):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@internal\ndef _foo(a: uint256, b: uint256, c: uint256) -> (uint256, uint256, uint256, uint256, uint256):\n    return 1, a, b, c, 5\n\n@internal\ndef _foo2() -> uint256:\n    a: uint256[10] = [6,7,8,9,10,11,12,13,15,16]\n    return 4\n\n@external\ndef foo() -> (uint256, uint256, uint256, uint256, uint256):\n    return self._foo(2, 3, self._foo2())\n    '
    c = get_contract(code)
    assert c.foo() == [1, 2, 3, 4, 5]

def test_nested_calls_in_tuple_return(get_contract):
    if False:
        while True:
            i = 10
    code = '\n@internal\ndef _foo(a: uint256, b: uint256, c: uint256) -> (uint256, uint256):\n    return 415, 3\n\n@internal\ndef _foo2(a: uint256) -> uint256:\n    b: uint256[10] = [6,7,8,9,10,11,12,13,14,15]\n    return 99\n\n@internal\ndef _foo3(a: uint256, b: uint256) -> uint256:\n    c: uint256[10] = [14,15,16,17,18,19,20,21,22,23]\n    return 42\n\n@internal\ndef _foo4() -> uint256:\n    c: uint256[10] = [14,15,16,17,18,19,20,21,22,23]\n    return 4\n\n@external\ndef foo() -> (uint256, uint256, uint256, uint256, uint256):\n    return 1, 2, self._foo(6, 7, self._foo2(self._foo3(9, 11)))[1], self._foo4(), 5\n    '
    c = get_contract(code)
    assert c.foo() == [1, 2, 3, 4, 5]

def test_external_call_in_return_tuple(get_contract):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@view\n@external\ndef foo() -> (uint256, uint256):\n    return 3, 4\n    '
    code2 = '\ninterface Foo:\n    def foo() -> (uint256, uint256): view\n\n@external\ndef foo(a: address) -> (uint256, uint256, uint256, uint256, uint256):\n    return 1, 2, Foo(a).foo()[0], 4, 5\n    '
    c = get_contract(code)
    c2 = get_contract(code2)
    assert c2.foo(c.address) == [1, 2, 3, 4, 5]

def test_nested_external_call_in_return_tuple(get_contract):
    if False:
        i = 10
        return i + 15
    code = '\n@view\n@external\ndef foo() -> (uint256, uint256):\n    return 3, 4\n\n@view\n@external\ndef bar(a: uint256) -> uint256:\n    return a+1\n    '
    code2 = '\ninterface Foo:\n    def foo() -> (uint256, uint256): view\n    def bar(a: uint256) -> uint256: view\n\n@external\ndef foo(a: address) -> (uint256, uint256, uint256, uint256, uint256):\n    return 1, 2, Foo(a).foo()[0], 4, Foo(a).bar(Foo(a).foo()[1])\n    '
    c = get_contract(code)
    c2 = get_contract(code2)
    assert c2.foo(c.address) == [1, 2, 3, 4, 5]

def test_single_type_tuple_int(get_contract):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@view\n@external\ndef foo() -> (uint256[3], uint256, uint256[2][2]):\n    return [1,2,3], 4, [[5,6], [7,8]]\n\n@view\n@external\ndef foo2(a: int128, b: int128) -> (int128[5], int128, int128[2]):\n    return [1,2,3,a,5], b, [7,8]\n    '
    c = get_contract(code)
    assert c.foo() == [[1, 2, 3], 4, [[5, 6], [7, 8]]]
    assert c.foo2(4, 6) == [[1, 2, 3, 4, 5], 6, [7, 8]]

def test_single_type_tuple_address(get_contract):
    if False:
        i = 10
        return i + 15
    code = '\n@view\n@external\ndef foo() -> (address, address[2]):\n    return (\n        self,\n        [0xF5D4020dCA6a62bB1efFcC9212AAF3c9819E30D7, 0xFFfFfFffFFfffFFfFFfFFFFFffFFFffffFfFFFfF]\n    )\n    '
    c = get_contract(code)
    assert c.foo() == [c.address, ['0xF5D4020dCA6a62bB1efFcC9212AAF3c9819E30D7', '0xFFfFfFffFFfffFFfFFfFFFFFffFFFffffFfFFFfF']]

def test_single_type_tuple_bytes(get_contract):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@view\n@external\ndef foo() -> (Bytes[5], Bytes[5]):\n    return b"hello", b"there"\n    '
    c = get_contract(code)
    assert c.foo() == [b'hello', b'there']