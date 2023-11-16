import pytest
from vyper.compiler import compile_code
from vyper.exceptions import InvalidLiteral, InvalidType, NonPayableViolation, StateAccessViolation, UndeclaredDefinition

def test_default_param_abi(get_contract):
    if False:
        while True:
            i = 10
    code = '\n@external\n@payable\ndef safeTransferFrom(_data: Bytes[100] = b"test", _b: int128 = 1):\n    pass\n    '
    abi = get_contract(code)._classic_contract.abi
    assert len(abi) == 3
    assert set([fdef['name'] for fdef in abi]) == {'safeTransferFrom'}
    assert abi[0]['inputs'] == []
    assert abi[1]['inputs'] == [{'type': 'bytes', 'name': '_data'}]
    assert abi[2]['inputs'] == [{'type': 'bytes', 'name': '_data'}, {'type': 'int128', 'name': '_b'}]

def test_basic_default_param_passthrough(get_contract):
    if False:
        print('Hello World!')
    code = '\n@external\ndef fooBar(_data: Bytes[100] = b"test", _b: int128 = 1) -> int128:\n    return 12321\n    '
    c = get_contract(code)
    assert c.fooBar() == 12321
    assert c.fooBar(b'drum drum') == 12321
    assert c.fooBar(b'drum drum', 2) == 12321

def test_basic_default_param_set(get_contract):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\ndef fooBar(a:int128, b: uint256 = 333) -> (int128, uint256):\n    return a, b\n    '
    c = get_contract(code)
    assert c.fooBar(456, 444) == [456, 444]
    assert c.fooBar(456) == [456, 333]

def test_basic_default_param_set_2args(get_contract):
    if False:
        return 10
    code = '\n@external\ndef fooBar(a:int128, b: uint256 = 999, c: address = 0x0000000000000000000000000000000000000001) -> (int128, uint256, address):  # noqa: E501\n    return a, b, c\n    '
    c = get_contract(code)
    c_default_value = '0x0000000000000000000000000000000000000001'
    b_default_value = 999
    addr2 = '0x1000000000000000000000000000000000004321'
    assert c.fooBar(123) == [123, b_default_value, c_default_value]
    assert c.fooBar(456, 444) == [456, 444, c_default_value]
    assert c.fooBar(6789, 4567, addr2) == [6789, 4567, addr2]

def test_default_param_bytes(get_contract):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef fooBar(a: Bytes[100], b: int128, c: Bytes[100] = b"testing", d: uint256 = 999) -> (Bytes[100], int128, Bytes[100], uint256):  # noqa: E501\n    return a, b, c, d\n    '
    c = get_contract(code)
    c_default = b'testing'
    d_default = 999
    assert c.fooBar(b'booo', 12321, b'woo') == [b'booo', 12321, b'woo', d_default]
    assert c.fooBar(b'booo', 12321, b'lucky', 777) == [b'booo', 12321, b'lucky', 777]
    assert c.fooBar(b'booo', 12321) == [b'booo', 12321, c_default, d_default]

def test_default_param_array(get_contract):
    if False:
        print('Hello World!')
    code = '\n@external\ndef fooBar(a: Bytes[100], b: uint256[2], c: Bytes[6] = b"hello", d: int128[3] = [6, 7, 8]) -> (Bytes[100], uint256, Bytes[6], int128):  # noqa: E501\n    return a, b[1], c, d[2]\n    '
    c = get_contract(code)
    c_default = b'hello'
    d_default = 8
    assert c.fooBar(b'booo', [99, 88], b'woo') == [b'booo', 88, b'woo', d_default]
    assert c.fooBar(b'booo', [22, 11], b'lucky', [24, 25, 26]) == [b'booo', 11, b'lucky', 26]
    assert c.fooBar(b'booo', [55, 66]) == [b'booo', 66, c_default, d_default]

def test_default_param_internal_function(get_contract):
    if False:
        print('Hello World!')
    code = '\n@internal\n@view\ndef _foo(a: int128[3] = [1, 2, 3]) -> int128[3]:\n    b: int128[3] = a\n    return b\n\n\n@external\n@view\ndef foo() -> int128[3]:\n    return self._foo([4, 5, 6])\n\n@external\n@view\ndef foo2() -> int128[3]:\n    return self._foo()\n    '
    c = get_contract(code)
    assert c.foo() == [4, 5, 6]
    assert c.foo2() == [1, 2, 3]

def test_default_param_external_function(get_contract):
    if False:
        return 10
    code = '\n@external\n@view\ndef foo(a: int128[3] = [1, 2, 3]) -> int128[3]:\n    b: int128[3] = a\n    return b\n    '
    c = get_contract(code)
    assert c.foo([4, 5, 6]) == [4, 5, 6]
    assert c.foo() == [1, 2, 3]

def test_default_param_clamp(get_contract, monkeypatch, assert_tx_failed):
    if False:
        return 10
    code = '\n@external\ndef bar(a: int128, b: int128 = -1) -> (int128, int128):  # noqa: E501\n    return a, b\n    '
    c = get_contract(code)
    assert c.bar(-123) == [-123, -1]
    assert c.bar(100, 100) == [100, 100]

    def validate_value(cls, value):
        if False:
            print('Hello World!')
        pass
    monkeypatch.setattr('eth_abi.encoding.NumberEncoder.validate_value', validate_value)
    assert c.bar(200, 2 ** 127 - 1) == [200, 2 ** 127 - 1]
    assert_tx_failed(lambda : c.bar(200, 2 ** 127))

def test_default_param_private(get_contract):
    if False:
        return 10
    code = '\n@internal\ndef fooBar(a: Bytes[100], b: uint256, c: Bytes[20] = b"crazy") -> (Bytes[100], uint256, Bytes[20]):\n    return a, b, c\n\n@external\ndef callMe() -> (Bytes[100], uint256, Bytes[20]):\n    return self.fooBar(b\'I just met you\', 123456)\n\n@external\ndef callMeMaybe() -> (Bytes[100], uint256, Bytes[20]):\n    # return self.fooBar(b\'here is my number\', 555123456, b\'baby\')\n    a: Bytes[100] = b""\n    b: uint256 = 0\n    c: Bytes[20] = b""\n    a, b, c = self.fooBar(b\'here is my number\', 555123456, b\'baby\')\n    return a, b, c\n    '
    c = get_contract(code)
    assert c.callMeMaybe() == [b'here is my number', 555123456, b'baby']

def test_environment_vars_as_default(get_contract):
    if False:
        return 10
    code = '\nxx: uint256\n\n@external\n@payable\ndef foo(a: uint256 = msg.value) -> bool:\n    self.xx += a\n    return True\n\n@external\ndef bar() -> uint256:\n    return self.xx\n\n@external\ndef get_balance() -> uint256:\n    return self.balance\n    '
    c = get_contract(code)
    c.foo(transact={'value': 31337})
    assert c.bar() == 31337
    c.foo(666, transact={'value': 9001})
    assert c.bar() == 31337 + 666
    assert c.get_balance() == 31337 + 9001
PASSING_CONTRACTS = ['\n@external\ndef foo(a: bool = True, b: bool[2] = [True, False]): pass\n    ', '\n@external\ndef foo(\n    a: address = 0x0c04792e92e6b2896a18568fD936781E9857feB7,\n    b: address[2] = [\n        0x0c04792e92e6b2896a18568fD936781E9857feB7,\n        0x0c04792e92e6b2896a18568fD936781E9857feB7\n    ]): pass\n    ', '\n@external\ndef foo(a: uint256 = 12345, b: uint256[2] = [31337, 42]): pass\n    ', '\n@external\ndef foo(a: int128 = -31, b: int128[2] = [64, -46]): pass\n    ', '\n@external\ndef foo(a: Bytes[6] = b"potato"): pass\n    ', '\n@external\ndef foo(a: decimal = 3.14, b: decimal[2] = [1.337, 2.69]): pass\n    ', '\n@external\ndef foo(a: address = msg.sender, b: address[3] = [msg.sender, tx.origin, block.coinbase]): pass\n    ', '\n@internal\ndef foo(a: address = msg.sender, b: address[3] = [msg.sender, tx.origin, block.coinbase]): pass\n    ', '\n@external\n@payable\ndef foo(a: uint256 = msg.value): pass\n    ', '\n@external\ndef foo(a: uint256 = 2**8): pass\n    ', '\nstruct Bar:\n    a: address\n    b: uint256\n\n@external\ndef foo(bar: Bar = Bar({a: msg.sender, b: 1})): pass\n    ', '\nstruct Baz:\n    c: address\n    d: int128\n\nstruct Bar:\n    a: address\n    b: Baz\n\n@external\ndef foo(bar: Bar = Bar({a: msg.sender, b: Baz({c: block.coinbase, d: -10})})): pass\n    ', '\nA: public(address)\n\n@external\ndef foo(a: address = empty(address)):\n    self.A = a\n    ', '\nA: public(int112)\n\n@external\ndef foo(a: int112 = min_value(int112)):\n    self.A = a\n    ']

@pytest.mark.parametrize('code', PASSING_CONTRACTS)
def test_good_default_params(code):
    if False:
        print('Hello World!')
    assert compile_code(code)
FAILING_CONTRACTS = [('\n# default params must be literals\nx: int128\n\n@external\ndef foo(xx: int128, y: int128 = xx): pass\n    ', UndeclaredDefinition), ('\n# value out of range for uint256\n@external\ndef foo(a: uint256 = -1): pass\n    ', InvalidType), ('\n# value out of range for int128\n@external\ndef foo(a: int128 = 170141183460469231731687303715884105728): pass\n    ', InvalidType), ('\n# value out of range for uint256 array\n@external\ndef foo(a: uint256[2] = [13, -42]): pass\n     ', InvalidType), ('\n# value out of range for int128 array\n@external\ndef foo(a: int128[2] = [12, 170141183460469231731687303715884105728]): pass\n    ', InvalidType), ('\n# array type mismatch\n@external\ndef foo(a: uint256[2] = [12, True]): pass\n    ', InvalidLiteral), ('\n# wrong length\n@external\ndef foo(a: uint256[2] = [1, 2, 3]): pass\n    ', InvalidType), ('\n# default params must be literals\nx: uint256\n\n@external\ndef foo(a: uint256 = self.x): pass\n     ', StateAccessViolation), ('\n# default params must be literals inside array\nx: uint256\n\n@external\ndef foo(a: uint256[2] = [2, self.x]): pass\n     ', StateAccessViolation), ('\n# msg.value in a nonpayable\n@external\ndef foo(a: uint256 = msg.value): pass\n', NonPayableViolation)]

@pytest.mark.parametrize('failing_contract', FAILING_CONTRACTS)
def test_bad_default_params(failing_contract, assert_compile_failed):
    if False:
        return 10
    (code, exc) = failing_contract
    assert_compile_failed(lambda : compile_code(code), exc)