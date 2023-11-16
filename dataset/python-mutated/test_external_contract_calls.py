from decimal import Decimal
import pytest
from eth.codecs import abi
from vyper.exceptions import ArgumentException, InvalidType, StateAccessViolation, StructureException, UndeclaredDefinition, UnknownType

def test_external_contract_calls(get_contract, get_contract_with_gas_estimation):
    if False:
        return 10
    contract_1 = '\n@external\ndef foo(arg1: int128) -> int128:\n    return arg1\n    '
    c = get_contract_with_gas_estimation(contract_1)
    contract_2 = '\ninterface Foo:\n        def foo(arg1: int128) -> int128: view\n\n@external\ndef bar(arg1: address, arg2: int128) -> int128:\n    return Foo(arg1).foo(arg2)\n    '
    c2 = get_contract(contract_2)
    assert c2.bar(c.address, 1) == 1
    print('Successfully executed an external contract call')

def test_complicated_external_contract_calls(get_contract, get_contract_with_gas_estimation):
    if False:
        return 10
    contract_1 = "\nlucky: public(int128)\n\n@external\ndef __init__(_lucky: int128):\n    self.lucky = _lucky\n\n@external\ndef foo() -> int128:\n    return self.lucky\n\n@external\ndef array() -> Bytes[3]:\n    return b'dog'\n    "
    lucky_number = 7
    c = get_contract_with_gas_estimation(contract_1, *[lucky_number])
    contract_2 = '\ninterface Foo:\n    def foo() -> int128: nonpayable\n    def array() -> Bytes[3]: view\n\n@external\ndef bar(arg1: address) -> int128:\n    return Foo(arg1).foo()\n    '
    c2 = get_contract(contract_2)
    assert c2.bar(c.address) == lucky_number
    print('Successfully executed a complicated external contract call')

@pytest.mark.parametrize('length', [3, 32, 33, 64])
def test_external_contract_calls_with_bytes(get_contract, length):
    if False:
        for i in range(10):
            print('nop')
    contract_1 = f"\n@external\ndef array() -> Bytes[{length}]:\n    return b'dog'\n    "
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def array() -> Bytes[3]: view\n\n@external\ndef get_array(arg1: address) -> Bytes[3]:\n    return Foo(arg1).array()\n'
    c2 = get_contract(contract_2)
    assert c2.get_array(c.address) == b'dog'

def test_bytes_too_long(get_contract, assert_tx_failed):
    if False:
        i = 10
        return i + 15
    contract_1 = "\n@external\ndef array() -> Bytes[4]:\n    return b'doge'\n    "
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def array() -> Bytes[3]: view\n\n@external\ndef get_array(arg1: address) -> Bytes[3]:\n    return Foo(arg1).array()\n'
    c2 = get_contract(contract_2)
    assert_tx_failed(lambda : c2.get_array(c.address))

@pytest.mark.parametrize('revert_string', ['Mayday, mayday!', 'A very long revert string' + '.' * 512])
def test_revert_propagation(get_contract, assert_tx_failed, revert_string):
    if False:
        i = 10
        return i + 15
    raiser = f'\n@external\ndef run():\n    raise "{revert_string}"\n    '
    caller = '\ninterface Raises:\n    def run(): pure\n\n@external\ndef run(raiser: address):\n    Raises(raiser).run()\n    '
    c1 = get_contract(raiser)
    c2 = get_contract(caller)
    assert_tx_failed(lambda : c2.run(c1.address), exc_text=revert_string)

@pytest.mark.parametrize('a,b', [(3, 3), (4, 3), (3, 4), (32, 32), (33, 33), (64, 64)])
@pytest.mark.parametrize('actual', [3, 32, 64])
def test_tuple_with_bytes(get_contract, a, b, actual):
    if False:
        print('Hello World!')
    contract_1 = f"\n@external\ndef array() -> (Bytes[{actual}], int128, Bytes[{actual}]):\n    return b'dog', 255, b'cat'\n    "
    c = get_contract(contract_1)
    contract_2 = f'\ninterface Foo:\n    def array() -> (Bytes[{a}], int128, Bytes[{b}]): view\n\n@external\ndef get_array(arg1: address) -> (Bytes[{a}], int128, Bytes[{b}]):\n    a: Bytes[{a}] = b""\n    b: int128 = 0\n    c: Bytes[{b}] = b""\n    a, b, c = Foo(arg1).array()\n    return a, b, c\n'
    c2 = get_contract(contract_2)
    assert c.array() == [b'dog', 255, b'cat']
    assert c2.get_array(c.address) == [b'dog', 255, b'cat']

@pytest.mark.parametrize('a,b', [(18, 7), (18, 18), (19, 6), (64, 6), (7, 19)])
@pytest.mark.parametrize('c,d', [(19, 7), (64, 64)])
def test_tuple_with_bytes_too_long(get_contract, assert_tx_failed, a, c, b, d):
    if False:
        i = 10
        return i + 15
    contract_1 = f"\n@external\ndef array() -> (Bytes[{c}], int128, Bytes[{d}]):\n    return b'nineteen characters', 255, b'seven!!'\n    "
    c = get_contract(contract_1)
    contract_2 = f'\ninterface Foo:\n    def array() -> (Bytes[{a}], int128, Bytes[{b}]): view\n\n@external\ndef get_array(arg1: address) -> (Bytes[{a}], int128, Bytes[{b}]):\n    a: Bytes[{a}] = b""\n    b: int128 = 0\n    c: Bytes[{b}] = b""\n    a, b, c = Foo(arg1).array()\n    return a, b, c\n'
    c2 = get_contract(contract_2)
    assert c.array() == [b'nineteen characters', 255, b'seven!!']
    assert_tx_failed(lambda : c2.get_array(c.address))

def test_tuple_with_bytes_too_long_two(get_contract, assert_tx_failed):
    if False:
        for i in range(10):
            print('nop')
    contract_1 = "\n@external\ndef array() -> (Bytes[30], int128, Bytes[30]):\n    return b'nineteen characters', 255, b'seven!!'\n    "
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def array() -> (Bytes[30], int128, Bytes[3]): view\n\n@external\ndef get_array(arg1: address) -> (Bytes[30], int128, Bytes[3]):\n    a: Bytes[30] = b""\n    b: int128 = 0\n    c: Bytes[3] = b""\n    a, b, c = Foo(arg1).array()\n    return a, b, c\n'
    c2 = get_contract(contract_2)
    assert c.array() == [b'nineteen characters', 255, b'seven!!']
    assert_tx_failed(lambda : c2.get_array(c.address))

@pytest.mark.parametrize('length', [8, 256])
def test_external_contract_calls_with_uint8(get_contract, length):
    if False:
        print('Hello World!')
    contract_1 = f'\n@external\ndef foo() -> uint{length}:\n    return 255\n    '
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def foo() -> uint8: view\n\n@external\ndef bar(arg1: address) -> uint8:\n    return Foo(arg1).foo()\n'
    c2 = get_contract(contract_2)
    assert c2.bar(c.address) == 255

def test_uint8_too_long(get_contract, assert_tx_failed):
    if False:
        while True:
            i = 10
    contract_1 = '\n@external\ndef foo() -> uint256:\n    return 2**255\n    '
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def foo() -> uint8: view\n\n@external\ndef bar(arg1: address) -> uint8:\n    return Foo(arg1).foo()\n'
    c2 = get_contract(contract_2)
    assert_tx_failed(lambda : c2.bar(c.address))

@pytest.mark.parametrize('a,b', [(8, 8), (8, 256), (256, 8), (256, 256)])
@pytest.mark.parametrize('actual', [8, 256])
def test_tuple_with_uint8(get_contract, a, b, actual):
    if False:
        print('Hello World!')
    contract_1 = f"\n@external\ndef foo() -> (uint{actual}, Bytes[3], uint{actual}):\n    return 255, b'dog', 255\n    "
    c = get_contract(contract_1)
    contract_2 = f'\ninterface Foo:\n    def foo() -> (uint{a}, Bytes[3], uint{b}): view\n\n@external\ndef bar(arg1: address) -> (uint{a}, Bytes[3], uint{b}):\n    a: uint{a} = 0\n    b: Bytes[3] = b""\n    c: uint{b} = 0\n    a, b, c = Foo(arg1).foo()\n    return a, b, c\n'
    c2 = get_contract(contract_2)
    assert c.foo() == [255, b'dog', 255]
    assert c2.bar(c.address) == [255, b'dog', 255]

@pytest.mark.parametrize('a,b', [(8, 256), (256, 8), (256, 256)])
def test_tuple_with_uint8_too_long(get_contract, assert_tx_failed, a, b):
    if False:
        return 10
    contract_1 = f"\n@external\ndef foo() -> (uint{a}, Bytes[3], uint{b}):\n    return {2 ** a - 1}, b'dog', {2 ** b - 1}\n    "
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def foo() -> (uint8, Bytes[3], uint8): view\n\n@external\ndef bar(arg1: address) -> (uint8, Bytes[3], uint8):\n    a: uint8 = 0\n    b: Bytes[3] = b""\n    c: uint8 = 0\n    a, b, c = Foo(arg1).foo()\n    return a, b, c\n'
    c2 = get_contract(contract_2)
    assert c.foo() == [int(f'{2 ** a - 1}'), b'dog', int(f'{2 ** b - 1}')]
    assert_tx_failed(lambda : c2.bar(c.address))

@pytest.mark.parametrize('a,b', [(8, 256), (256, 8)])
def test_tuple_with_uint8_too_long_two(get_contract, assert_tx_failed, a, b):
    if False:
        for i in range(10):
            print('nop')
    contract_1 = f"\n@external\ndef foo() -> (uint{b}, Bytes[3], uint{a}):\n    return {2 ** b - 1}, b'dog', {2 ** a - 1}\n    "
    c = get_contract(contract_1)
    contract_2 = f'\ninterface Foo:\n    def foo() -> (uint{a}, Bytes[3], uint{b}): view\n\n@external\ndef bar(arg1: address) -> (uint{a}, Bytes[3], uint{b}):\n    a: uint{a} = 0\n    b: Bytes[3] = b""\n    c: uint{b} = 0\n    a, b, c = Foo(arg1).foo()\n    return a, b, c\n'
    c2 = get_contract(contract_2)
    assert c.foo() == [int(f'{2 ** b - 1}'), b'dog', int(f'{2 ** a - 1}')]
    assert_tx_failed(lambda : c2.bar(c.address))

@pytest.mark.parametrize('length', [128, 256])
def test_external_contract_calls_with_int128(get_contract, length):
    if False:
        for i in range(10):
            print('nop')
    contract_1 = f'\n@external\ndef foo() -> int{length}:\n    return 1\n    '
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def foo() -> int128: view\n\n@external\ndef bar(arg1: address) -> int128:\n    return Foo(arg1).foo()\n'
    c2 = get_contract(contract_2)
    assert c2.bar(c.address) == 1

def test_int128_too_long(get_contract, assert_tx_failed):
    if False:
        return 10
    contract_1 = '\n@external\ndef foo() -> int256:\n    return (2**255)-1\n    '
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def foo() -> int128: view\n\n@external\ndef bar(arg1: address) -> int128:\n    return Foo(arg1).foo()\n'
    c2 = get_contract(contract_2)
    assert_tx_failed(lambda : c2.bar(c.address))

@pytest.mark.parametrize('a,b', [(128, 128), (128, 256), (256, 128), (256, 256)])
@pytest.mark.parametrize('actual', [128, 256])
def test_tuple_with_int128(get_contract, a, b, actual):
    if False:
        print('Hello World!')
    contract_1 = f"\n@external\ndef foo() -> (int{actual}, Bytes[3], int{actual}):\n    return 255, b'dog', 255\n    "
    c = get_contract(contract_1)
    contract_2 = f'\ninterface Foo:\n    def foo() -> (int{a}, Bytes[3], int{b}): view\n\n@external\ndef bar(arg1: address) -> (int{a}, Bytes[3], int{b}):\n    a: int{a} = 0\n    b: Bytes[3] = b""\n    c: int{b} = 0\n    a, b, c = Foo(arg1).foo()\n    return a, b, c\n'
    c2 = get_contract(contract_2)
    assert c.foo() == [255, b'dog', 255]
    assert c2.bar(c.address) == [255, b'dog', 255]

@pytest.mark.parametrize('a,b', [(128, 256), (256, 128), (256, 256)])
def test_tuple_with_int128_too_long(get_contract, assert_tx_failed, a, b):
    if False:
        print('Hello World!')
    contract_1 = f"\n@external\ndef foo() -> (int{a}, Bytes[3], int{b}):\n    return {2 ** (a - 1) - 1}, b'dog', {2 ** (b - 1) - 1}\n    "
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def foo() -> (int128, Bytes[3], int128): view\n\n@external\ndef bar(arg1: address) -> (int128, Bytes[3], int128):\n    a: int128 = 0\n    b: Bytes[3] = b""\n    c: int128 = 0\n    a, b, c = Foo(arg1).foo()\n    return a, b, c\n'
    c2 = get_contract(contract_2)
    assert c.foo() == [int(f'{2 ** (a - 1) - 1}'), b'dog', int(f'{2 ** (b - 1) - 1}')]
    assert_tx_failed(lambda : c2.bar(c.address))

@pytest.mark.parametrize('a,b', [(128, 256), (256, 128)])
def test_tuple_with_int128_too_long_two(get_contract, assert_tx_failed, a, b):
    if False:
        print('Hello World!')
    contract_1 = f"\n@external\ndef foo() -> (int{b}, Bytes[3], int{a}):\n    return {2 ** (b - 1) - 1}, b'dog', {2 ** (a - 1) - 1}\n    "
    c = get_contract(contract_1)
    contract_2 = f'\ninterface Foo:\n    def foo() -> (int{a}, Bytes[3], int{b}): view\n\n@external\ndef bar(arg1: address) -> (int{a}, Bytes[3], int{b}):\n    a: int{a} = 0\n    b: Bytes[3] = b""\n    c: int{b} = 0\n    a, b, c = Foo(arg1).foo()\n    return a, b, c\n'
    c2 = get_contract(contract_2)
    assert c.foo() == [int(f'{2 ** (b - 1) - 1}'), b'dog', int(f'{2 ** (a - 1) - 1}')]
    assert_tx_failed(lambda : c2.bar(c.address))

@pytest.mark.parametrize('type', ['uint8', 'uint256', 'int128', 'int256'])
def test_external_contract_calls_with_decimal(get_contract, type):
    if False:
        for i in range(10):
            print('nop')
    contract_1 = f'\n@external\ndef foo() -> {type}:\n    return 1\n    '
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def foo() -> decimal: view\n\n@external\ndef bar(arg1: address) -> decimal:\n    return Foo(arg1).foo()\n'
    c2 = get_contract(contract_2)
    assert c2.bar(c.address) == Decimal('1e-10')

def test_decimal_too_long(get_contract, assert_tx_failed):
    if False:
        for i in range(10):
            print('nop')
    contract_1 = '\n@external\ndef foo() -> uint256:\n    return 2**255\n    '
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def foo() -> decimal: view\n\n@external\ndef bar(arg1: address) -> decimal:\n    return Foo(arg1).foo()\n'
    c2 = get_contract(contract_2)
    assert_tx_failed(lambda : c2.bar(c.address))

@pytest.mark.parametrize('a', ['uint8', 'uint256', 'int128', 'int256'])
@pytest.mark.parametrize('b', ['uint8', 'uint256', 'int128', 'int256'])
def test_tuple_with_decimal(get_contract, a, b):
    if False:
        i = 10
        return i + 15
    contract_1 = f"\n@external\ndef foo() -> ({a}, Bytes[3], {b}):\n    return 0, b'dog', 1\n    "
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def foo() -> (decimal, Bytes[3], decimal): view\n\n@external\ndef bar(arg1: address) -> (decimal, Bytes[3], decimal):\n    a: decimal = 0.0\n    b: Bytes[3] = b""\n    c: decimal = 0.0\n    a, b, c = Foo(arg1).foo()\n    return a, b, c\n'
    c2 = get_contract(contract_2)
    assert c.foo() == [0, b'dog', 1]
    result = c2.bar(c.address)
    assert result == [Decimal('0.0'), b'dog', Decimal('1e-10')]

@pytest.mark.parametrize('a,b', [(8, 256), (256, 8), (256, 256)])
def test_tuple_with_decimal_too_long(get_contract, assert_tx_failed, a, b):
    if False:
        while True:
            i = 10
    contract_1 = f"\n@external\ndef foo() -> (uint{a}, Bytes[3], uint{b}):\n    return {2 ** (a - 1)}, b'dog', {2 ** (b - 1)}\n    "
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def foo() -> (decimal, Bytes[3], decimal): view\n\n@external\ndef bar(arg1: address) -> (decimal, Bytes[3], decimal):\n    a: decimal = 0.0\n    b: Bytes[3] = b""\n    c: decimal = 0.0\n    a, b, c = Foo(arg1).foo()\n    return a, b, c\n'
    c2 = get_contract(contract_2)
    assert c.foo() == [2 ** (a - 1), b'dog', 2 ** (b - 1)]
    assert_tx_failed(lambda : c2.bar(c.address))

@pytest.mark.parametrize('type', ['uint8', 'uint256', 'int128', 'int256'])
def test_external_contract_calls_with_bool(get_contract, type):
    if False:
        while True:
            i = 10
    contract_1 = f'\n@external\ndef foo() -> {type}:\n    return 1\n    '
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def foo() -> bool: view\n\n@external\ndef bar(arg1: address) -> bool:\n    return Foo(arg1).foo()\n'
    c2 = get_contract(contract_2)
    assert c2.bar(c.address) is True

def test_bool_too_long(get_contract, assert_tx_failed):
    if False:
        return 10
    contract_1 = '\n@external\ndef foo() -> uint256:\n    return 2\n    '
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def foo() -> bool: view\n\n@external\ndef bar(arg1: address) -> bool:\n    return Foo(arg1).foo()\n'
    c2 = get_contract(contract_2)
    assert_tx_failed(lambda : c2.bar(c.address))

@pytest.mark.parametrize('a', ['uint8', 'uint256', 'int128', 'int256'])
@pytest.mark.parametrize('b', ['uint8', 'uint256', 'int128', 'int256'])
def test_tuple_with_bool(get_contract, a, b):
    if False:
        return 10
    contract_1 = f"\n@external\ndef foo() -> ({a}, Bytes[3], {b}):\n    return 1, b'dog', 0\n    "
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def foo() -> (bool, Bytes[3], bool): view\n\n@external\ndef bar(arg1: address) -> (bool, Bytes[3], bool):\n    a: bool = False\n    b: Bytes[3] = b""\n    c: bool = False\n    a, b, c = Foo(arg1).foo()\n    return a, b, c\n'
    c2 = get_contract(contract_2)
    assert c.foo() == [1, b'dog', 0]
    assert c2.bar(c.address) == [True, b'dog', False]

@pytest.mark.parametrize('a', ['uint8', 'uint256', 'int128', 'int256'])
@pytest.mark.parametrize('b', ['uint8', 'uint256', 'int128', 'int256'])
def test_tuple_with_bool_too_long(get_contract, assert_tx_failed, a, b):
    if False:
        print('Hello World!')
    contract_1 = f"\n@external\ndef foo() -> ({a}, Bytes[3], {b}):\n    return 1, b'dog', 2\n    "
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def foo() -> (bool, Bytes[3], bool): view\n\n@external\ndef bar(arg1: address) -> (bool, Bytes[3], bool):\n    a: bool = False\n    b: Bytes[3] = b""\n    c: bool = False\n    a, b, c = Foo(arg1).foo()\n    return a, b, c\n'
    c2 = get_contract(contract_2)
    assert c.foo() == [1, b'dog', 2]
    assert_tx_failed(lambda : c2.bar(c.address))

@pytest.mark.parametrize('type', ['uint8', 'int128', 'uint256', 'int256'])
def test_external_contract_calls_with_address(get_contract, type):
    if False:
        for i in range(10):
            print('nop')
    contract_1 = f'\n@external\ndef foo() -> {type}:\n    return 1\n    '
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def foo() -> address: view\n\n@external\ndef bar(arg1: address) -> address:\n    return Foo(arg1).foo()\n'
    c2 = get_contract(contract_2)
    assert c2.bar(c.address) == '0x0000000000000000000000000000000000000001'

@pytest.mark.parametrize('type', ['uint256', 'int256'])
def test_external_contract_calls_with_address_two(get_contract, type):
    if False:
        return 10
    contract_1 = f'\n@external\ndef foo() -> {type}:\n    return (2**160)-1\n    '
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def foo() -> address: view\n\n@external\ndef bar(arg1: address) -> address:\n    return Foo(arg1).foo()\n'
    c2 = get_contract(contract_2)
    assert c2.bar(c.address).lower() == '0xffffffffffffffffffffffffffffffffffffffff'

@pytest.mark.parametrize('type', ['uint256', 'int256'])
def test_address_too_long(get_contract, assert_tx_failed, type):
    if False:
        while True:
            i = 10
    contract_1 = f'\n@external\ndef foo() -> {type}:\n    return 2**160\n    '
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def foo() -> address: view\n\n@external\ndef bar(arg1: address) -> address:\n    return Foo(arg1).foo()\n'
    c2 = get_contract(contract_2)
    assert_tx_failed(lambda : c2.bar(c.address))

@pytest.mark.parametrize('a', ['uint8', 'int128', 'uint256', 'int256'])
@pytest.mark.parametrize('b', ['uint8', 'int128', 'uint256', 'int256'])
def test_tuple_with_address(get_contract, a, b):
    if False:
        i = 10
        return i + 15
    contract_1 = f"\n@external\ndef foo() -> ({a}, Bytes[3], {b}):\n    return 16, b'dog', 1\n    "
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def foo() -> (address, Bytes[3], address): view\n\n@external\ndef bar(arg1: address) -> (address, Bytes[3], address):\n    a: address = empty(address)\n    b: Bytes[3] = b""\n    c: address = empty(address)\n    a, b, c = Foo(arg1).foo()\n    return a, b, c\n'
    c2 = get_contract(contract_2)
    assert c.foo() == [16, b'dog', 1]
    assert c2.bar(c.address) == ['0x0000000000000000000000000000000000000010', b'dog', '0x0000000000000000000000000000000000000001']

@pytest.mark.parametrize('a', ['uint256', 'int256'])
@pytest.mark.parametrize('b', ['uint256', 'int256'])
def test_tuple_with_address_two(get_contract, a, b):
    if False:
        for i in range(10):
            print('nop')
    contract_1 = f"\n@external\ndef foo() -> ({a}, Bytes[3], {b}):\n    return (2**160)-1, b'dog', (2**160)-2\n    "
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def foo() -> (address, Bytes[3], address): view\n\n@external\ndef bar(arg1: address) -> (address, Bytes[3], address):\n    a: address = empty(address)\n    b: Bytes[3] = b""\n    c: address = empty(address)\n    a, b, c = Foo(arg1).foo()\n    return a, b, c\n'
    c2 = get_contract(contract_2)
    assert c.foo() == [2 ** 160 - 1, b'dog', 2 ** 160 - 2]
    result = c2.bar(c.address)
    assert len(result) == 3
    assert result[0].lower() == '0xffffffffffffffffffffffffffffffffffffffff'
    assert result[1] == b'dog'
    assert result[2].lower() == '0xfffffffffffffffffffffffffffffffffffffffe'

@pytest.mark.parametrize('a', ['uint256', 'int256'])
@pytest.mark.parametrize('b', ['uint256', 'int256'])
def test_tuple_with_address_too_long(get_contract, assert_tx_failed, a, b):
    if False:
        return 10
    contract_1 = f"\n@external\ndef foo() -> ({a}, Bytes[3], {b}):\n    return (2**160)-1, b'dog', 2**160\n    "
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def foo() -> (address, Bytes[3], address): view\n\n@external\ndef bar(arg1: address) -> (address, Bytes[3], address):\n    a: address = empty(address)\n    b: Bytes[3] = b""\n    c: address = empty(address)\n    a, b, c = Foo(arg1).foo()\n    return a, b, c\n'
    c2 = get_contract(contract_2)
    assert c.foo() == [2 ** 160 - 1, b'dog', 2 ** 160]
    assert_tx_failed(lambda : c2.bar(c.address))

def test_external_contract_call_state_change(get_contract):
    if False:
        while True:
            i = 10
    contract_1 = '\nlucky: public(int128)\n\n@external\ndef set_lucky(_lucky: int128):\n    self.lucky = _lucky\n    '
    lucky_number = 7
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def set_lucky(_lucky: int128): nonpayable\n\n@external\ndef set_lucky(arg1: address, arg2: int128):\n    Foo(arg1).set_lucky(arg2)\n    '
    c2 = get_contract(contract_2)
    assert c.lucky() == 0
    c2.set_lucky(c.address, lucky_number, transact={})
    assert c.lucky() == lucky_number
    print('Successfully executed an external contract call state change')

def test_constant_external_contract_call_cannot_change_state(assert_compile_failed, get_contract_with_gas_estimation):
    if False:
        return 10
    c = '\ninterface Foo:\n    def set_lucky(_lucky: int128) -> int128: nonpayable\n\n@external\n@view\ndef set_lucky_expr(arg1: address, arg2: int128):\n    Foo(arg1).set_lucky(arg2)\n\n@external\n@view\ndef set_lucky_stmt(arg1: address, arg2: int128) -> int128:\n    return Foo(arg1).set_lucky(arg2)\n    '
    assert_compile_failed(lambda : get_contract_with_gas_estimation(c), StateAccessViolation)
    print('Successfully blocked an external contract call from a constant function')

def test_external_contract_can_be_changed_based_on_address(get_contract):
    if False:
        i = 10
        return i + 15
    contract_1 = '\nlucky: public(int128)\n\n@external\ndef set_lucky(_lucky: int128):\n    self.lucky = _lucky\n    '
    lucky_number_1 = 7
    c = get_contract(contract_1)
    contract_2 = '\nlucky: public(int128)\n\n@external\ndef set_lucky(_lucky: int128) -> int128:\n    self.lucky = _lucky\n    return self.lucky\n    '
    lucky_number_2 = 3
    c2 = get_contract(contract_2)
    contract_3 = '\ninterface Foo:\n    def set_lucky(_lucky: int128): nonpayable\n\n@external\ndef set_lucky(arg1: address, arg2: int128):\n    Foo(arg1).set_lucky(arg2)\n    '
    c3 = get_contract(contract_3)
    c3.set_lucky(c.address, lucky_number_1, transact={})
    c3.set_lucky(c2.address, lucky_number_2, transact={})
    assert c.lucky() == lucky_number_1
    assert c2.lucky() == lucky_number_2
    print('Successfully executed multiple external contract calls to different contracts based on address')

def test_external_contract_calls_with_public_globals(get_contract):
    if False:
        while True:
            i = 10
    contract_1 = '\nlucky: public(int128)\n\n@external\ndef __init__(_lucky: int128):\n    self.lucky = _lucky\n    '
    lucky_number = 7
    c = get_contract(contract_1, *[lucky_number])
    contract_2 = '\ninterface Foo:\n    def lucky() -> int128: view\n\n@external\ndef bar(arg1: address) -> int128:\n    return Foo(arg1).lucky()\n    '
    c2 = get_contract(contract_2)
    assert c2.bar(c.address) == lucky_number
    print('Successfully executed an external contract call with public globals')

def test_external_contract_calls_with_multiple_contracts(get_contract):
    if False:
        return 10
    contract_1 = '\nlucky: public(int128)\n\n@external\ndef __init__(_lucky: int128):\n    self.lucky = _lucky\n    '
    lucky_number = 7
    c = get_contract(contract_1, *[lucky_number])
    contract_2 = '\ninterface Foo:\n    def lucky() -> int128: view\n\nmagic_number: public(int128)\n\n@external\ndef __init__(arg1: address):\n    self.magic_number = Foo(arg1).lucky()\n    '
    c2 = get_contract(contract_2, *[c.address])
    contract_3 = '\ninterface Bar:\n    def magic_number() -> int128: view\n\nbest_number: public(int128)\n\n@external\ndef __init__(arg1: address):\n    self.best_number = Bar(arg1).magic_number()\n    '
    c3 = get_contract(contract_3, *[c2.address])
    assert c3.best_number() == lucky_number
    print('Successfully executed a multiple external contract calls')

def test_external_contract_calls_with_default_value(get_contract):
    if False:
        while True:
            i = 10
    contract_1 = '\n@external\ndef foo(arg1: uint256=1) -> uint256:\n    return arg1\n    '
    contract_2 = '\ninterface Foo:\n    def foo(arg1: uint256=1) -> uint256: nonpayable\n\n@external\ndef bar(addr: address) -> uint256:\n    return Foo(addr).foo()\n    '
    c1 = get_contract(contract_1)
    c2 = get_contract(contract_2)
    assert c1.foo() == 1
    assert c1.foo(2) == 2
    assert c2.bar(c1.address) == 1

def test_external_contract_calls_with_default_value_two(get_contract):
    if False:
        i = 10
        return i + 15
    contract_1 = '\n@external\ndef foo(arg1: uint256, arg2: uint256=1) -> uint256:\n    return arg1 + arg2\n    '
    contract_2 = '\ninterface Foo:\n    def foo(arg1: uint256, arg2: uint256=1) -> uint256: nonpayable\n\n@external\ndef bar(addr: address, arg1: uint256) -> uint256:\n    return Foo(addr).foo(arg1)\n    '
    c1 = get_contract(contract_1)
    c2 = get_contract(contract_2)
    assert c1.foo(2) == 3
    assert c1.foo(2, 3) == 5
    assert c2.bar(c1.address, 2) == 3

def test_invalid_external_contract_call_to_the_same_contract(get_contract):
    if False:
        print('Hello World!')
    contract_1 = '\n@external\ndef bar() -> int128:\n    return 1\n    '
    contract_2 = '\ninterface Bar:\n    def bar() -> int128: view\n\n@external\ndef bar() -> int128:\n    return 1\n\n@external\ndef _stmt(x: address):\n    Bar(x).bar()\n\n@external\ndef _expr(x: address) -> int128:\n    return Bar(x).bar()\n    '
    c1 = get_contract(contract_1)
    c2 = get_contract(contract_2)
    c2._stmt(c1.address)
    c2._stmt(c2.address)
    assert c2._expr(c1.address) == 1
    assert c2._expr(c2.address) == 1

def test_invalid_nonexistent_contract_call(w3, assert_tx_failed, get_contract):
    if False:
        while True:
            i = 10
    contract_1 = '\n@external\ndef bar() -> int128:\n    return 1\n    '
    contract_2 = '\ninterface Bar:\n    def bar() -> int128: view\n\n@external\ndef foo(x: address) -> int128:\n    return Bar(x).bar()\n    '
    c1 = get_contract(contract_1)
    c2 = get_contract(contract_2)
    assert c2.foo(c1.address) == 1
    assert_tx_failed(lambda : c2.foo(w3.eth.accounts[0]))
    assert_tx_failed(lambda : c2.foo(w3.eth.accounts[3]))

def test_invalid_contract_reference_declaration(assert_tx_failed, get_contract):
    if False:
        for i in range(10):
            print('nop')
    contract = '\ninterface Bar:\n    get_magic_number: 1\n\nbest_number: public(int128)\n\n@external\ndef __init__():\n    pass\n'
    assert_tx_failed(lambda : get_contract(contract), exception=StructureException)

def test_invalid_contract_reference_call(assert_tx_failed, get_contract):
    if False:
        print('Hello World!')
    contract = '\n@external\ndef bar(arg1: address, arg2: int128) -> int128:\n    return Foo(arg1).foo(arg2)\n'
    assert_tx_failed(lambda : get_contract(contract), exception=UndeclaredDefinition)

def test_invalid_contract_reference_return_type(assert_tx_failed, get_contract):
    if False:
        while True:
            i = 10
    contract = '\ninterface Foo:\n    def foo(arg2: int128) -> invalid: view\n\n@external\ndef bar(arg1: address, arg2: int128) -> int128:\n    return Foo(arg1).foo(arg2)\n'
    assert_tx_failed(lambda : get_contract(contract), exception=UnknownType)

def test_external_contract_call_declaration_expr(get_contract):
    if False:
        for i in range(10):
            print('nop')
    contract_1 = '\n@external\ndef bar() -> int128:\n    return 1\n'
    contract_2 = '\ninterface Bar:\n    def bar() -> int128: view\n\nbar_contract: Bar\n\n@external\ndef foo(contract_address: address) -> int128:\n    self.bar_contract = Bar(contract_address)\n    return self.bar_contract.bar()\n    '
    c1 = get_contract(contract_1)
    c2 = get_contract(contract_2)
    assert c2.foo(c1.address) == 1

def test_external_contract_call_declaration_stmt(get_contract):
    if False:
        return 10
    contract_1 = '\nlucky: int128\n\n@external\ndef set_lucky(_lucky: int128):\n    self.lucky = _lucky\n\n@external\ndef get_lucky() -> int128:\n    return self.lucky\n'
    contract_2 = '\ninterface Bar:\n    def set_lucky(arg1: int128): nonpayable\n    def get_lucky() -> int128: view\n\nbar_contract: Bar\n\n@external\ndef set_lucky(contract_address: address):\n    self.bar_contract = Bar(contract_address)\n    self.bar_contract.set_lucky(1)\n\n@external\ndef get_lucky(contract_address: address) -> int128:\n    self.bar_contract = Bar(contract_address)\n    return self.bar_contract.get_lucky()\n    '
    c1 = get_contract(contract_1)
    c2 = get_contract(contract_2)
    assert c1.get_lucky() == 0
    assert c2.get_lucky(c1.address) == 0
    c1.set_lucky(6, transact={})
    assert c1.get_lucky() == 6
    assert c2.get_lucky(c1.address) == 6
    c2.set_lucky(c1.address, transact={})
    assert c1.get_lucky() == 1
    assert c2.get_lucky(c1.address) == 1

def test_complex_external_contract_call_declaration(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    contract_1 = '\n@external\ndef get_lucky() -> int128:\n    return 1\n'
    contract_2 = '\n@external\ndef get_lucky() -> int128:\n    return 2\n'
    contract_3 = '\ninterface Bar:\n    def set_lucky(arg1: int128): nonpayable\n    def get_lucky() -> int128: view\n\nbar_contract: Bar\n\n@external\ndef set_contract(contract_address: address):\n    self.bar_contract = Bar(contract_address)\n\n@external\ndef get_lucky() -> int128:\n    return self.bar_contract.get_lucky()\n'
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    c3 = get_contract_with_gas_estimation(contract_3)
    assert c1.get_lucky() == 1
    assert c2.get_lucky() == 2
    c3.set_contract(c1.address, transact={})
    assert c3.get_lucky() == 1
    c3.set_contract(c2.address, transact={})
    assert c3.get_lucky() == 2

def test_address_can_returned_from_contract_type(get_contract):
    if False:
        while True:
            i = 10
    contract_1 = '\n@external\ndef bar() -> int128:\n    return 1\n'
    contract_2 = '\ninterface Bar:\n    def bar() -> int128: view\n\nbar_contract: public(Bar)\n\n@external\ndef foo(contract_address: address):\n    self.bar_contract = Bar(contract_address)\n\n@external\ndef get_bar() -> int128:\n    return self.bar_contract.bar()\n'
    c1 = get_contract(contract_1)
    c2 = get_contract(contract_2)
    c2.foo(c1.address, transact={})
    assert c2.bar_contract() == c1.address
    assert c2.get_bar() == 1

def test_invalid_external_contract_call_declaration_1(assert_compile_failed, get_contract):
    if False:
        for i in range(10):
            print('nop')
    contract_1 = '\ninterface Bar:\n    def bar() -> int128: view\n\nbar_contract: Bar\n\n@external\ndef foo(contract_address: contract(Boo)) -> int128:\n    self.bar_contract = Bar(contract_address)\n    return self.bar_contract.bar()\n    '
    assert_compile_failed(lambda : get_contract(contract_1), InvalidType)

def test_invalid_external_contract_call_declaration_2(assert_compile_failed, get_contract):
    if False:
        while True:
            i = 10
    contract_1 = '\ninterface Bar:\n    def bar() -> int128: view\n\nbar_contract: Boo\n\n@external\ndef foo(contract_address: address) -> int128:\n    self.bar_contract = Bar(contract_address)\n    return self.bar_contract.bar()\n    '
    assert_compile_failed(lambda : get_contract(contract_1), UnknownType)

def test_external_with_payable_value(w3, get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    contract_1 = '\n@payable\n@external\ndef get_lucky() -> int128:\n    return 1\n\n@external\ndef get_balance() -> uint256:\n    return self.balance\n'
    contract_2 = '\ninterface Bar:\n    def get_lucky() -> int128: payable\n\nbar_contract: Bar\n\n@external\ndef set_contract(contract_address: address):\n    self.bar_contract = Bar(contract_address)\n\n@payable\n@external\ndef get_lucky(amount_to_send: uint256) -> int128:\n    if amount_to_send != 0:\n        return self.bar_contract.get_lucky(value=amount_to_send)\n    else: # send it all\n        return self.bar_contract.get_lucky(value=msg.value)\n'
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    assert c1.get_lucky() == 1
    assert c1.get_balance() == 0
    c2.set_contract(c1.address, transact={})
    assert c2.get_lucky(0, call={'value': 500}) == 1
    c2.get_lucky(0, transact={'value': 500})
    assert c1.get_balance() == 500
    assert w3.eth.get_balance(c1.address) == 500
    assert w3.eth.get_balance(c2.address) == 0
    assert c2.get_lucky(250, call={'value': 500}) == 1
    c2.get_lucky(250, transact={'value': 500})
    assert c1.get_balance() == 750
    assert w3.eth.get_balance(c1.address) == 750
    assert w3.eth.get_balance(c2.address) == 250

def test_external_call_with_gas(assert_tx_failed, get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    contract_1 = '\n@external\ndef get_lucky() -> int128:\n    return 656598\n'
    contract_2 = '\ninterface Bar:\n    def set_lucky(arg1: int128): nonpayable\n    def get_lucky() -> int128: view\n\nbar_contract: Bar\n\n@external\ndef set_contract(contract_address: address):\n    self.bar_contract = Bar(contract_address)\n\n@external\ndef get_lucky(gas_amount: uint256) -> int128:\n    return self.bar_contract.get_lucky(gas=gas_amount)\n'
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    c2.set_contract(c1.address, transact={})
    assert c2.get_lucky(1000) == 656598
    assert_tx_failed(lambda : c2.get_lucky(50))

def test_skip_contract_check(get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    contract_2 = '\n@external\n@view\ndef bar():\n    pass\n    '
    contract_1 = '\ninterface Bar:\n    def bar() -> uint256: view\n    def baz(): view\n\n@external\ndef call_bar(addr: address):\n    # would fail if returndatasize check were on\n    x: uint256 = Bar(addr).bar(skip_contract_check=True)\n@external\ndef call_baz():\n    # some address with no code\n    addr: address = 0x1234567890AbcdEF1234567890aBcdef12345678\n    # would fail if extcodesize check were on\n    Bar(addr).baz(skip_contract_check=True)\n    '
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    c1.call_bar(c2.address)
    c1.call_baz()

def test_invalid_keyword_on_call(assert_compile_failed, get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    contract_1 = '\ninterface Bar:\n    def set_lucky(arg1: int128): nonpayable\n    def get_lucky() -> int128: view\n\nbar_contract: Bar\n\n@external\ndef get_lucky(amount_to_send: int128) -> int128:\n    return self.bar_contract.get_lucky(gass=1)\n    '
    assert_compile_failed(lambda : get_contract_with_gas_estimation(contract_1), ArgumentException)

def test_invalid_contract_declaration(assert_compile_failed, get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    contract_1 = '\ninterface Bar:\n    def set_lucky(arg1: int128): nonpayable\n\nbar_contract: Barr\n\n    '
    assert_compile_failed(lambda : get_contract_with_gas_estimation(contract_1), UnknownType)
FAILING_CONTRACTS_STRUCTURE_EXCEPTION = ['\n# wrong arg count\ninterface Bar:\n    def bar(arg1: int128) -> bool: view\n\n@external\ndef foo(a: address):\n    Bar(a).bar(1, 2)\n    ', '\n# expected args, none given\ninterface Bar:\n    def bar(arg1: int128) -> bool: view\n\n@external\ndef foo(a: address):\n    Bar(a).bar()\n    ', '\n# expected no args, args given\ninterface Bar:\n    def bar() -> bool: view\n\n@external\ndef foo(a: address):\n    Bar(a).bar(1)\n    ', '\ninterface Bar:\n    def bar(x: uint256, y: uint256) -> uint256: view\n\n@external\ndef foo(a: address, x: uint256, y: uint256):\n    Bar(a).bar(x, y=y)\n    ']

@pytest.mark.parametrize('bad_code', FAILING_CONTRACTS_STRUCTURE_EXCEPTION)
def test_bad_code_struct_exc(assert_compile_failed, get_contract_with_gas_estimation, bad_code):
    if False:
        for i in range(10):
            print('nop')
    assert_compile_failed(lambda : get_contract_with_gas_estimation(bad_code), ArgumentException)

def test_bad_skip_contract_check(assert_compile_failed, get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    code = '\n# variable value for skip_contract_check\ninterface Bar:\n    def bar(): payable\n\n@external\ndef foo():\n    x: bool = True\n    Bar(msg.sender).bar(skip_contract_check=x)\n    '
    assert_compile_failed(lambda : get_contract_with_gas_estimation(code), InvalidType)

def test_tuple_return_external_contract_call(get_contract):
    if False:
        print('Hello World!')
    contract_1 = '\n@external\ndef out_literals() -> (int128, address, Bytes[10]):\n    return 1, 0x0000000000000000000000000000000000000123, b"random"\n    '
    contract_2 = '\ninterface Test:\n    def out_literals() -> (int128, address, Bytes[10]) : view\n\n@external\ndef test(addr: address) -> (int128, address, Bytes[10]):\n    a: int128 = 0\n    b: address = empty(address)\n    c: Bytes[10] = b""\n    (a, b, c) = Test(addr).out_literals()\n    return a, b,c\n\n    '
    c1 = get_contract(contract_1)
    c2 = get_contract(contract_2)
    assert c1.out_literals() == [1, '0x0000000000000000000000000000000000000123', b'random']
    assert c2.test(c1.address) == [1, '0x0000000000000000000000000000000000000123', b'random']

def test_struct_return_external_contract_call_1(get_contract_with_gas_estimation):
    if False:
        return 10
    contract_1 = '\nstruct X:\n    x: int128\n    y: address\n@external\ndef out_literals() -> X:\n    return X({x: 1, y: 0x0000000000000000000000000000000000012345})\n    '
    contract_2 = '\nstruct X:\n    x: int128\n    y: address\ninterface Test:\n    def out_literals() -> X : view\n\n@external\ndef test(addr: address) -> (int128, address):\n    ret: X = Test(addr).out_literals()\n    return ret.x, ret.y\n\n    '
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    assert c1.out_literals() == (1, '0x0000000000000000000000000000000000012345')
    assert c2.test(c1.address) == list(c1.out_literals())

@pytest.mark.parametrize('i,ln,s,', [(100, 6, 'abcde'), (41, 40, 'a' * 34), (57, 70, 'z' * 68)])
def test_struct_return_external_contract_call_2(get_contract_with_gas_estimation, i, ln, s):
    if False:
        while True:
            i = 10
    contract_1 = f'\nstruct X:\n    x: int128\n    y: String[{ln}]\n    z: Bytes[{ln}]\n@external\ndef get_struct_x() -> X:\n    return X({{x: {i}, y: "{s}", z: b"{s}"}})\n    '
    contract_2 = f'\nstruct X:\n    x: int128\n    y: String[{ln}]\n    z: Bytes[{ln}]\ninterface Test:\n    def get_struct_x() -> X : view\n\n@external\ndef test(addr: address) -> (int128, String[{ln}], Bytes[{ln}]):\n    ret: X = Test(addr).get_struct_x()\n    return ret.x, ret.y, ret.z\n\n    '
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    assert c1.get_struct_x() == (i, s, bytes(s, 'utf-8'))
    assert c2.test(c1.address) == list(c1.get_struct_x())

def test_struct_return_external_contract_call_3(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    contract_1 = '\nstruct X:\n    x: int128\n@external\ndef out_literals() -> X:\n    return X({x: 1})\n    '
    contract_2 = '\nstruct X:\n    x: int128\ninterface Test:\n    def out_literals() -> X : view\n\n@external\ndef test(addr: address) -> int128:\n    ret: X = Test(addr).out_literals()\n    return ret.x\n\n    '
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    assert c1.out_literals() == (1,)
    assert [c2.test(c1.address)] == list(c1.out_literals())

def test_constant_struct_return_external_contract_call_1(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    contract_1 = '\nstruct X:\n    x: int128\n    y: address\n\nBAR: constant(X) = X({x: 1, y: 0x0000000000000000000000000000000000012345})\n\n@external\ndef out_literals() -> X:\n    return BAR\n    '
    contract_2 = '\nstruct X:\n    x: int128\n    y: address\ninterface Test:\n    def out_literals() -> X : view\n\n@external\ndef test(addr: address) -> (int128, address):\n    ret: X = Test(addr).out_literals()\n    return ret.x, ret.y\n\n    '
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    assert c1.out_literals() == (1, '0x0000000000000000000000000000000000012345')
    assert c2.test(c1.address) == list(c1.out_literals())

@pytest.mark.parametrize('i,ln,s,', [(100, 6, 'abcde'), (41, 40, 'a' * 34), (57, 70, 'z' * 68)])
def test_constant_struct_return_external_contract_call_2(get_contract_with_gas_estimation, i, ln, s):
    if False:
        return 10
    contract_1 = f'\nstruct X:\n    x: int128\n    y: String[{ln}]\n    z: Bytes[{ln}]\n\nBAR: constant(X) = X({{x: {i}, y: "{s}", z: b"{s}"}})\n\n@external\ndef get_struct_x() -> X:\n    return BAR\n    '
    contract_2 = f'\nstruct X:\n    x: int128\n    y: String[{ln}]\n    z: Bytes[{ln}]\ninterface Test:\n    def get_struct_x() -> X : view\n\n@external\ndef test(addr: address) -> (int128, String[{ln}], Bytes[{ln}]):\n    ret: X = Test(addr).get_struct_x()\n    return ret.x, ret.y, ret.z\n\n    '
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    assert c1.get_struct_x() == (i, s, bytes(s, 'utf-8'))
    assert c2.test(c1.address) == list(c1.get_struct_x())

def test_constant_struct_return_external_contract_call_3(get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    contract_1 = '\nstruct X:\n    x: int128\n\nBAR: constant(X) = X({x: 1})\n\n@external\ndef out_literals() -> X:\n    return BAR\n    '
    contract_2 = '\nstruct X:\n    x: int128\ninterface Test:\n    def out_literals() -> X : view\n\n@external\ndef test(addr: address) -> int128:\n    ret: X = Test(addr).out_literals()\n    return ret.x\n\n    '
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    assert c1.out_literals() == (1,)
    assert [c2.test(c1.address)] == list(c1.out_literals())

def test_constant_struct_member_return_external_contract_call_1(get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    contract_1 = '\nstruct X:\n    x: int128\n    y: address\n\nBAR: constant(X) = X({x: 1, y: 0x0000000000000000000000000000000000012345})\n\n@external\ndef get_y() -> address:\n    return BAR.y\n    '
    contract_2 = '\ninterface Test:\n    def get_y() -> address : view\n\n@external\ndef test(addr: address) -> address:\n    ret: address = Test(addr).get_y()\n    return ret\n    '
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    assert c1.get_y() == '0x0000000000000000000000000000000000012345'
    assert c2.test(c1.address) == '0x0000000000000000000000000000000000012345'

@pytest.mark.parametrize('i,ln,s,', [(100, 6, 'abcde'), (41, 40, 'a' * 34), (57, 70, 'z' * 68)])
def test_constant_struct_member_return_external_contract_call_2(get_contract_with_gas_estimation, i, ln, s):
    if False:
        while True:
            i = 10
    contract_1 = f'\nstruct X:\n    x: int128\n    y: String[{ln}]\n    z: Bytes[{ln}]\n\nBAR: constant(X) = X({{x: {i}, y: "{s}", z: b"{s}"}})\n\n@external\ndef get_y() -> String[{ln}]:\n    return BAR.y\n    '
    contract_2 = f'\ninterface Test:\n    def get_y() -> String[{ln}] : view\n\n@external\ndef test(addr: address) -> String[{ln}]:\n    ret: String[{ln}] = Test(addr).get_y()\n    return ret\n\n    '
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    assert c1.get_y() == s
    assert c2.test(c1.address) == s

def test_constant_struct_member_return_external_contract_call_3(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    contract_1 = '\nstruct X:\n    x: int128\n\nBAR: constant(X) = X({x: 1})\n\n@external\ndef get_x() -> int128:\n    return BAR.x\n    '
    contract_2 = '\ninterface Test:\n    def get_x() -> int128 : view\n\n@external\ndef test(addr: address) -> int128:\n    ret: int128 = Test(addr).get_x()\n    return ret\n\n    '
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    assert c1.get_x() == 1
    assert c2.test(c1.address) == 1

def test_constant_nested_struct_return_external_contract_call_1(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    contract_1 = '\nstruct X:\n    x: int128\n    y: address\n\nstruct A:\n    a: X\n    b: uint256\n\nBAR: constant(A) = A({a: X({x: 1, y: 0x0000000000000000000000000000000000012345}), b: 777})\n\n@external\ndef out_literals() -> A:\n    return BAR\n    '
    contract_2 = '\nstruct X:\n    x: int128\n    y: address\n\nstruct A:\n    a: X\n    b: uint256\n\ninterface Test:\n    def out_literals() -> A : view\n\n@external\ndef test(addr: address) -> (X, uint256):\n    ret: A = Test(addr).out_literals()\n    return ret.a, ret.b\n    '
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    assert c1.out_literals() == ((1, '0x0000000000000000000000000000000000012345'), 777)
    assert c2.test(c1.address) == list(c1.out_literals())

@pytest.mark.parametrize('i,ln,s,', [(100, 6, 'abcde'), (41, 40, 'a' * 34), (57, 70, 'z' * 68)])
def test_constant_nested_struct_return_external_contract_call_2(get_contract_with_gas_estimation, i, ln, s):
    if False:
        return 10
    contract_1 = f'\nstruct X:\n    x: int128\n    y: String[{ln}]\n    z: Bytes[{ln}]\n\nstruct A:\n    a: X\n    b: uint256\n\nBAR: constant(A) = A({{a: X({{x: {i}, y: "{s}", z: b"{s}"}}), b: 777}})\n\n@external\ndef get_struct_a() -> A:\n    return BAR\n    '
    contract_2 = f'\nstruct X:\n    x: int128\n    y: String[{ln}]\n    z: Bytes[{ln}]\n\nstruct A:\n    a: X\n    b: uint256\n\ninterface Test:\n    def get_struct_a() -> A : view\n\n@external\ndef test(addr: address) -> (X, uint256):\n    ret: A = Test(addr).get_struct_a()\n    return ret.a, ret.b\n\n    '
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    assert c1.get_struct_a() == ((i, s, bytes(s, 'utf-8')), 777)
    assert c2.test(c1.address) == list(c1.get_struct_a())

def test_constant_nested_struct_return_external_contract_call_3(get_contract_with_gas_estimation):
    if False:
        return 10
    contract_1 = '\nstruct X:\n    x: int128\n    y: int128\n\nstruct A:\n    a: X\n    b: uint256\n\nstruct C:\n    c: A\n    d: bool\n\nBAR: constant(C) = C({c: A({a: X({x: 1, y: -1}), b: 777}), d: True})\n\n@external\ndef out_literals() -> C:\n    return BAR\n    '
    contract_2 = '\nstruct X:\n    x: int128\n    y: int128\n\nstruct A:\n    a: X\n    b: uint256\n\nstruct C:\n    c: A\n    d: bool\n\ninterface Test:\n    def out_literals() -> C : view\n\n@external\ndef test(addr: address) -> (A, bool):\n    ret: C = Test(addr).out_literals()\n    return ret.c, ret.d\n    '
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    assert c1.out_literals() == (((1, -1), 777), True)
    assert c2.test(c1.address) == list(c1.out_literals())

def test_constant_nested_struct_member_return_external_contract_call_1(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    contract_1 = '\nstruct X:\n    x: int128\n    y: address\n\nstruct A:\n    a: X\n    b: uint256\n\nBAR: constant(A) = A({a: X({x: 1, y: 0x0000000000000000000000000000000000012345}), b: 777})\n\n@external\ndef get_y() -> address:\n    return BAR.a.y\n    '
    contract_2 = '\ninterface Test:\n    def get_y() -> address : view\n\n@external\ndef test(addr: address) -> address:\n    ret: address = Test(addr).get_y()\n    return ret\n    '
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    assert c1.get_y() == '0x0000000000000000000000000000000000012345'
    assert c2.test(c1.address) == '0x0000000000000000000000000000000000012345'

@pytest.mark.parametrize('i,ln,s,', [(100, 6, 'abcde'), (41, 40, 'a' * 34), (57, 70, 'z' * 68)])
def test_constant_nested_struct_member_return_external_contract_call_2(get_contract_with_gas_estimation, i, ln, s):
    if False:
        for i in range(10):
            print('nop')
    contract_1 = f'\nstruct X:\n    x: int128\n    y: String[{ln}]\n    z: Bytes[{ln}]\n\nstruct A:\n    a: X\n    b: uint256\n    c: bool\n\nBAR: constant(A) = A({{a: X({{x: {i}, y: "{s}", z: b"{s}"}}), b: 777, c: True}})\n\n@external\ndef get_y() -> String[{ln}]:\n    return BAR.a.y\n    '
    contract_2 = f'\ninterface Test:\n    def get_y() -> String[{ln}] : view\n\n@external\ndef test(addr: address) -> String[{ln}]:\n    ret: String[{ln}] = Test(addr).get_y()\n    return ret\n\n    '
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    assert c1.get_y() == s
    assert c2.test(c1.address) == s

def test_constant_nested_struct_member_return_external_contract_call_3(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    contract_1 = '\nstruct X:\n    x: int128\n    y: int128\n\nstruct A:\n    a: X\n    b: uint256\n\nstruct C:\n    c: A\n    d: bool\n\nBAR: constant(C) = C({c: A({a: X({x: 1, y: -1}), b: 777}), d: True})\n\n@external\ndef get_y() -> int128:\n    return BAR.c.a.y\n\n@external\ndef get_b() -> uint256:\n    return BAR.c.b\n    '
    contract_2 = '\ninterface Test:\n    def get_y() -> int128 : view\n    def get_b() -> uint256 : view\n\n@external\ndef test(addr: address) -> int128:\n    ret: int128 = Test(addr).get_y()\n    return ret\n\n@external\ndef test2(addr: address) -> uint256:\n    ret: uint256 = Test(addr).get_b()\n    return ret\n    '
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    assert c1.get_y() == -1
    assert c2.test(c1.address) == -1
    assert c1.get_b() == 777
    assert c2.test2(c1.address) == 777

def test_dynamically_sized_struct_external_contract_call(get_contract_with_gas_estimation):
    if False:
        return 10
    contract_1 = '\nstruct X:\n    x: uint256\n    y: Bytes[6]\n\n@external\ndef foo(x: X) -> Bytes[6]:\n    return x.y\n    '
    contract_2 = '\nstruct X:\n    x: uint256\n    y: Bytes[6]\n\ninterface Foo:\n    def foo(x: X) -> Bytes[6]: nonpayable\n\n@external\ndef bar(addr: address) -> Bytes[6]:\n    _X: X = X({x: 1, y: b"hello"})\n    return Foo(addr).foo(_X)\n    '
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    assert c1.foo((1, b'hello')) == b'hello'
    assert c2.bar(c1.address) == b'hello'

def test_dynamically_sized_struct_external_contract_call_2(get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    contract_1 = '\nstruct X:\n    x: uint256\n    y: String[6]\n\n@external\ndef foo(x: X) -> String[6]:\n    return x.y\n    '
    contract_2 = '\nstruct X:\n    x: uint256\n    y: String[6]\n\ninterface Foo:\n    def foo(x: X) -> String[6]: nonpayable\n\n@external\ndef bar(addr: address) -> String[6]:\n    _X: X = X({x: 1, y: "hello"})\n    return Foo(addr).foo(_X)\n    '
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    assert c1.foo((1, 'hello')) == 'hello'
    assert c2.bar(c1.address) == 'hello'

def test_dynamically_sized_struct_member_external_contract_call(get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    contract_1 = '\n@external\ndef foo(b: Bytes[6]) -> Bytes[6]:\n    return b\n    '
    contract_2 = '\nstruct X:\n    x: uint256\n    y: Bytes[6]\n\ninterface Foo:\n    def foo(b: Bytes[6]) -> Bytes[6]: nonpayable\n\n@external\ndef bar(addr: address) -> Bytes[6]:\n    _X: X = X({x: 1, y: b"hello"})\n    return Foo(addr).foo(_X.y)\n    '
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    assert c1.foo(b'hello') == b'hello'
    assert c2.bar(c1.address) == b'hello'

def test_dynamically_sized_struct_member_external_contract_call_2(get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    contract_1 = '\n@external\ndef foo(s: String[6]) -> String[6]:\n    return s\n    '
    contract_2 = '\nstruct X:\n    x: uint256\n    y: String[6]\n\ninterface Foo:\n    def foo(b: String[6]) -> String[6]: nonpayable\n\n@external\ndef bar(addr: address) -> String[6]:\n    _X: X = X({x: 1, y: "hello"})\n    return Foo(addr).foo(_X.y)\n    '
    c1 = get_contract_with_gas_estimation(contract_1)
    c2 = get_contract_with_gas_estimation(contract_2)
    assert c1.foo('hello') == 'hello'
    assert c2.bar(c1.address) == 'hello'

def test_list_external_contract_call(get_contract, get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    contract_1 = '\n@external\ndef array() -> int128[3]:\n    return [0, 0, 0]\n    '
    c = get_contract_with_gas_estimation(contract_1)
    contract_2 = '\ninterface Foo:\n    def array() -> int128[3]: view\n@external\ndef get_array(arg1: address) -> int128[3]:\n    return Foo(arg1).array()\n'
    c2 = get_contract(contract_2)
    assert c2.get_array(c.address) == [0, 0, 0]

def test_returndatasize_too_short(get_contract, assert_tx_failed):
    if False:
        print('Hello World!')
    contract_1 = '\n@external\ndef bar(a: int128) -> int128:\n    return a\n'
    contract_2 = '\ninterface Bar:\n    def bar(a: int128) -> (int128, int128): view\n\n@external\ndef foo(_addr: address):\n    Bar(_addr).bar(456)\n'
    c1 = get_contract(contract_1)
    c2 = get_contract(contract_2)
    assert_tx_failed(lambda : c2.foo(c1.address))

def test_returndatasize_empty(get_contract, assert_tx_failed):
    if False:
        return 10
    contract_1 = '\n@external\ndef bar(a: int128):\n    pass\n'
    contract_2 = '\ninterface Bar:\n    def bar(a: int128) -> int128: view\n\n@external\ndef foo(_addr: address) -> int128:\n    return Bar(_addr).bar(456)\n'
    c1 = get_contract(contract_1)
    c2 = get_contract(contract_2)
    assert_tx_failed(lambda : c2.foo(c1.address))

def test_returndatasize_too_long(get_contract):
    if False:
        for i in range(10):
            print('nop')
    contract_1 = '\n@external\ndef bar(a: int128) -> (int128, int128):\n    return a, 789\n'
    contract_2 = '\ninterface Bar:\n    def bar(a: int128) -> int128: view\n\n@external\ndef foo(_addr: address) -> int128:\n    return Bar(_addr).bar(456)\n'
    c1 = get_contract(contract_1)
    c2 = get_contract(contract_2)
    assert c2.foo(c1.address) == 456

def test_no_returndata(get_contract, assert_tx_failed):
    if False:
        return 10
    contract_1 = '\n@external\ndef bar(a: int128) -> int128:\n    return a\n'
    contract_2 = '\ninterface Bar:\n    def bar(a: int128) -> int128: view\n\n@external\ndef foo(_addr: address, _addr2: address) -> int128:\n    x: int128 = Bar(_addr).bar(456)\n    # make two calls to confirm EVM behavior: RETURNDATA is always based on the last call\n    y: int128 = Bar(_addr2).bar(123)\n    return y\n\n'
    c1 = get_contract(contract_1)
    c2 = get_contract(contract_2)
    assert c2.foo(c1.address, c1.address) == 123
    assert_tx_failed(lambda : c2.foo(c1.address, '0x1234567890123456789012345678901234567890'))

def test_default_override(get_contract, assert_tx_failed):
    if False:
        while True:
            i = 10
    bad_erc20_code = '\n@external\ndef transfer(receiver: address, amount: uint256):\n    pass\n    '
    negative_transfer_code = '\n@external\ndef transfer(receiver: address, amount: uint256) -> bool:\n    return False\n    '
    self_destructing_code = '\n@external\ndef transfer(receiver: address, amount: uint256):\n    selfdestruct(msg.sender)\n    '
    code = '\nfrom vyper.interfaces import ERC20\n@external\ndef safeTransfer(erc20: ERC20, receiver: address, amount: uint256) -> uint256:\n    assert erc20.transfer(receiver, amount, default_return_value=True)\n    return 7\n\n@external\ndef transferBorked(erc20: ERC20, receiver: address, amount: uint256):\n    assert erc20.transfer(receiver, amount)\n    '
    bad_erc20 = get_contract(bad_erc20_code)
    c = get_contract(code)
    assert_tx_failed(lambda : c.transferBorked(bad_erc20.address, c.address, 0))
    assert c.safeTransfer(bad_erc20.address, c.address, 0) == 7
    negative_contract = get_contract(negative_transfer_code)
    assert_tx_failed(lambda : c.safeTransfer(negative_contract.address, c.address, 0))
    random_address = '0x0000000000000000000000000000000000001234'
    assert_tx_failed(lambda : c.safeTransfer(random_address, c.address, 1))
    self_destructing_contract = get_contract(self_destructing_code)
    assert c.safeTransfer(self_destructing_contract.address, c.address, 0) == 7

def test_default_override2(get_contract, assert_tx_failed):
    if False:
        while True:
            i = 10
    bad_code_1 = '\n@external\ndef return_64_bytes() -> bool:\n    return True\n    '
    bad_code_2 = '\n@external\ndef return_64_bytes():\n    pass\n    '
    code = '\nstruct BoolPair:\n    x: bool\n    y: bool\ninterface Foo:\n    def return_64_bytes() -> BoolPair: nonpayable\n@external\ndef bar(foo: Foo):\n    t: BoolPair = foo.return_64_bytes(default_return_value=BoolPair({x: True, y:True}))\n    assert t.x and t.y\n    '
    bad_1 = get_contract(bad_code_1)
    bad_2 = get_contract(bad_code_2)
    c = get_contract(code)
    assert_tx_failed(lambda : c.bar(bad_1.address))
    c.bar(bad_2.address)

def test_contract_address_evaluation(get_contract):
    if False:
        print('Hello World!')
    callee_code = '\n# implements: Foo\n\ninterface Counter:\n    def increment_counter(): nonpayable\n\n@external\ndef foo():\n    pass\n\n@external\ndef bar() -> address:\n    Counter(msg.sender).increment_counter()\n    return self\n    '
    code = '\n# implements: Counter\n\ninterface Foo:\n    def foo(): nonpayable\n    def bar() -> address: nonpayable\n\ncounter: uint256\n\n@external\ndef increment_counter():\n    self.counter += 1\n\n@external\ndef do_stuff(f: Foo) -> uint256:\n    Foo(f.bar()).foo()\n    return self.counter\n    '
    c1 = get_contract(code)
    c2 = get_contract(callee_code)
    assert c1.do_stuff(c2.address) == 1
TEST_ADDR = b''.join((chr(i).encode('utf-8') for i in range(20))).hex()

@pytest.mark.parametrize('typ,val', [('address', TEST_ADDR)])
def test_calldata_clamp(w3, get_contract, assert_tx_failed, keccak, typ, val):
    if False:
        i = 10
        return i + 15
    code = f'\n@external\ndef foo(a: {typ}):\n    pass\n    '
    c1 = get_contract(code)
    sig = keccak(f'foo({typ})'.encode()).hex()[:10]
    encoded = abi.encode(f'({typ})', (val,)).hex()
    data = f'{sig}{encoded}'
    malformed = data[:-2]
    assert_tx_failed(lambda : w3.eth.send_transaction({'to': c1.address, 'data': malformed}))
    w3.eth.send_transaction({'to': c1.address, 'data': data})
    w3.eth.send_transaction({'to': c1.address, 'data': data + 'ff'})

@pytest.mark.parametrize('typ,val', [('address', ([TEST_ADDR] * 3, 'vyper'))])
def test_dynamic_calldata_clamp(w3, get_contract, assert_tx_failed, keccak, typ, val):
    if False:
        while True:
            i = 10
    code = f'\n@external\ndef foo(a: DynArray[{typ}, 3], b: String[5]):\n    pass\n    '
    c1 = get_contract(code)
    sig = keccak(f'foo({typ}[],string)'.encode()).hex()[:10]
    encoded = abi.encode(f'({typ}[],string)', val).hex()
    data = f'{sig}{encoded}'
    malformed = data[:264]
    assert_tx_failed(lambda : w3.eth.send_transaction({'to': c1.address, 'data': malformed}))
    valid = data[:266]
    w3.eth.send_transaction({'to': c1.address, 'data': valid})