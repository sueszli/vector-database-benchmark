from decimal import Decimal
import pytest
from eth.codecs import abi
from eth_utils import keccak
from vyper.evm.opcodes import EVM_VERSIONS
from vyper.utils import int_bounds

def _make_tx(w3, address, signature, values):
    if False:
        while True:
            i = 10
    sig = keccak(signature.encode()).hex()[:8]
    data = ''.join((int(i).to_bytes(32, 'big', signed=i < 0).hex() for i in values))
    w3.eth.send_transaction({'to': address, 'data': f'0x{sig}{data}'})

def _make_abi_encode_tx(w3, address, signature, input_types, values):
    if False:
        return 10
    sig = keccak(signature.encode()).hex()[:8]
    data = abi.encode(input_types, values).hex()
    w3.eth.send_transaction({'to': address, 'data': f'0x{sig}{data}'})

def _make_dynarray_data(offset, length, values):
    if False:
        while True:
            i = 10
    input = [offset] + [length] + values
    data = ''.join((int(i).to_bytes(32, 'big', signed=i < 0).hex() for i in input))
    return data

def _make_invalid_dynarray_tx(w3, address, signature, data):
    if False:
        for i in range(10):
            print('nop')
    sig = keccak(signature.encode()).hex()[:8]
    w3.eth.send_transaction({'to': address, 'data': f'0x{sig}{data}'})

def test_bytes_clamper(assert_tx_failed, get_contract_with_gas_estimation):
    if False:
        return 10
    clamper_test_code = '\n@external\ndef foo(s: Bytes[3]) -> Bytes[3]:\n    return s\n    '
    c = get_contract_with_gas_estimation(clamper_test_code)
    assert c.foo(b'ca') == b'ca'
    assert c.foo(b'cat') == b'cat'
    assert_tx_failed(lambda : c.foo(b'cate'))

def test_bytes_clamper_multiple_slots(assert_tx_failed, get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    clamper_test_code = '\n@external\ndef foo(s: Bytes[40]) -> Bytes[40]:\n    return s\n    '
    data = b'this is exactly forty characters long!!!'
    c = get_contract_with_gas_estimation(clamper_test_code)
    assert c.foo(data[:30]) == data[:30]
    assert c.foo(data) == data
    assert_tx_failed(lambda : c.foo(data + b'!'))

def test_bytes_clamper_on_init(assert_tx_failed, get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    clamper_test_code = '\nfoo: Bytes[3]\n\n@external\ndef __init__(x: Bytes[3]):\n    self.foo = x\n\n@external\ndef get_foo() -> Bytes[3]:\n    return self.foo\n    '
    c = get_contract_with_gas_estimation(clamper_test_code, *[b'cat'])
    assert c.get_foo() == b'cat'
    assert_tx_failed(lambda : get_contract_with_gas_estimation(clamper_test_code, *[b'cats']))

@pytest.mark.parametrize('evm_version', list(EVM_VERSIONS))
@pytest.mark.parametrize('n', list(range(1, 33)))
def test_bytes_m_clamper_passing(w3, get_contract, n, evm_version):
    if False:
        return 10
    values = [b'\xff' * (i + 1) for i in range(n)]
    code = f'\n@external\ndef foo(s: bytes{n}) -> bytes{n}:\n    return s\n    '
    c = get_contract(code, evm_version=evm_version)
    for v in values:
        v = v.ljust(n, b'\x00')
        assert c.foo(v) == v

@pytest.mark.parametrize('evm_version', list(EVM_VERSIONS))
@pytest.mark.parametrize('n', list(range(1, 32)))
def test_bytes_m_clamper_failing(w3, get_contract, assert_tx_failed, n, evm_version):
    if False:
        for i in range(10):
            print('nop')
    values = []
    values.append(b'\x00' * n + b'\x80')
    values.append(b'\xff' * n + b'\x80')
    values.append(b'\x00' * 31 + b'\x01')
    values.append(b'\xff' * 32)
    values.append(bytes(range(32)))
    values.append(bytes(range(1, 33)))
    values.append(b'\xff' * 32)
    code = f'\n@external\ndef foo(s: bytes{n}) -> bytes{n}:\n    return s\n    '
    c = get_contract(code, evm_version=evm_version)
    for v in values:
        v = int.from_bytes(v, byteorder='big')
        assert_tx_failed(lambda : _make_tx(w3, c.address, f'foo(bytes{n})', [v]))

@pytest.mark.parametrize('evm_version', list(EVM_VERSIONS))
@pytest.mark.parametrize('n', list(range(32)))
def test_sint_clamper_passing(w3, get_contract, n, evm_version):
    if False:
        for i in range(10):
            print('nop')
    bits = 8 * (n + 1)
    (lo, hi) = int_bounds(True, bits)
    values = [-1, 0, 1, lo, hi]
    code = f'\n@external\ndef foo(s: int{bits}) -> int{bits}:\n    return s\n    '
    c = get_contract(code, evm_version=evm_version)
    for v in values:
        assert c.foo(v) == v

@pytest.mark.parametrize('evm_version', list(EVM_VERSIONS))
@pytest.mark.parametrize('n', list(range(31)))
def test_sint_clamper_failing(w3, assert_tx_failed, get_contract, n, evm_version):
    if False:
        print('Hello World!')
    bits = 8 * (n + 1)
    (lo, hi) = int_bounds(True, bits)
    values = [-2 ** 255, 2 ** 255 - 1, lo - 1, hi + 1]
    code = f'\n@external\ndef foo(s: int{bits}) -> int{bits}:\n    return s\n    '
    c = get_contract(code, evm_version=evm_version)
    for v in values:
        assert_tx_failed(lambda : _make_tx(w3, c.address, f'foo(int{bits})', [v]))

@pytest.mark.parametrize('evm_version', list(EVM_VERSIONS))
@pytest.mark.parametrize('value', [True, False])
def test_bool_clamper_passing(w3, get_contract, value, evm_version):
    if False:
        return 10
    code = '\n@external\ndef foo(s: bool) -> bool:\n    return s\n    '
    c = get_contract(code, evm_version=evm_version)
    assert c.foo(value) == value

@pytest.mark.parametrize('evm_version', list(EVM_VERSIONS))
@pytest.mark.parametrize('value', [2, 3, 4, 8, 16, 2 ** 256 - 1])
def test_bool_clamper_failing(w3, assert_tx_failed, get_contract, value, evm_version):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\ndef foo(s: bool) -> bool:\n    return s\n    '
    c = get_contract(code, evm_version=evm_version)
    assert_tx_failed(lambda : _make_tx(w3, c.address, 'foo(bool)', [value]))

@pytest.mark.parametrize('evm_version', list(EVM_VERSIONS))
@pytest.mark.parametrize('value', [0] + [2 ** i for i in range(5)])
def test_enum_clamper_passing(w3, get_contract, value, evm_version):
    if False:
        while True:
            i = 10
    code = '\nenum Roles:\n    USER\n    STAFF\n    ADMIN\n    MANAGER\n    CEO\n\n@external\ndef foo(s: Roles) -> Roles:\n    return s\n    '
    c = get_contract(code, evm_version=evm_version)
    assert c.foo(value) == value

@pytest.mark.parametrize('evm_version', list(EVM_VERSIONS))
@pytest.mark.parametrize('value', [2 ** i for i in range(5, 256)])
def test_enum_clamper_failing(w3, assert_tx_failed, get_contract, value, evm_version):
    if False:
        for i in range(10):
            print('nop')
    code = '\nenum Roles:\n    USER\n    STAFF\n    ADMIN\n    MANAGER\n    CEO\n\n@external\ndef foo(s: Roles) -> Roles:\n    return s\n    '
    c = get_contract(code, evm_version=evm_version)
    assert_tx_failed(lambda : _make_tx(w3, c.address, 'foo(uint256)', [value]))

@pytest.mark.parametrize('evm_version', list(EVM_VERSIONS))
@pytest.mark.parametrize('n', list(range(32)))
def test_uint_clamper_passing(w3, get_contract, evm_version, n):
    if False:
        while True:
            i = 10
    bits = 8 * (n + 1)
    values = [0, 1, 2 ** bits - 1]
    code = f'\n@external\ndef foo(s: uint{bits}) -> uint{bits}:\n    return s\n    '
    c = get_contract(code, evm_version=evm_version)
    for v in values:
        assert c.foo(v) == v

@pytest.mark.parametrize('evm_version', list(EVM_VERSIONS))
@pytest.mark.parametrize('n', list(range(31)))
def test_uint_clamper_failing(w3, assert_tx_failed, get_contract, evm_version, n):
    if False:
        i = 10
        return i + 15
    bits = 8 * (n + 1)
    values = [-1, -2 ** 255, 2 ** bits]
    code = f'\n@external\ndef foo(s: uint{bits}) -> uint{bits}:\n    return s\n    '
    c = get_contract(code, evm_version=evm_version)
    for v in values:
        assert_tx_failed(lambda : _make_tx(w3, c.address, f'foo(uint{bits})', [v]))

@pytest.mark.parametrize('evm_version', list(EVM_VERSIONS))
@pytest.mark.parametrize('value,expected', [('0x0000000000000000000000000000000000000000', None), ('0x0000000000000000000000000000000000000001', '0x0000000000000000000000000000000000000001'), ('0xFFfFfFffFFfffFFfFFfFFFFFffFFFffffFfFFFfF', '0xFFfFfFffFFfffFFfFFfFFFFFffFFFffffFfFFFfF')])
def test_address_clamper_passing(w3, get_contract, value, expected, evm_version):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo(s: address) -> address:\n    return s\n    '
    c = get_contract(code, evm_version=evm_version)
    assert c.foo(value) == expected

@pytest.mark.parametrize('evm_version', list(EVM_VERSIONS))
@pytest.mark.parametrize('value', [2 ** 160, 2 ** 256 - 1])
def test_address_clamper_failing(w3, assert_tx_failed, get_contract, value, evm_version):
    if False:
        i = 10
        return i + 15
    code = '\n@external\ndef foo(s: address) -> address:\n    return s\n    '
    c = get_contract(code, evm_version=evm_version)
    assert_tx_failed(lambda : _make_tx(w3, c.address, 'foo(address)', [value]))

@pytest.mark.parametrize('evm_version', list(EVM_VERSIONS))
@pytest.mark.parametrize('value', [0, 1, -1, Decimal(2 ** 167 - 1) / 10 ** 10, -Decimal(2 ** 167) / 10 ** 10, '0.0', '1.0', '-1.0', '0.0000000001', '0.9999999999', '-0.0000000001', '-0.9999999999', '18707220957835557353007165858768422651595.9365500927', '-18707220957835557353007165858768422651595.9365500928'])
def test_decimal_clamper_passing(get_contract, value, evm_version):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo(s: decimal) -> decimal:\n    return s\n    '
    c = get_contract(code, evm_version=evm_version)
    assert c.foo(Decimal(value)) == Decimal(value)

@pytest.mark.parametrize('evm_version', list(EVM_VERSIONS))
@pytest.mark.parametrize('value', [2 ** 167, -(2 ** 167 + 1), 187072209578355573530071658587684226515959365500928, -187072209578355573530071658587684226515959365500929])
def test_decimal_clamper_failing(w3, assert_tx_failed, get_contract, value, evm_version):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo(s: decimal) -> decimal:\n    return s\n    '
    c = get_contract(code, evm_version=evm_version)
    assert_tx_failed(lambda : _make_tx(w3, c.address, 'foo(fixed168x10)', [value]))

@pytest.mark.parametrize('value', [0, 1, -1, 2 ** 127 - 1, -2 ** 127])
def test_int128_array_clamper_passing(w3, get_contract, value):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo(a: uint256, b: int128[5], c: uint256) -> int128[5]:\n    return b\n    '
    d = [value] * 5
    c = get_contract(code)
    assert c.foo(2 ** 127, [value] * 5, 2 ** 127) == d

@pytest.mark.parametrize('bad_value', [2 ** 127, -2 ** 127 - 1, 2 ** 255 - 1, -2 ** 255])
@pytest.mark.parametrize('idx', range(5))
def test_int128_array_clamper_failing(w3, assert_tx_failed, get_contract, bad_value, idx):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo(b: int128[5]) -> int128[5]:\n    return b\n    '
    values = [0] * 5
    values[idx] = bad_value
    c = get_contract(code)
    assert_tx_failed(lambda : _make_tx(w3, c.address, 'foo(int128[5])', values))

@pytest.mark.parametrize('value', [0, 1, -1, 2 ** 127 - 1, -2 ** 127])
def test_int128_array_looped_clamper_passing(w3, get_contract, value):
    if False:
        print('Hello World!')
    code = '\n@external\ndef foo(a: uint256, b: int128[10], c: uint256) -> int128[10]:\n    return b\n    '
    d = [value] * 10
    c = get_contract(code)
    assert c.foo(2 ** 127, d, 2 ** 127) == d

@pytest.mark.parametrize('bad_value', [2 ** 127, -2 ** 127 - 1, 2 ** 255 - 1, -2 ** 255])
@pytest.mark.parametrize('idx', range(10))
def test_int128_array_looped_clamper_failing(w3, assert_tx_failed, get_contract, bad_value, idx):
    if False:
        return 10
    code = '\n@external\ndef foo(b: int128[10]) -> int128[10]:\n    return b\n    '
    values = [0] * 10
    values[idx] = bad_value
    c = get_contract(code)
    assert_tx_failed(lambda : _make_tx(w3, c.address, 'foo(int128[10])', values))

@pytest.mark.parametrize('value', [0, 1, -1, 2 ** 127 - 1, -2 ** 127])
def test_multidimension_array_clamper_passing(w3, get_contract, value):
    if False:
        print('Hello World!')
    code = '\n@external\ndef foo(a: uint256, b: int128[6][3][1][8], c: uint256) -> int128[6][3][1][8]:\n    return b\n    '
    d = [[[[value] * 6] * 3] * 1] * 8
    c = get_contract(code)
    assert c.foo(2 ** 127, d, 2 ** 127, call={'gasPrice': 0}) == d

@pytest.mark.parametrize('bad_value', [2 ** 127, -2 ** 127 - 1, 2 ** 255 - 1, -2 ** 255])
@pytest.mark.parametrize('idx', range(12))
def test_multidimension_array_clamper_failing(w3, assert_tx_failed, get_contract, bad_value, idx):
    if False:
        i = 10
        return i + 15
    code = '\n@external\ndef foo(b: int128[6][1][2]) -> int128[6][1][2]:\n    return b\n    '
    values = [0] * 12
    values[idx] = bad_value
    c = get_contract(code)
    assert_tx_failed(lambda : _make_tx(w3, c.address, 'foo(int128[6][1][2]])', values))

@pytest.mark.parametrize('value', [0, 1, -1, 2 ** 127 - 1, -2 ** 127])
def test_int128_dynarray_clamper_passing(w3, get_contract, value):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\ndef foo(a: uint256, b: DynArray[int128, 5], c: uint256) -> DynArray[int128, 5]:\n    return b\n    '
    d = [value] * 5
    c = get_contract(code)
    assert c.foo(2 ** 127, d, 2 ** 127) == d

@pytest.mark.parametrize('bad_value', [2 ** 127, -2 ** 127 - 1, 2 ** 255 - 1, -2 ** 255])
@pytest.mark.parametrize('idx', range(5))
def test_int128_dynarray_clamper_failing(w3, assert_tx_failed, get_contract, bad_value, idx):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo(b: int128[5]) -> int128[5]:\n    return b\n    '
    values = [0] * 5
    values[idx] = bad_value
    signature = 'foo(int128[])'
    c = get_contract(code)
    data = _make_dynarray_data(32, 5, values)
    assert_tx_failed(lambda : _make_invalid_dynarray_tx(w3, c.address, signature, data))

@pytest.mark.parametrize('value', [0, 1, -1, 2 ** 127 - 1, -2 ** 127])
def test_int128_dynarray_looped_clamper_passing(w3, get_contract, value):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo(a: uint256, b: DynArray[int128, 10], c: uint256) -> DynArray[int128, 10]:\n    return b\n    '
    d = [value] * 10
    c = get_contract(code)
    assert c.foo(2 ** 127, d, 2 ** 127) == d

@pytest.mark.parametrize('bad_value', [2 ** 127, -2 ** 127 - 1, 2 ** 255 - 1, -2 ** 255])
@pytest.mark.parametrize('idx', range(10))
def test_int128_dynarray_looped_clamper_failing(w3, assert_tx_failed, get_contract, bad_value, idx):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo(b: DynArray[int128, 10]) -> DynArray[int128, 10]:\n    return b\n    '
    values = [0] * 10
    values[idx] = bad_value
    c = get_contract(code)
    data = _make_dynarray_data(32, 10, values)
    signature = 'foo(int128[])'
    assert_tx_failed(lambda : _make_invalid_dynarray_tx(w3, c.address, signature, data))

@pytest.mark.parametrize('value', [0, 1, -1, 2 ** 127 - 1, -2 ** 127])
def test_multidimension_dynarray_clamper_passing(w3, get_contract, value):
    if False:
        print('Hello World!')
    code = '\n@external\ndef foo(\n    a: uint256,\n    b: DynArray[DynArray[DynArray[DynArray[int128, 5], 6], 7], 8],\n    c: uint256\n) -> DynArray[DynArray[DynArray[DynArray[int128, 5], 6], 7], 8]:\n    return b\n    '
    d = [[[[value] * 5] * 6] * 7] * 8
    c = get_contract(code)
    assert c.foo(2 ** 127, d, 2 ** 127, call={'gasPrice': 0}) == d

@pytest.mark.parametrize('bad_value', [2 ** 127, -2 ** 127 - 1, 2 ** 255 - 1, -2 ** 255])
@pytest.mark.parametrize('idx', range(4))
def test_multidimension_dynarray_clamper_failing(w3, assert_tx_failed, get_contract, bad_value, idx):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo(b: DynArray[DynArray[int128, 2], 2]) -> DynArray[DynArray[int128, 2], 2]:\n    return b\n    '
    values = [[0] * 2] * 2
    values[idx // 2][idx % 2] = bad_value
    data = _make_dynarray_data(32, 2, [64, 160])
    for v in values:
        v = [2] + v
        inner_data = ''.join((int(_v).to_bytes(32, 'big', signed=_v < 0).hex() for _v in v))
        data += inner_data
    signature = 'foo(int128[][])'
    c = get_contract(code)
    assert_tx_failed(lambda : _make_invalid_dynarray_tx(w3, c.address, signature, data))

@pytest.mark.parametrize('value', [0, 1, -1, 2 ** 127 - 1, -2 ** 127])
def test_dynarray_list_clamper_passing(w3, get_contract, value):
    if False:
        return 10
    code = '\n@external\ndef foo(\n    a: uint256,\n    b: DynArray[int128[5], 6],\n    c: uint256\n) -> DynArray[int128[5], 6]:\n    return b\n    '
    d = [[value] * 5] * 6
    c = get_contract(code)
    assert c.foo(2 ** 127, d, 2 ** 127) == d

@pytest.mark.parametrize('bad_value', [2 ** 127, -2 ** 127 - 1, 2 ** 255 - 1, -2 ** 255])
@pytest.mark.parametrize('idx', range(10))
def test_dynarray_list_clamper_failing(w3, assert_tx_failed, get_contract, bad_value, idx):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo(b: DynArray[int128[5], 2]) -> DynArray[int128[5], 2]:\n    return b\n    '
    values = [[0] * 5, [0] * 5]
    values[idx // 5][idx % 5] = bad_value
    data = _make_dynarray_data(32, 2, [])
    for v in values:
        inner_data = ''.join((int(_v).to_bytes(32, 'big', signed=_v < 0).hex() for _v in v))
        data += inner_data
    c = get_contract(code)
    signature = 'foo(int128[5][])'
    assert_tx_failed(lambda : _make_invalid_dynarray_tx(w3, c.address, signature, data))