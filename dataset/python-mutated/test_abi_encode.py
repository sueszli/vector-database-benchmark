from decimal import Decimal
import pytest
from eth.codecs import abi

def test_abi_encode(get_contract):
    if False:
        print('Hello World!')
    code = '\nstruct Animal:\n  name: String[5]\n  address_: address\n  id_: int128\n  is_furry: bool\n  price: decimal\n  data: uint256[3]\n  metadata: bytes32\n\nstruct Human:\n  name: String[64]\n  pet: Animal\n\n@external\n# TODO accept struct input once the functionality is available\ndef abi_encode(\n    name: String[64],\n    pet_name: String[5],\n    pet_address: address,\n    pet_id: int128,\n    pet_is_furry: bool,\n    pet_price: decimal,\n    pet_data: uint256[3],\n    pet_metadata: bytes32,\n    ensure_tuple: bool,\n    include_method_id: bool\n) -> Bytes[548]:\n    human: Human = Human({\n      name: name,\n      pet: Animal({\n        name: pet_name,\n        address_: pet_address,\n        id_: pet_id,\n        is_furry: pet_is_furry,\n        price: pet_price,\n        data: pet_data,\n        metadata: pet_metadata\n      }),\n    })\n    if ensure_tuple:\n        if not include_method_id:\n            return _abi_encode(human) # default ensure_tuple=True\n        return _abi_encode(human, method_id=0xdeadbeef)\n    else:\n        if not include_method_id:\n            return _abi_encode(human, ensure_tuple=False)\n        return _abi_encode(human, ensure_tuple=False, method_id=0xdeadbeef)\n\n@external\ndef abi_encode2(name: String[32], ensure_tuple: bool, include_method_id: bool) -> Bytes[100]:\n    if ensure_tuple:\n        if not include_method_id:\n            return _abi_encode(name) # default ensure_tuple=True\n        return _abi_encode(name, method_id=0xdeadbeef)\n    else:\n        if not include_method_id:\n            return _abi_encode(name, ensure_tuple=False)\n        return _abi_encode(name, ensure_tuple=False, method_id=0xdeadbeef)\n\n@external\ndef abi_encode3(x: uint256, ensure_tuple: bool, include_method_id: bool) -> Bytes[36]:\n\n    if ensure_tuple:\n        if not include_method_id:\n            return _abi_encode(x) # default ensure_tuple=True\n\n        return _abi_encode(x, method_id=0xdeadbeef)\n\n    else:\n        if not include_method_id:\n            return _abi_encode(x, ensure_tuple=False)\n\n        return _abi_encode(x, ensure_tuple=False, method_id=0xdeadbeef)\n    '
    c = get_contract(code)
    method_id = 3735928559 .to_bytes(4, 'big')
    arg = 123
    assert c.abi_encode3(arg, False, False).hex() == abi.encode('uint256', arg).hex()
    assert c.abi_encode3(arg, True, False).hex() == abi.encode('(uint256)', (arg,)).hex()
    assert c.abi_encode3(arg, False, True).hex() == (method_id + abi.encode('uint256', arg)).hex()
    assert c.abi_encode3(arg, True, True).hex() == (method_id + abi.encode('(uint256)', (arg,))).hex()
    arg = 'some string'
    assert c.abi_encode2(arg, False, False).hex() == abi.encode('string', arg).hex()
    assert c.abi_encode2(arg, True, False).hex() == abi.encode('(string)', (arg,)).hex()
    assert c.abi_encode2(arg, False, True).hex() == (method_id + abi.encode('string', arg)).hex()
    assert c.abi_encode2(arg, True, True).hex() == (method_id + abi.encode('(string)', (arg,))).hex()
    test_addr = '0x' + b''.join((chr(i).encode('utf-8') for i in range(20))).hex()
    test_bytes32 = b''.join((chr(i).encode('utf-8') for i in range(32)))
    human_tuple = ('foobar', ('vyper', test_addr, 123, True, Decimal('123.4'), [123, 456, 789], test_bytes32))
    args = tuple([human_tuple[0]] + list(human_tuple[1]))
    human_t = '(string,(string,address,int128,bool,fixed168x10,uint256[3],bytes32))'
    human_encoded = abi.encode(human_t, human_tuple)
    assert c.abi_encode(*args, False, False).hex() == human_encoded.hex()
    assert c.abi_encode(*args, False, True).hex() == (method_id + human_encoded).hex()
    human_encoded = abi.encode(f'({human_t})', (human_tuple,))
    assert c.abi_encode(*args, True, False).hex() == human_encoded.hex()
    assert c.abi_encode(*args, True, True).hex() == (method_id + human_encoded).hex()

@pytest.mark.parametrize('type,value', [('Bytes', b'hello'), ('String', 'hello')])
def test_abi_encode_length_failing(get_contract, assert_compile_failed, type, value):
    if False:
        for i in range(10):
            print('nop')
    code = f'\nstruct WrappedBytes:\n    bs: {type}[6]\n\n@internal\ndef foo():\n    x: WrappedBytes = WrappedBytes({{bs: {value}}})\n    y: {type}[96] = _abi_encode(x, ensure_tuple=True) # should be Bytes[128]\n    '
    assert_compile_failed(lambda : get_contract(code))

def test_abi_encode_dynarray(get_contract):
    if False:
        return 10
    code = '\n@external\ndef abi_encode(d: DynArray[uint256, 3], ensure_tuple: bool, include_method_id: bool) -> Bytes[164]:\n    if ensure_tuple:\n        if not include_method_id:\n            return _abi_encode(d) # default ensure_tuple=True\n        return _abi_encode(d, method_id=0xdeadbeef)\n    else:\n        if not include_method_id:\n            return _abi_encode(d, ensure_tuple=False)\n        return _abi_encode(d, ensure_tuple=False, method_id=0xdeadbeef)\n    '
    c = get_contract(code)
    method_id = 3735928559 .to_bytes(4, 'big')
    arg = [123, 456, 789]
    assert c.abi_encode(arg, False, False).hex() == abi.encode('uint256[]', arg).hex()
    assert c.abi_encode(arg, True, False).hex() == abi.encode('(uint256[])', (arg,)).hex()
    assert c.abi_encode(arg, False, True).hex() == (method_id + abi.encode('uint256[]', arg)).hex()
    assert c.abi_encode(arg, True, True).hex() == (method_id + abi.encode('(uint256[])', (arg,))).hex()
nested_2d_array_args = [[[123, 456, 789], [234, 567, 891], [345, 678, 912]], [[], [], []], [[123, 456], [234, 567, 891]], [[123, 456, 789], [234, 567], [345]], [[123], [], [345, 678, 912]], [[], [], [345, 678, 912]], [[], [], [345]], [[], [234], []], [[], [234, 567, 891], []], [[]], [[123], [234]]]

@pytest.mark.parametrize('args', nested_2d_array_args)
def test_abi_encode_nested_dynarray(get_contract, args):
    if False:
        i = 10
        return i + 15
    code = '\n@external\ndef abi_encode(\n    d: DynArray[DynArray[uint256, 3], 3], ensure_tuple: bool, include_method_id: bool\n) -> Bytes[548]:\n    if ensure_tuple:\n        if not include_method_id:\n            return _abi_encode(d) # default ensure_tuple=True\n        return _abi_encode(d, method_id=0xdeadbeef)\n    else:\n        if not include_method_id:\n            return _abi_encode(d, ensure_tuple=False)\n        return _abi_encode(d, ensure_tuple=False, method_id=0xdeadbeef)\n    '
    c = get_contract(code)
    method_id = 3735928559 .to_bytes(4, 'big')
    assert c.abi_encode(args, False, False).hex() == abi.encode('uint256[][]', args).hex()
    assert c.abi_encode(args, True, False).hex() == abi.encode('(uint256[][])', (args,)).hex()
    assert c.abi_encode(args, False, True).hex() == (method_id + abi.encode('uint256[][]', args)).hex()
    assert c.abi_encode(args, True, True).hex() == (method_id + abi.encode('(uint256[][])', (args,))).hex()
nested_3d_array_args = [[[[123, 456, 789], [234, 567, 891], [345, 678, 912]], [[234, 567, 891], [345, 678, 912], [123, 456, 789]], [[345, 678, 912], [123, 456, 789], [234, 567, 891]]], [[[123, 789], [234], [345, 678, 912]], [[234, 567], [345, 678]], [[345]]], [[[123], [234, 567, 891]], [[234]]], [[[], [], []], [[], [], []], [[], [], []]], [[[123, 456, 789], [234, 567, 891], [345, 678, 912]], [[234, 567, 891], [345, 678, 912]], [[]]], [[[]], [[]], [[234]]], [[[123]], [[]], [[]]], [[[]], [[123]], [[]]], [[[123, 456, 789], [234, 567]], [[234]], [[567], [912], [345]]], [[[]]]]

@pytest.mark.parametrize('args', nested_3d_array_args)
def test_abi_encode_nested_dynarray_2(get_contract, args):
    if False:
        return 10
    code = '\n@external\ndef abi_encode(\n    d: DynArray[DynArray[DynArray[uint256, 3], 3], 3],\n    ensure_tuple: bool,\n    include_method_id: bool\n) -> Bytes[1700]:\n    if ensure_tuple:\n        if not include_method_id:\n            return _abi_encode(d) # default ensure_tuple=True\n        return _abi_encode(d, method_id=0xdeadbeef)\n    else:\n        if not include_method_id:\n            return _abi_encode(d, ensure_tuple=False)\n        return _abi_encode(d, ensure_tuple=False, method_id=0xdeadbeef)\n    '
    c = get_contract(code)
    method_id = 3735928559 .to_bytes(4, 'big')
    assert c.abi_encode(args, False, False).hex() == abi.encode('uint256[][][]', args).hex()
    assert c.abi_encode(args, True, False).hex() == abi.encode('(uint256[][][])', (args,)).hex()
    assert c.abi_encode(args, False, True).hex() == (method_id + abi.encode('uint256[][][]', args)).hex()
    assert c.abi_encode(args, True, True).hex() == (method_id + abi.encode('(uint256[][][])', (args,))).hex()

def test_side_effects_evaluation(get_contract):
    if False:
        i = 10
        return i + 15
    contract_1 = '\ncounter: uint256\n\n@external\ndef __init__():\n    self.counter = 0\n\n@external\ndef get_counter() -> (uint256, String[6]):\n    self.counter += 1\n    return (self.counter, "hello")\n    '
    c = get_contract(contract_1)
    contract_2 = '\ninterface Foo:\n    def get_counter() -> (uint256, String[6]): nonpayable\n\n@external\ndef foo(addr: address) -> Bytes[164]:\n    return _abi_encode(Foo(addr).get_counter(), method_id=0xdeadbeef)\n    '
    c2 = get_contract(contract_2)
    method_id = 3735928559 .to_bytes(4, 'big')
    get_counter_encoded = abi.encode('((uint256,string))', ((1, 'hello'),))
    assert c2.foo(c.address).hex() == (method_id + get_counter_encoded).hex()

def test_abi_encode_private(get_contract):
    if False:
        i = 10
        return i + 15
    code = '\nbytez: Bytes[96]\n@internal\ndef _foo(bs: Bytes[32]):\n    self.bytez = _abi_encode(bs)\n\n@external\ndef foo(bs: Bytes[32]) -> (uint256, Bytes[96]):\n    dont_clobber_me: uint256 = max_value(uint256)\n    self._foo(bs)\n    return dont_clobber_me, self.bytez\n    '
    c = get_contract(code)
    bs = b'\x00' * 32
    assert c.foo(bs) == [2 ** 256 - 1, abi.encode('(bytes)', (bs,))]

def test_abi_encode_private_dynarray(get_contract):
    if False:
        i = 10
        return i + 15
    code = '\nbytez: Bytes[160]\n@internal\ndef _foo(bs: DynArray[uint256, 3]):\n    self.bytez = _abi_encode(bs)\n@external\ndef foo(bs: DynArray[uint256, 3]) -> (uint256, Bytes[160]):\n    dont_clobber_me: uint256 = max_value(uint256)\n    self._foo(bs)\n    return dont_clobber_me, self.bytez\n    '
    c = get_contract(code)
    bs = [1, 2, 3]
    assert c.foo(bs) == [2 ** 256 - 1, abi.encode('(uint256[])', (bs,))]

def test_abi_encode_private_nested_dynarray(get_contract):
    if False:
        while True:
            i = 10
    code = '\nbytez: Bytes[1696]\n@internal\ndef _foo(bs: DynArray[DynArray[DynArray[uint256, 3], 3], 3]):\n    self.bytez = _abi_encode(bs)\n\n@external\ndef foo(bs: DynArray[DynArray[DynArray[uint256, 3], 3], 3]) -> (uint256, Bytes[1696]):\n    dont_clobber_me: uint256 = max_value(uint256)\n    self._foo(bs)\n    return dont_clobber_me, self.bytez\n    '
    c = get_contract(code)
    bs = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]
    assert c.foo(bs) == [2 ** 256 - 1, abi.encode('(uint256[][][])', (bs,))]

@pytest.mark.parametrize('empty_literal', ('b""', '""', 'empty(Bytes[1])', 'empty(String[1])'))
def test_abi_encode_empty_string(get_contract, empty_literal):
    if False:
        i = 10
        return i + 15
    code = f'\n@external\ndef foo(ensure_tuple: bool) -> Bytes[96]:\n    if ensure_tuple:\n        return _abi_encode({empty_literal}) # default ensure_tuple=True\n    else:\n        return _abi_encode({empty_literal}, ensure_tuple=False)\n    '
    c = get_contract(code)
    expected_output = b'\x00' * 32
    assert c.foo(False) == expected_output
    expected_output = b'\x00' * 31 + b' ' + b'\x00' * 32
    assert c.foo(True) == expected_output