import pytest
import rlp
from eth.codecs import abi
from hexbytes import HexBytes
import vyper.ir.compile_ir as compile_ir
from vyper.codegen.ir_node import IRnode
from vyper.compiler.settings import OptimizationLevel
from vyper.utils import EIP_170_LIMIT, checksum_encode, keccak256

def eip1167_initcode(_addr):
    if False:
        i = 10
        return i + 15
    addr = HexBytes(_addr)
    pre = HexBytes('0x602D3D8160093D39F3363d3d373d3d3d363d73')
    post = HexBytes('0x5af43d82803e903d91602b57fd5bf3')
    return HexBytes(pre + (addr + HexBytes(0) * (20 - len(addr))) + post)

def vyper_initcode(runtime_bytecode):
    if False:
        i = 10
        return i + 15
    bytecode_len_hex = hex(len(runtime_bytecode))[2:].rjust(6, '0')
    return HexBytes('0x62' + bytecode_len_hex + '3d81600b3d39f3') + runtime_bytecode

def test_create_minimal_proxy_to_create(get_contract):
    if False:
        print('Hello World!')
    code = '\nmain: address\n\n@external\ndef test() -> address:\n    self.main = create_minimal_proxy_to(self)\n    return self.main\n    '
    c = get_contract(code)
    address_bits = int(c.address, 16)
    nonce = 1
    rlp_encoded = rlp.encode([address_bits, nonce])
    expected_create_address = keccak256(rlp_encoded)[12:].rjust(20, b'\x00')
    assert c.test() == checksum_encode('0x' + expected_create_address.hex())

def test_create_minimal_proxy_to_call(get_contract, w3):
    if False:
        return 10
    code = '\n\ninterface SubContract:\n\n    def hello() -> Bytes[100]: view\n\n\nother: public(address)\n\n\n@external\ndef test() -> address:\n    self.other = create_minimal_proxy_to(self)\n    return self.other\n\n\n@external\ndef hello() -> Bytes[100]:\n    return b"hello world!"\n\n\n@external\ndef test2() -> Bytes[100]:\n    return SubContract(self.other).hello()\n\n    '
    c = get_contract(code)
    assert c.hello() == b'hello world!'
    c.test(transact={})
    assert c.test2() == b'hello world!'

def test_minimal_proxy_exception(w3, get_contract, assert_tx_failed):
    if False:
        i = 10
        return i + 15
    code = '\n\ninterface SubContract:\n\n    def hello(a: uint256) -> Bytes[100]: view\n\n\nother: public(address)\n\n\n@external\ndef test() -> address:\n    self.other = create_minimal_proxy_to(self)\n    return self.other\n\n\n@external\ndef hello(a: uint256) -> Bytes[100]:\n    assert a > 0, "invaliddddd"\n    return b"hello world!"\n\n\n@external\ndef test2(a: uint256) -> Bytes[100]:\n    return SubContract(self.other).hello(a)\n    '
    c = get_contract(code)
    assert c.hello(1) == b'hello world!'
    c.test(transact={})
    assert c.test2(1) == b'hello world!'
    assert_tx_failed(lambda : c.test2(0))
    GAS_SENT = 30000
    tx_hash = c.test2(0, transact={'gas': GAS_SENT})
    receipt = w3.eth.get_transaction_receipt(tx_hash)
    assert receipt['status'] == 0
    assert receipt['gasUsed'] < GAS_SENT

def test_create_minimal_proxy_to_create2(get_contract, create2_address_of, keccak, assert_tx_failed):
    if False:
        return 10
    code = '\nmain: address\n\n@external\ndef test(_salt: bytes32) -> address:\n    self.main = create_minimal_proxy_to(self, salt=_salt)\n    return self.main\n    '
    c = get_contract(code)
    salt = keccak(b'vyper')
    assert HexBytes(c.test(salt)) == create2_address_of(c.address, salt, eip1167_initcode(c.address))
    c.test(salt, transact={})
    assert_tx_failed(lambda : c.test(salt, transact={}))

@pytest.mark.parametrize('blueprint_prefix', [b'', b'\xfe', b'\xfe9\x00'])
def test_create_from_blueprint(get_contract, deploy_blueprint_for, w3, keccak, create2_address_of, assert_tx_failed, blueprint_prefix):
    if False:
        i = 10
        return i + 15
    code = '\n@external\ndef foo() -> uint256:\n    return 123\n    '
    prefix_len = len(blueprint_prefix)
    deployer_code = f'\ncreated_address: public(address)\n\n@external\ndef test(target: address):\n    self.created_address = create_from_blueprint(target, code_offset={prefix_len})\n\n@external\ndef test2(target: address, salt: bytes32):\n    self.created_address = create_from_blueprint(target, code_offset={prefix_len}, salt=salt)\n    '
    foo_contract = get_contract(code)
    expected_runtime_code = w3.eth.get_code(foo_contract.address)
    (f, FooContract) = deploy_blueprint_for(code, initcode_prefix=blueprint_prefix)
    d = get_contract(deployer_code)
    d.test(f.address, transact={})
    test = FooContract(d.created_address())
    assert w3.eth.get_code(test.address) == expected_runtime_code
    assert test.foo() == 123
    zero_address = '0x' + '00' * 20
    assert_tx_failed(lambda : d.test(zero_address))
    salt = keccak(b'vyper')
    d.test2(f.address, salt, transact={})
    test = FooContract(d.created_address())
    assert w3.eth.get_code(test.address) == expected_runtime_code
    assert test.foo() == 123
    initcode = w3.eth.get_code(f.address)
    initcode = initcode[len(blueprint_prefix):]
    assert HexBytes(test.address) == create2_address_of(d.address, salt, initcode)
    assert_tx_failed(lambda : d.test2(f.address, salt))

def test_create_from_blueprint_bad_code_offset(get_contract, get_contract_from_ir, deploy_blueprint_for, w3, assert_tx_failed):
    if False:
        while True:
            i = 10
    deployer_code = '\nBLUEPRINT: immutable(address)\n\n@external\ndef __init__(blueprint_address: address):\n    BLUEPRINT = blueprint_address\n\n@external\ndef test(code_ofst: uint256) -> address:\n    return create_from_blueprint(BLUEPRINT, code_offset=code_ofst)\n    '
    initcode_len = 100
    ir = IRnode.from_list(['deploy', 0, ['seq'] + ['stop'] * initcode_len, 0])
    (bytecode, _) = compile_ir.assembly_to_evm(compile_ir.compile_to_assembly(ir, optimize=OptimizationLevel.NONE))
    c = w3.eth.contract(abi=[], bytecode=bytecode)
    deploy_transaction = c.constructor()
    tx_info = {'from': w3.eth.accounts[0], 'value': 0, 'gasPrice': 0}
    tx_hash = deploy_transaction.transact(tx_info)
    blueprint_address = w3.eth.get_transaction_receipt(tx_hash)['contractAddress']
    blueprint_code = w3.eth.get_code(blueprint_address)
    print('BLUEPRINT CODE:', blueprint_code)
    d = get_contract(deployer_code, blueprint_address)
    d.test(0)
    d.test(initcode_len - 1)
    assert_tx_failed(lambda : d.test(initcode_len))
    assert_tx_failed(lambda : d.test(EIP_170_LIMIT))

def test_create_from_blueprint_args(get_contract, deploy_blueprint_for, w3, keccak, create2_address_of, assert_tx_failed):
    if False:
        return 10
    code = '\nstruct Bar:\n    x: String[32]\n\nFOO: immutable(String[128])\nBAR: immutable(Bar)\n\n@external\ndef __init__(foo: String[128], bar: Bar):\n    FOO = foo\n    BAR = bar\n\n@external\ndef foo() -> String[128]:\n    return FOO\n\n@external\ndef bar() -> Bar:\n    return BAR\n    '
    deployer_code = '\nstruct Bar:\n    x: String[32]\n\ncreated_address: public(address)\n\n@external\ndef test(target: address, arg1: String[128], arg2: Bar):\n    self.created_address = create_from_blueprint(target, arg1, arg2)\n\n@external\ndef test2(target: address, arg1: String[128], arg2: Bar, salt: bytes32):\n    self.created_address = create_from_blueprint(target, arg1, arg2, salt=salt)\n\n@external\ndef test3(target: address, argdata: Bytes[1024]):\n    self.created_address = create_from_blueprint(target, argdata, raw_args=True)\n\n@external\ndef test4(target: address, argdata: Bytes[1024], salt: bytes32):\n    self.created_address = create_from_blueprint(target, argdata, salt=salt, raw_args=True)\n\n@external\ndef should_fail(target: address, arg1: String[129], arg2: Bar):\n    self.created_address = create_from_blueprint(target, arg1, arg2)\n    '
    FOO = 'hello!'
    BAR = ('world!',)
    foo_contract = get_contract(code, FOO, BAR)
    expected_runtime_code = w3.eth.get_code(foo_contract.address)
    (f, FooContract) = deploy_blueprint_for(code)
    d = get_contract(deployer_code)
    initcode = w3.eth.get_code(f.address)
    d.test(f.address, FOO, BAR, transact={})
    test = FooContract(d.created_address())
    assert w3.eth.get_code(test.address) == expected_runtime_code
    assert test.foo() == FOO
    assert test.bar() == BAR
    assert_tx_failed(lambda : d.test('0x' + '00' * 20, FOO, BAR))
    salt = keccak(b'vyper')
    d.test2(f.address, FOO, BAR, salt, transact={})
    test = FooContract(d.created_address())
    assert w3.eth.get_code(test.address) == expected_runtime_code
    assert test.foo() == FOO
    assert test.bar() == BAR
    encoded_args = abi.encode('(string,(string))', (FOO, BAR))
    assert HexBytes(test.address) == create2_address_of(d.address, salt, initcode + encoded_args)
    d.test3(f.address, encoded_args, transact={})
    test = FooContract(d.created_address())
    assert w3.eth.get_code(test.address) == expected_runtime_code
    assert test.foo() == FOO
    assert test.bar() == BAR
    d.test4(f.address, encoded_args, keccak(b'test4'), transact={})
    test = FooContract(d.created_address())
    assert w3.eth.get_code(test.address) == expected_runtime_code
    assert test.foo() == FOO
    assert test.bar() == BAR
    assert_tx_failed(lambda : d.test2(f.address, FOO, BAR, salt))
    assert_tx_failed(lambda : d.test4(f.address, encoded_args, salt))
    FOO = 'bar'
    d.test2(f.address, FOO, BAR, salt, transact={})
    assert FooContract(d.created_address()).foo() == FOO
    assert FooContract(d.created_address()).bar() == BAR
    FOO = '01' * 129
    BAR = ('',)
    sig = keccak('should_fail(address,string,(string))'.encode()).hex()[:10]
    encoded = abi.encode('(address,string,(string))', (f.address, FOO, BAR)).hex()
    assert_tx_failed(lambda : w3.eth.send_transaction({'to': d.address, 'data': f'{sig}{encoded}'}))

def test_create_copy_of(get_contract, w3, keccak, create2_address_of, assert_tx_failed):
    if False:
        print('Hello World!')
    code = '\ncreated_address: public(address)\n@internal\ndef _create_copy_of(target: address):\n    self.created_address = create_copy_of(target)\n\n@internal\ndef _create_copy_of2(target: address, salt: bytes32):\n    self.created_address = create_copy_of(target, salt=salt)\n\n@external\ndef test(target: address) -> address:\n    x: uint256 = 0\n    self._create_copy_of(target)\n    assert x == 0  # check memory not clobbered\n    return self.created_address\n\n@external\ndef test2(target: address, salt: bytes32) -> address:\n    x: uint256 = 0\n    self._create_copy_of2(target, salt)\n    assert x == 0  # check memory not clobbered\n    return self.created_address\n    '
    c = get_contract(code)
    bytecode = w3.eth.get_code(c.address)
    c.test(c.address, transact={})
    test1 = c.created_address()
    assert w3.eth.get_code(test1) == bytecode
    assert_tx_failed(lambda : c.test('0x' + '00' * 20))
    salt = keccak(b'vyper')
    c.test2(c.address, salt, transact={})
    test2 = c.created_address()
    assert w3.eth.get_code(test2) == bytecode
    assert HexBytes(test2) == create2_address_of(c.address, salt, vyper_initcode(bytecode))
    assert_tx_failed(lambda : c.test2(c.address, salt, transact={}))

@pytest.mark.parametrize('blueprint_prefix', [b'', b'\xfe', b'\xfe9\x00'])
def test_create_from_blueprint_complex_value(get_contract, deploy_blueprint_for, w3, blueprint_prefix):
    if False:
        i = 10
        return i + 15
    code = '\nvar: uint256\n\n@external\n@payable\ndef __init__(x: uint256):\n    self.var = x\n\n@external\ndef foo()-> uint256:\n    return self.var\n    '
    prefix_len = len(blueprint_prefix)
    some_constant = b'\x00' * 31 + b'\x0c'
    deployer_code = f'\ncreated_address: public(address)\nx: constant(Bytes[32]) = {some_constant}\n\n@internal\ndef foo() -> uint256:\n    g:uint256 = 42\n    return 3\n\n@external\n@payable\ndef test(target: address):\n    self.created_address = create_from_blueprint(\n        target,\n        x,\n        code_offset={prefix_len},\n        value=self.foo(),\n        raw_args=True\n    )\n    '
    foo_contract = get_contract(code, 12)
    expected_runtime_code = w3.eth.get_code(foo_contract.address)
    (f, FooContract) = deploy_blueprint_for(code, initcode_prefix=blueprint_prefix)
    d = get_contract(deployer_code)
    d.test(f.address, transact={'value': 3})
    test = FooContract(d.created_address())
    assert w3.eth.get_code(test.address) == expected_runtime_code
    assert test.foo() == 12

@pytest.mark.parametrize('blueprint_prefix', [b'', b'\xfe', b'\xfe9\x00'])
def test_create_from_blueprint_complex_salt_raw_args(get_contract, deploy_blueprint_for, w3, blueprint_prefix):
    if False:
        for i in range(10):
            print('nop')
    code = '\nvar: uint256\n\n@external\n@payable\ndef __init__(x: uint256):\n    self.var = x\n\n@external\ndef foo()-> uint256:\n    return self.var\n    '
    some_constant = b'\x00' * 31 + b'\x0c'
    prefix_len = len(blueprint_prefix)
    deployer_code = f'\ncreated_address: public(address)\n\nx: constant(Bytes[32]) = {some_constant}\nsalt: constant(bytes32) = keccak256("kebab")\n\n@internal\ndef foo() -> bytes32:\n    g:uint256 = 42\n    return salt\n\n@external\n@payable\ndef test(target: address):\n    self.created_address = create_from_blueprint(\n        target,\n        x,\n        code_offset={prefix_len},\n        salt=self.foo(),\n        raw_args= True\n    )\n    '
    foo_contract = get_contract(code, 12)
    expected_runtime_code = w3.eth.get_code(foo_contract.address)
    (f, FooContract) = deploy_blueprint_for(code, initcode_prefix=blueprint_prefix)
    d = get_contract(deployer_code)
    d.test(f.address, transact={})
    test = FooContract(d.created_address())
    assert w3.eth.get_code(test.address) == expected_runtime_code
    assert test.foo() == 12

@pytest.mark.parametrize('blueprint_prefix', [b'', b'\xfe', b'\xfe9\x00'])
def test_create_from_blueprint_complex_salt_no_constructor_args(get_contract, deploy_blueprint_for, w3, blueprint_prefix):
    if False:
        while True:
            i = 10
    code = '\nvar: uint256\n\n@external\n@payable\ndef __init__():\n    self.var = 12\n\n@external\ndef foo()-> uint256:\n    return self.var\n    '
    prefix_len = len(blueprint_prefix)
    deployer_code = f'\ncreated_address: public(address)\n\nsalt: constant(bytes32) = keccak256("kebab")\n\n@external\n@payable\ndef test(target: address):\n    self.created_address = create_from_blueprint(\n        target,\n        code_offset={prefix_len},\n        salt=keccak256(_abi_encode(target))\n    )\n    '
    foo_contract = get_contract(code)
    expected_runtime_code = w3.eth.get_code(foo_contract.address)
    (f, FooContract) = deploy_blueprint_for(code, initcode_prefix=blueprint_prefix)
    d = get_contract(deployer_code)
    d.test(f.address, transact={})
    test = FooContract(d.created_address())
    assert w3.eth.get_code(test.address) == expected_runtime_code
    assert test.foo() == 12

def test_create_copy_of_complex_kwargs(get_contract, w3):
    if False:
        for i in range(10):
            print('nop')
    complex_salt = '\ncreated_address: public(address)\n\n@external\ndef test(target: address) -> address:\n    self.created_address = create_copy_of(\n        target,\n        salt=keccak256(_abi_encode(target))\n    )\n    return self.created_address\n\n    '
    c = get_contract(complex_salt)
    bytecode = w3.eth.get_code(c.address)
    c.test(c.address, transact={})
    test1 = c.created_address()
    assert w3.eth.get_code(test1) == bytecode
    complex_value = '\ncreated_address: public(address)\n\n@external\n@payable\ndef test(target: address) -> address:\n    value: uint256 = 2\n    self.created_address = create_copy_of(target, value = [2,2,2][value])\n    return self.created_address\n\n    '
    c = get_contract(complex_value)
    bytecode = w3.eth.get_code(c.address)
    c.test(c.address, transact={'value': 2})
    test1 = c.created_address()
    assert w3.eth.get_code(test1) == bytecode