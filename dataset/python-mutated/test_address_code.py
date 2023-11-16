from typing import Type
import pytest
from eth_tester.exceptions import TransactionFailed
from web3 import Web3
from vyper import compiler
from vyper.compiler.settings import Settings
from vyper.exceptions import NamespaceCollision, StructureException, VyperException
PRECOMPILED_ABI = '[{"stateMutability": "view", "type": "function", "name": "hello", "inputs": [], "outputs": [{"name": "", "type": "uint256"}], "gas": 2460}]'
PRECOMPILED_BYTECODE = '0x61004456600436101561000d57610035565b60046000601c376000513461003b576319ff1d2181186100335760005460e052602060e0f35b505b60006000fd5b600080fd5b61000461004403610004600039610004610044036000f3'
PRECOMPILED_BYTECODE_RUNTIME = '0x600436101561000d57610035565b60046000601c376000513461003b576319ff1d2181186100335760005460e052602060e0f35b505b60006000fd5b600080fd'
PRECOMPILED = bytes.fromhex(PRECOMPILED_BYTECODE_RUNTIME[2:])

def _deploy_precompiled_contract(w3: Web3):
    if False:
        while True:
            i = 10
    Precompiled = w3.eth.contract(abi=PRECOMPILED_ABI, bytecode=PRECOMPILED_BYTECODE)
    tx_hash = Precompiled.constructor().transact()
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    address = tx_receipt['contractAddress']
    return w3.eth.contract(address=address, abi=PRECOMPILED_ABI)

@pytest.mark.parametrize(('start', 'length', 'expected'), [(0, 5, PRECOMPILED[:5]), (5, 10, PRECOMPILED[5:][:10])])
def test_address_code_slice(start: int, length: int, expected: bytes, w3: Web3, get_contract):
    if False:
        for i in range(10):
            print('nop')
    code = f'\n@external\ndef code_slice(x: address) -> Bytes[{length}]:\n    return slice(x.code, {start}, {length})\n'
    contract = get_contract(code)
    precompiled_contract = _deploy_precompiled_contract(w3)
    actual = contract.code_slice(precompiled_contract.address)
    assert actual == expected

def test_address_code_runtime_error_slice_too_long(w3: Web3, get_contract):
    if False:
        return 10
    start = len(PRECOMPILED) - 5
    length = 10
    code = f'\n@external\ndef code_slice(x: address) -> Bytes[{length}]:\n    return slice(x.code, {start}, {length})\n'
    contract = get_contract(code)
    precompiled_contract = _deploy_precompiled_contract(w3)
    with pytest.raises(TransactionFailed):
        contract.code_slice(precompiled_contract.address)

def test_address_code_runtime_error_no_code(get_contract):
    if False:
        i = 10
        return i + 15
    code = '\n@external\ndef code_slice(x: address) -> Bytes[4]:\n    return slice(x.code, 0, 4)\n'
    contract = get_contract(code)
    with pytest.raises(TransactionFailed):
        contract.code_slice(b'\x00' * 20)

@pytest.mark.parametrize(('bad_code', 'error_type', 'error_message'), [('\n@external\ndef code_slice(x: address) -> uint256:\n    y: uint256 = convert(x.code, uint256)\n    return y\n', StructureException, '(address).code is only allowed inside of a slice function with a constant length'), ('\na: HashMap[Bytes[4], uint256]\n\n@external\ndef foo(x: address):\n    self.a[x.code] += 1\n', StructureException, '(address).code is only allowed inside of a slice function with a constant length'), ('\n@external\ndef code_slice(x: address) -> uint256:\n    y: uint256 = len(x.code)\n    return y\n', StructureException, '(address).code is only allowed inside of a slice function with a constant length'), ('\n@external\ndef code_slice(x: address, y: uint256) -> Bytes[4]:\n    z: Bytes[4] = slice(x.code, 0, y)\n    return z\n', StructureException, '(address).code is only allowed inside of a slice function with a constant length'), ('\ncode: public(Bytes[4])\n', NamespaceCollision, "Value 'code' has already been declared")])
def test_address_code_compile_error(bad_code: str, error_type: Type[VyperException], error_message: str):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(error_type) as excinfo:
        compiler.compile_code(bad_code)
    assert type(excinfo.value) == error_type
    assert excinfo.value.message == error_message

@pytest.mark.parametrize('code', ['\n@external\ndef foo() -> Bytes[4]:\n    return slice(msg.sender.code, 0, 4)\n', '\nstruct S:\n    a: address\n\n@external\ndef foo(s: S) -> Bytes[4]:\n    return slice(s.a.code, 0, 4)\n', '\ninterface Test:\n    def out_literals() -> address : view\n\n@external\ndef foo(x: address) -> Bytes[4]:\n    return slice(Test(x).out_literals().code, 0, 4)\n'])
def test_address_code_compile_success(code: str):
    if False:
        return 10
    compiler.compile_code(code)

def test_address_code_self_success(get_contract, optimize):
    if False:
        i = 10
        return i + 15
    code = '\ncode_deployment: public(Bytes[32])\n\n@external\ndef __init__():\n    self.code_deployment = slice(self.code, 0, 32)\n\n@external\ndef code_runtime() -> Bytes[32]:\n    return slice(self.code, 0, 32)\n'
    contract = get_contract(code)
    settings = Settings(optimize=optimize)
    code_compiled = compiler.compile_code(code, output_formats=['bytecode', 'bytecode_runtime'], settings=settings)
    assert contract.code_deployment() == bytes.fromhex(code_compiled['bytecode'][2:])[:32]
    assert contract.code_runtime() == bytes.fromhex(code_compiled['bytecode_runtime'][2:])[:32]

def test_address_code_self_runtime_error_deployment(get_contract):
    if False:
        while True:
            i = 10
    code = '\ndummy: public(Bytes[1000000])\n\n@external\ndef __init__():\n    self.dummy = slice(self.code, 0, 1000000)\n'
    with pytest.raises(TransactionFailed):
        get_contract(code)

def test_address_code_self_runtime_error_runtime(get_contract):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef code_runtime() -> Bytes[1000000]:\n    return slice(self.code, 0, 1000000)\n'
    contract = get_contract(code)
    with pytest.raises(TransactionFailed):
        contract.code_runtime()