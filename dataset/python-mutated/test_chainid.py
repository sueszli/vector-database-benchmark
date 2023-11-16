import pytest
from vyper import compiler
from vyper.compiler.settings import Settings
from vyper.evm.opcodes import EVM_VERSIONS
from vyper.exceptions import InvalidType, TypeMismatch

@pytest.mark.parametrize('evm_version', list(EVM_VERSIONS))
def test_evm_version(evm_version):
    if False:
        i = 10
        return i + 15
    code = '\n@external\ndef foo():\n    a: uint256 = chain.id\n    '
    settings = Settings(evm_version=evm_version)
    assert compiler.compile_code(code, settings=settings) is not None
fail_list = [('\n@external\ndef foo() -> int128[2]:\n    return [3,chain.id]\n    ', InvalidType), '\n@external\ndef foo() -> decimal:\n    x: int128 = as_wei_value(5, "finney")\n    y: int128 = chain.id + 50\n    return x / y\n    ', '\n@external\ndef foo():\n    x: int128 = 7\n    y: int128 = min(x, chain.id)\n    ', '\na: HashMap[uint256, int128]\n\n@external\ndef add_record():\n    self.a[chain.id] = chain.id + 20\n    ', '\na: HashMap[int128, uint256]\n\n@external\ndef add_record():\n    self.a[chain.id] = chain.id + 20\n    ', ('\n@external\ndef foo(inp: Bytes[10]) -> Bytes[3]:\n    return slice(inp, chain.id, -3)\n    ', InvalidType)]

@pytest.mark.parametrize('bad_code', fail_list)
def test_chain_fail(bad_code):
    if False:
        i = 10
        return i + 15
    if isinstance(bad_code, tuple):
        with pytest.raises(bad_code[1]):
            compiler.compile_code(bad_code[0])
    else:
        with pytest.raises(TypeMismatch):
            compiler.compile_code(bad_code)
valid_list = ['\n@external\n@view\ndef get_chain_id() -> uint256:\n    return chain.id\n    ', '\n@external\n@view\ndef check_chain_id(c: uint256) -> bool:\n    return chain.id == c\n    ']

@pytest.mark.parametrize('good_code', valid_list)
def test_chain_success(good_code):
    if False:
        i = 10
        return i + 15
    assert compiler.compile_code(good_code) is not None

def test_chainid_operation(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    code = '\n@external\n@view\ndef get_chain_id() -> uint256:\n    return chain.id\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.get_chain_id() == 131277322940537