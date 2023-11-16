import pytest
from vyper.compiler import compile_code
from vyper.compiler.settings import Settings
from vyper.evm.opcodes import EVM_VERSIONS
from vyper.utils import keccak256

@pytest.mark.parametrize('evm_version', list(EVM_VERSIONS))
def test_get_extcodehash(get_contract, evm_version, optimize):
    if False:
        for i in range(10):
            print('nop')
    code = '\na: address\n\n@external\ndef __init__():\n    self.a = self\n\n@external\ndef foo(x: address) -> bytes32:\n    return x.codehash\n\n@external\ndef foo2(x: address) -> bytes32:\n    b: address = x\n    return b.codehash\n\n@external\ndef foo3() -> bytes32:\n    return self.codehash\n\n@external\ndef foo4() -> bytes32:\n    return self.a.codehash\n    '
    settings = Settings(evm_version=evm_version, optimize=optimize)
    compiled = compile_code(code, output_formats=['bytecode_runtime'], settings=settings)
    bytecode = bytes.fromhex(compiled['bytecode_runtime'][2:])
    hash_ = keccak256(bytecode)
    c = get_contract(code, evm_version=evm_version)
    assert c.foo(c.address) == hash_
    assert not int(c.foo('0xDeaDbeefdEAdbeefdEadbEEFdeadbeEFdEaDbeeF').hex(), 16)
    assert c.foo2(c.address) == hash_
    assert not int(c.foo2('0xDeaDbeefdEAdbeefdEadbEEFdeadbeEFdEaDbeeF').hex(), 16)
    assert c.foo3() == hash_
    assert c.foo4() == hash_