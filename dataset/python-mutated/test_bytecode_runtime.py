import cbor2
import pytest
import vyper
from vyper.compiler.settings import OptimizationLevel, Settings
simple_contract_code = '\n@external\ndef a() -> bool:\n    return True\n'
many_functions = '\n@external\ndef foo1():\n    pass\n\n@external\ndef foo2():\n    pass\n\n@external\ndef foo3():\n    pass\n\n@external\ndef foo4():\n    pass\n\n@external\ndef foo5():\n    pass\n'
has_immutables = '\nA_GOOD_PRIME: public(immutable(uint256))\n\n@external\ndef __init__():\n    A_GOOD_PRIME = 967\n'

def _parse_cbor_metadata(initcode):
    if False:
        return 10
    metadata_ofst = int.from_bytes(initcode[-2:], 'big')
    metadata = cbor2.loads(initcode[-metadata_ofst:-2])
    return metadata

def test_bytecode_runtime():
    if False:
        for i in range(10):
            print('nop')
    out = vyper.compile_code(simple_contract_code, output_formats=['bytecode_runtime', 'bytecode'])
    assert len(out['bytecode']) > len(out['bytecode_runtime'])
    assert out['bytecode_runtime'].removeprefix('0x') in out['bytecode'].removeprefix('0x')

def test_bytecode_signature():
    if False:
        return 10
    out = vyper.compile_code(simple_contract_code, output_formats=['bytecode_runtime', 'bytecode'])
    runtime_code = bytes.fromhex(out['bytecode_runtime'].removeprefix('0x'))
    initcode = bytes.fromhex(out['bytecode'].removeprefix('0x'))
    metadata = _parse_cbor_metadata(initcode)
    (runtime_len, data_section_lengths, immutables_len, compiler) = metadata
    assert runtime_len == len(runtime_code)
    assert data_section_lengths == []
    assert immutables_len == 0
    assert compiler == {'vyper': list(vyper.version.version_tuple)}

def test_bytecode_signature_dense_jumptable():
    if False:
        print('Hello World!')
    settings = Settings(optimize=OptimizationLevel.CODESIZE)
    out = vyper.compile_code(many_functions, output_formats=['bytecode_runtime', 'bytecode'], settings=settings)
    runtime_code = bytes.fromhex(out['bytecode_runtime'].removeprefix('0x'))
    initcode = bytes.fromhex(out['bytecode'].removeprefix('0x'))
    metadata = _parse_cbor_metadata(initcode)
    (runtime_len, data_section_lengths, immutables_len, compiler) = metadata
    assert runtime_len == len(runtime_code)
    assert data_section_lengths == [5, 35]
    assert immutables_len == 0
    assert compiler == {'vyper': list(vyper.version.version_tuple)}

def test_bytecode_signature_sparse_jumptable():
    if False:
        while True:
            i = 10
    settings = Settings(optimize=OptimizationLevel.GAS)
    out = vyper.compile_code(many_functions, output_formats=['bytecode_runtime', 'bytecode'], settings=settings)
    runtime_code = bytes.fromhex(out['bytecode_runtime'].removeprefix('0x'))
    initcode = bytes.fromhex(out['bytecode'].removeprefix('0x'))
    metadata = _parse_cbor_metadata(initcode)
    (runtime_len, data_section_lengths, immutables_len, compiler) = metadata
    assert runtime_len == len(runtime_code)
    assert data_section_lengths == [8]
    assert immutables_len == 0
    assert compiler == {'vyper': list(vyper.version.version_tuple)}

def test_bytecode_signature_immutables():
    if False:
        print('Hello World!')
    out = vyper.compile_code(has_immutables, output_formats=['bytecode_runtime', 'bytecode'])
    runtime_code = bytes.fromhex(out['bytecode_runtime'].removeprefix('0x'))
    initcode = bytes.fromhex(out['bytecode'].removeprefix('0x'))
    metadata = _parse_cbor_metadata(initcode)
    (runtime_len, data_section_lengths, immutables_len, compiler) = metadata
    assert runtime_len == len(runtime_code)
    assert data_section_lengths == []
    assert immutables_len == 32
    assert compiler == {'vyper': list(vyper.version.version_tuple)}

@pytest.mark.parametrize('code', [simple_contract_code, has_immutables, many_functions])
def test_bytecode_signature_deployed(code, get_contract, w3):
    if False:
        print('Hello World!')
    c = get_contract(code)
    deployed_code = w3.eth.get_code(c.address)
    initcode = c._classic_contract.bytecode
    metadata = _parse_cbor_metadata(initcode)
    (runtime_len, data_section_lengths, immutables_len, compiler) = metadata
    assert compiler == {'vyper': list(vyper.version.version_tuple)}
    assert len(deployed_code) == runtime_len + immutables_len