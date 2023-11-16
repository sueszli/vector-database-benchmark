import pytest
from vyper.compiler import compile_code
from vyper.compiler.settings import Settings
from vyper.evm.opcodes import EVM_VERSIONS
from vyper.exceptions import StructureException
post_cancun = {k: v for (k, v) in EVM_VERSIONS.items() if v >= EVM_VERSIONS['cancun']}

@pytest.mark.parametrize('evm_version', list(EVM_VERSIONS.keys()))
def test_transient_blocked(evm_version):
    if False:
        for i in range(10):
            print('nop')
    code = '\nmy_map: transient(HashMap[address, uint256])\n    '
    settings = Settings(evm_version=evm_version)
    if EVM_VERSIONS[evm_version] >= EVM_VERSIONS['cancun']:
        assert compile_code(code, settings=settings) is not None
    else:
        with pytest.raises(StructureException):
            compile_code(code, settings=settings)

@pytest.mark.parametrize('evm_version', list(post_cancun.keys()))
def test_transient_compiles(evm_version):
    if False:
        i = 10
        return i + 15
    settings = Settings(evm_version=evm_version)
    getter_code = '\nmy_map: public(transient(HashMap[address, uint256]))\n    '
    t = compile_code(getter_code, settings=settings, output_formats=['opcodes_runtime'])
    t = t['opcodes_runtime'].split(' ')
    assert 'TLOAD' in t
    assert 'TSTORE' not in t
    setter_code = '\nmy_map: transient(HashMap[address, uint256])\n\n@external\ndef setter(k: address, v: uint256):\n    self.my_map[k] = v\n    '
    t = compile_code(setter_code, settings=settings, output_formats=['opcodes_runtime'])
    t = t['opcodes_runtime'].split(' ')
    assert 'TLOAD' not in t
    assert 'TSTORE' in t
    getter_setter_code = '\nmy_map: public(transient(HashMap[address, uint256]))\n\n@external\ndef setter(k: address, v: uint256):\n    self.my_map[k] = v\n    '
    t = compile_code(getter_setter_code, settings=settings, output_formats=['opcodes_runtime'])
    t = t['opcodes_runtime'].split(' ')
    assert 'TLOAD' in t
    assert 'TSTORE' in t