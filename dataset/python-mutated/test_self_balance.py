import pytest
from vyper import compiler
from vyper.compiler.settings import Settings
from vyper.evm.opcodes import EVM_VERSIONS

@pytest.mark.parametrize('evm_version', list(EVM_VERSIONS))
def test_self_balance(w3, get_contract_with_gas_estimation, evm_version):
    if False:
        print('Hello World!')
    code = '\n@external\n@view\ndef get_balance() -> uint256:\n    a: uint256 = self.balance\n    return a\n\n@external\n@payable\ndef __default__():\n    pass\n    '
    settings = Settings(evm_version=evm_version)
    opcodes = compiler.compile_code(code, output_formats=['opcodes'], settings=settings)['opcodes']
    if EVM_VERSIONS[evm_version] >= EVM_VERSIONS['istanbul']:
        assert 'SELFBALANCE' in opcodes
    else:
        assert 'SELFBALANCE' not in opcodes
    c = get_contract_with_gas_estimation(code, evm_version=evm_version)
    w3.eth.send_transaction({'to': c.address, 'value': 1337})
    assert c.get_balance() == 1337