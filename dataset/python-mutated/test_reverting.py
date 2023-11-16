import pytest
from eth.codecs import abi
from eth_tester.exceptions import TransactionFailed
from vyper.utils import method_id
pytestmark = pytest.mark.usefixtures('memory_mocker')

def test_revert_reason(w3, assert_tx_failed, get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    reverty_code = '\n@external\ndef foo():\n    data: Bytes[4] = method_id("NoFives()")\n    raw_revert(data)\n    '
    revert_bytes = method_id('NoFives()')
    assert_tx_failed(lambda : get_contract_with_gas_estimation(reverty_code).foo(transact={}), TransactionFailed, exc_text=f'execution reverted: {revert_bytes}')

def test_revert_reason_typed(w3, assert_tx_failed, get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    reverty_code = '\n@external\ndef foo():\n    val: uint256 = 5\n    data: Bytes[100] = _abi_encode(val, method_id=method_id("NoFives(uint256)"))\n    raw_revert(data)\n    '
    revert_bytes = method_id('NoFives(uint256)') + abi.encode('(uint256)', (5,))
    assert_tx_failed(lambda : get_contract_with_gas_estimation(reverty_code).foo(transact={}), TransactionFailed, exc_text=f'execution reverted: {revert_bytes}')

def test_revert_reason_typed_no_variable(w3, assert_tx_failed, get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    reverty_code = '\n@external\ndef foo():\n    val: uint256 = 5\n    raw_revert(_abi_encode(val, method_id=method_id("NoFives(uint256)")))\n    '
    revert_bytes = method_id('NoFives(uint256)') + abi.encode('(uint256)', (5,))
    assert_tx_failed(lambda : get_contract_with_gas_estimation(reverty_code).foo(transact={}), TransactionFailed, exc_text=f'execution reverted: {revert_bytes}')