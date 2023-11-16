import pytest
from vyper.exceptions import FunctionDeclarationException

def test_nonreentrant_decorator(get_contract, assert_tx_failed):
    if False:
        for i in range(10):
            print('nop')
    calling_contract_code = "\ninterface SpecialContract:\n    def unprotected_function(val: String[100], do_callback: bool): nonpayable\n    def protected_function(val: String[100], do_callback: bool): nonpayable\n    def special_value() -> String[100]: nonpayable\n\n@external\ndef updated():\n    SpecialContract(msg.sender).unprotected_function('surprise!', False)\n\n@external\ndef updated_protected():\n    # This should fail.\n    SpecialContract(msg.sender).protected_function('surprise protected!', False)\n    "
    reentrant_code = "\ninterface Callback:\n    def updated(): nonpayable\n    def updated_protected(): nonpayable\ninterface Self:\n    def protected_function(val: String[100], do_callback: bool) -> uint256: nonpayable\n    def protected_function2(val: String[100], do_callback: bool) -> uint256: nonpayable\n    def protected_view_fn() -> String[100]: view\n\nspecial_value: public(String[100])\ncallback: public(Callback)\n\n@external\ndef set_callback(c: address):\n    self.callback = Callback(c)\n\n@external\n@nonreentrant('protect_special_value')\ndef protected_function(val: String[100], do_callback: bool) -> uint256:\n    self.special_value = val\n\n    if do_callback:\n        self.callback.updated_protected()\n        return 1\n    else:\n        return 2\n\n@external\n@nonreentrant('protect_special_value')\ndef protected_function2(val: String[100], do_callback: bool) -> uint256:\n    self.special_value = val\n    if do_callback:\n        # call other function with same nonreentrancy key\n        Self(self).protected_function(val, False)\n        return 1\n    return 2\n\n@external\n@nonreentrant('protect_special_value')\ndef protected_function3(val: String[100], do_callback: bool) -> uint256:\n    self.special_value = val\n    if do_callback:\n        # call other function with same nonreentrancy key\n        assert self.special_value == Self(self).protected_view_fn()\n        return 1\n    return 2\n\n\n@external\n@nonreentrant('protect_special_value')\ndef protected_view_fn() -> String[100]:\n    return self.special_value\n\n@external\ndef unprotected_function(val: String[100], do_callback: bool):\n    self.special_value = val\n\n    if do_callback:\n        self.callback.updated()\n    "
    reentrant_contract = get_contract(reentrant_code)
    calling_contract = get_contract(calling_contract_code)
    reentrant_contract.set_callback(calling_contract.address, transact={})
    assert reentrant_contract.callback() == calling_contract.address
    reentrant_contract.unprotected_function('some value', True, transact={})
    assert reentrant_contract.special_value() == 'surprise!'
    reentrant_contract.protected_function('some value', False, transact={})
    assert reentrant_contract.special_value() == 'some value'
    assert reentrant_contract.protected_view_fn() == 'some value'
    assert_tx_failed(lambda : reentrant_contract.protected_function('zzz value', True, transact={}))
    reentrant_contract.protected_function2('another value', False, transact={})
    assert reentrant_contract.special_value() == 'another value'
    assert_tx_failed(lambda : reentrant_contract.protected_function2('zzz value', True, transact={}))
    reentrant_contract.protected_function3('another value', False, transact={})
    assert reentrant_contract.special_value() == 'another value'
    assert_tx_failed(lambda : reentrant_contract.protected_function3('zzz value', True, transact={}))

def test_nonreentrant_decorator_for_default(w3, get_contract, assert_tx_failed):
    if False:
        print('Hello World!')
    calling_contract_code = '\n@external\ndef send_funds(_amount: uint256):\n    # raw_call() is used to overcome gas limit of send()\n    response: Bytes[32] = raw_call(\n        msg.sender,\n        _abi_encode(msg.sender, _amount, method_id=method_id("transfer(address,uint256)")),\n        max_outsize=32,\n        value=_amount\n    )\n\n@external\n@payable\ndef __default__():\n    pass\n    '
    reentrant_code = '\ninterface Callback:\n    def send_funds(_amount: uint256): nonpayable\n\nspecial_value: public(String[100])\ncallback: public(Callback)\n\n@external\ndef set_callback(c: address):\n    self.callback = Callback(c)\n\n@external\n@payable\n@nonreentrant("lock")\ndef protected_function(val: String[100], do_callback: bool) -> uint256:\n    self.special_value = val\n    _amount: uint256 = msg.value\n    send(self.callback.address, msg.value)\n\n    if do_callback:\n        self.callback.send_funds(_amount)\n        return 1\n    else:\n        return 2\n\n@external\n@payable\ndef unprotected_function(val: String[100], do_callback: bool):\n    self.special_value = val\n    _amount: uint256 = msg.value\n    send(self.callback.address, msg.value)\n\n    if do_callback:\n        self.callback.send_funds(_amount)\n\n@external\n@payable\n@nonreentrant("lock")\ndef __default__():\n    pass\n    '
    reentrant_contract = get_contract(reentrant_code)
    calling_contract = get_contract(calling_contract_code)
    reentrant_contract.set_callback(calling_contract.address, transact={})
    assert reentrant_contract.callback() == calling_contract.address
    reentrant_contract.unprotected_function('some value', False, transact={'value': 1000})
    assert reentrant_contract.special_value() == 'some value'
    assert w3.eth.get_balance(reentrant_contract.address) == 0
    assert w3.eth.get_balance(calling_contract.address) == 1000
    reentrant_contract.unprotected_function('another value', True, transact={'value': 1000})
    assert reentrant_contract.special_value() == 'another value'
    assert w3.eth.get_balance(reentrant_contract.address) == 1000
    assert w3.eth.get_balance(calling_contract.address) == 1000
    reentrant_contract.protected_function('surprise!', False, transact={'value': 1000})
    assert reentrant_contract.special_value() == 'surprise!'
    assert w3.eth.get_balance(reentrant_contract.address) == 1000
    assert w3.eth.get_balance(calling_contract.address) == 2000
    assert_tx_failed(lambda : reentrant_contract.protected_function('zzz value', True, transact={'value': 1000}))

def test_disallow_on_init_function(get_contract):
    if False:
        for i in range(10):
            print('nop')
    code = '\n\n@external\n@nonreentrant("lock")\ndef __init__():\n    foo: uint256 = 0\n'
    with pytest.raises(FunctionDeclarationException):
        get_contract(code)