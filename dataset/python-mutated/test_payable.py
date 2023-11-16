import pytest
from vyper.exceptions import CallViolation

@pytest.mark.parametrize('source', ['\ninterface PiggyBank:\n    def deposit(): nonpayable\n\npiggy: PiggyBank\n\n@external\ndef foo():\n    self.piggy.deposit()\n    ', '\ninterface PiggyBank:\n    def deposit(): payable\n\npiggy: PiggyBank\n\n@external\ndef foo():\n    self.piggy.deposit()\n    '])
def test_payable_call_compiles(source, get_contract):
    if False:
        for i in range(10):
            print('nop')
    get_contract(source)

@pytest.mark.parametrize('source', ['\ninterface PiggyBank:\n    def deposit(): nonpayable\n\npiggy: PiggyBank\n\n@external\ndef foo():\n    self.piggy.deposit(value=self.balance)\n    '])
def test_payable_compile_fail(source, get_contract, assert_compile_failed):
    if False:
        while True:
            i = 10
    assert_compile_failed(lambda : get_contract(source), CallViolation)
nonpayable_code = ['\n# single function, nonpayable\n@external\ndef foo() -> bool:\n    return True\n    ', '\n# multiple functions, one is payable\n@external\ndef foo() -> bool:\n    return True\n\n@payable\n@external\ndef bar() -> bool:\n    return True\n    ', '\n# multiple functions, nonpayable\n@external\ndef foo() -> bool:\n    return True\n\n@external\ndef bar() -> bool:\n    return True\n    ', '\n# multiple functions and default func, nonpayable\n@external\ndef foo() -> bool:\n    return True\n\n@external\ndef bar() -> bool:\n    return True\n\n@external\ndef __default__():\n    pass\n    ', '\n    # multiple functions and default func, payable\n@external\ndef foo() -> bool:\n    return True\n\n@external\ndef bar() -> bool:\n    return True\n\n@external\n@payable\ndef __default__():\n    pass\n    ', '\n# multiple functions, nonpayable (view)\n@external\ndef foo() -> bool:\n    return True\n\n@view\n@external\ndef bar() -> bool:\n    return True\n    ', '\n# payable init function\n@external\n@payable\ndef __init__():\n    a: int128 = 1\n\n@external\ndef foo() -> bool:\n    return True\n    ', '\n# payable default function\n@external\n@payable\ndef __default__():\n    a: int128 = 1\n\n@external\ndef foo() -> bool:\n    return True\n    ', '\n# payable default function and other function\n@external\n@payable\ndef __default__():\n    a: int128 = 1\n\n@external\ndef foo() -> bool:\n    return True\n\n@external\n@payable\ndef bar() -> bool:\n    return True\n    ', '\n# several functions, one payable\n@external\ndef foo() -> bool:\n    return True\n\n@payable\n@external\ndef bar() -> bool:\n    return True\n\n@external\ndef baz() -> bool:\n    return True\n    ']

@pytest.mark.parametrize('code', nonpayable_code)
def test_nonpayable_runtime_assertion(w3, keccak, assert_tx_failed, get_contract, code):
    if False:
        for i in range(10):
            print('nop')
    c = get_contract(code)
    c.foo(transact={'value': 0})
    sig = keccak('foo()'.encode()).hex()[:10]
    assert_tx_failed(lambda : w3.eth.send_transaction({'to': c.address, 'data': sig, 'value': 10 ** 18}))
payable_code = ['\n# single function, payable\n@payable\n@external\ndef foo() -> bool:\n    return True\n    ', '\n# two functions, one is payable\n@payable\n@external\ndef foo() -> bool:\n    return True\n\n@external\ndef bar() -> bool:\n    return True\n    ', '\n# two functions, payable\n@payable\n@external\ndef foo() -> bool:\n    return True\n\n@payable\n@external\ndef bar() -> bool:\n    return True\n    ', '\n# two functions, one nonpayable (view)\n@payable\n@external\ndef foo() -> bool:\n    return True\n\n@view\n@external\ndef bar() -> bool:\n    return True\n    ', '\n# several functions, all payable\n@payable\n@external\ndef foo() -> bool:\n    return True\n\n@payable\n@external\ndef bar() -> bool:\n    return True\n\n@payable\n@external\ndef baz() -> bool:\n    return True\n    ', '\n# several functions, one payable\n@payable\n@external\ndef foo() -> bool:\n    return True\n\n@external\ndef bar() -> bool:\n    return True\n\n@external\ndef baz() -> bool:\n    return True\n    ', '\n# several functions, two payable\n@payable\n@external\ndef foo() -> bool:\n    return True\n\n@external\ndef bar() -> bool:\n    return True\n\n@payable\n@external\ndef baz() -> bool:\n    return True\n    ', '\n# init function\n@external\ndef __init__():\n    a: int128 = 1\n\n@payable\n@external\ndef foo() -> bool:\n    return True\n    ', '\n# default function\n@external\ndef __default__():\n    a: int128 = 1\n\n@external\n@payable\ndef foo() -> bool:\n    return True\n    ', '\n# payable default function\n@external\n@payable\ndef __default__():\n    a: int128 = 1\n\n@external\n@payable\ndef foo() -> bool:\n    return True\n    ', '\n# payable default function and nonpayable other function\n@external\n@payable\ndef __default__():\n    a: int128 = 1\n\n@external\n@payable\ndef foo() -> bool:\n    return True\n\n@external\ndef bar() -> bool:\n    return True\n    ']

@pytest.mark.parametrize('code', payable_code)
def test_payable_runtime_assertion(get_contract, code):
    if False:
        return 10
    c = get_contract(code)
    c.foo(transact={'value': 10 ** 18})
    c.foo(transact={'value': 0})

def test_payable_default_func_invalid_calldata(get_contract, w3):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo() -> bool:\n    return True\n\n@payable\n@external\ndef __default__():\n    pass\n    '
    c = get_contract(code)
    (w3.eth.send_transaction({'to': c.address, 'value': 100, 'data': '0x12345678'}),)

def test_nonpayable_default_func_invalid_calldata(get_contract, w3, assert_tx_failed):
    if False:
        i = 10
        return i + 15
    code = '\n@external\n@payable\ndef foo() -> bool:\n    return True\n\n@external\ndef __default__():\n    pass\n    '
    c = get_contract(code)
    w3.eth.send_transaction({'to': c.address, 'value': 0, 'data': '0x12345678'})
    assert_tx_failed(lambda : w3.eth.send_transaction({'to': c.address, 'value': 100, 'data': '0x12345678'}))

def test_batch_nonpayable(get_contract, w3, assert_tx_failed):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\ndef foo() -> bool:\n    return True\n\n@external\ndef __default__():\n    pass\n    '
    c = get_contract(code)
    w3.eth.send_transaction({'to': c.address, 'value': 0, 'data': '0x12345678'})
    data = bytes([1, 2, 3, 4])
    for i in range(5):
        calldata = '0x' + data[:i].hex()
        assert_tx_failed(lambda : w3.eth.send_transaction({'to': c.address, 'value': 100, 'data': calldata}))