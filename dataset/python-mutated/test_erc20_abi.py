import pytest
from web3.exceptions import ValidationError
TOKEN_NAME = 'Vypercoin'
TOKEN_SYMBOL = 'FANG'
TOKEN_DECIMALS = 18
TOKEN_INITIAL_SUPPLY = 21 * 10 ** 6
TOKEN_TOTAL_SUPPLY = TOKEN_INITIAL_SUPPLY * 10 ** TOKEN_DECIMALS

@pytest.fixture
def erc20(get_contract):
    if False:
        i = 10
        return i + 15
    with open('examples/tokens/ERC20.vy') as f:
        contract = get_contract(f.read(), *[TOKEN_NAME, TOKEN_SYMBOL, TOKEN_DECIMALS, TOKEN_INITIAL_SUPPLY])
    return contract

@pytest.fixture
def erc20_caller(erc20, get_contract):
    if False:
        return 10
    erc20_caller_code = '\ninterface ERC20Contract:\n    def name() -> String[64]: view\n    def symbol() -> String[32]: view\n    def decimals() -> uint256: view\n    def balanceOf(_owner: address) -> uint256: view\n    def totalSupply() -> uint256: view\n    def transfer(_to: address, _amount: uint256) -> bool: nonpayable\n    def transferFrom(_from: address, _to: address, _value: uint256) -> bool: nonpayable\n    def approve(_spender: address, _amount: uint256) -> bool: nonpayable\n    def allowance(_owner: address, _spender: address) -> uint256: nonpayable\n\ntoken_address: ERC20Contract\n\n@external\ndef __init__(token_addr: address):\n    self.token_address = ERC20Contract(token_addr)\n\n@external\ndef name() -> String[64]:\n    return self.token_address.name()\n\n@external\ndef symbol() -> String[32]:\n    return self.token_address.symbol()\n\n@external\ndef decimals() -> uint256:\n    return self.token_address.decimals()\n\n@external\ndef balanceOf(_owner: address) -> uint256:\n    return self.token_address.balanceOf(_owner)\n\n@external\ndef totalSupply() -> uint256:\n    return self.token_address.totalSupply()\n\n@external\ndef transfer(_to: address, _value: uint256) -> bool:\n    return self.token_address.transfer(_to, _value)\n\n@external\ndef transferFrom(_from: address, _to: address, _value: uint256) -> bool:\n    return self.token_address.transferFrom(_from, _to, _value)\n\n@external\ndef allowance(_owner: address, _spender: address) -> uint256:\n    return self.token_address.allowance(_owner, _spender)\n    '
    return get_contract(erc20_caller_code, *[erc20.address])

def test_initial_state(w3, erc20_caller):
    if False:
        return 10
    assert erc20_caller.totalSupply() == TOKEN_TOTAL_SUPPLY
    assert erc20_caller.balanceOf(w3.eth.accounts[0]) == TOKEN_TOTAL_SUPPLY
    assert erc20_caller.balanceOf(w3.eth.accounts[1]) == 0
    assert erc20_caller.name() == TOKEN_NAME
    assert erc20_caller.symbol() == TOKEN_SYMBOL
    assert erc20_caller.decimals() == TOKEN_DECIMALS

def test_call_transfer(w3, erc20, erc20_caller, assert_tx_failed):
    if False:
        return 10
    erc20.transfer(erc20_caller.address, 10, transact={})
    assert erc20.balanceOf(erc20_caller.address) == 10
    erc20_caller.transfer(w3.eth.accounts[1], 10, transact={})
    assert erc20.balanceOf(erc20_caller.address) == 0
    assert erc20.balanceOf(w3.eth.accounts[1]) == 10
    assert_tx_failed(lambda : erc20_caller.transfer(w3.eth.accounts[1], TOKEN_TOTAL_SUPPLY))
    assert_tx_failed(function_to_test=lambda : erc20_caller.transfer(w3.eth.accounts[1], -1), exception=ValidationError)

def test_caller_approve_allowance(w3, erc20, erc20_caller):
    if False:
        for i in range(10):
            print('nop')
    assert erc20_caller.allowance(erc20.address, erc20_caller.address) == 0
    assert erc20.approve(erc20_caller.address, 10, transact={})
    assert erc20_caller.allowance(w3.eth.accounts[0], erc20_caller.address) == 10

def test_caller_tranfer_from(w3, erc20, erc20_caller, assert_tx_failed):
    if False:
        for i in range(10):
            print('nop')
    assert_tx_failed(lambda : erc20_caller.transferFrom(w3.eth.accounts[0], erc20_caller.address, 10))
    assert erc20.balanceOf(erc20_caller.address) == 0
    assert erc20.approve(erc20_caller.address, 10, transact={})
    erc20_caller.transferFrom(w3.eth.accounts[0], erc20_caller.address, 5, transact={})
    assert erc20.balanceOf(erc20_caller.address) == 5
    assert erc20_caller.allowance(w3.eth.accounts[0], erc20_caller.address) == 5
    erc20_caller.transferFrom(w3.eth.accounts[0], erc20_caller.address, 3, transact={})
    assert erc20.balanceOf(erc20_caller.address) == 8
    assert erc20_caller.allowance(w3.eth.accounts[0], erc20_caller.address) == 2