import pytest
from web3.exceptions import ValidationError
ZERO_ADDRESS = '0x0000000000000000000000000000000000000000'
MAX_UINT256 = 2 ** 256 - 1
TOKEN_NAME = 'Vypercoin'
TOKEN_SYMBOL = 'FANG'
TOKEN_DECIMALS = 18
TOKEN_INITIAL_SUPPLY = 0

@pytest.fixture
def c(get_contract, w3):
    if False:
        print('Hello World!')
    with open('examples/tokens/ERC20.vy') as f:
        code = f.read()
    c = get_contract(code, *[TOKEN_NAME, TOKEN_SYMBOL, TOKEN_DECIMALS, TOKEN_INITIAL_SUPPLY])
    return c

@pytest.fixture
def c_bad(get_contract, w3):
    if False:
        i = 10
        return i + 15
    with open('examples/tokens/ERC20.vy') as f:
        code = f.read()
    bad_code = code.replace('self.totalSupply += _value', '').replace('self.totalSupply -= _value', '')
    c = get_contract(bad_code, *[TOKEN_NAME, TOKEN_SYMBOL, TOKEN_DECIMALS, TOKEN_INITIAL_SUPPLY])
    return c

@pytest.fixture
def get_log_args(get_logs):
    if False:
        return 10

    def get_log_args(tx_hash, c, event_name):
        if False:
            i = 10
            return i + 15
        logs = get_logs(tx_hash, c, event_name)
        assert len(logs) > 0
        args = logs[0].args
        return args
    return get_log_args

def test_initial_state(c, w3):
    if False:
        i = 10
        return i + 15
    (a1, a2, a3) = w3.eth.accounts[1:4]
    assert c.totalSupply() == TOKEN_INITIAL_SUPPLY
    assert c.name() == TOKEN_NAME
    assert c.symbol() == TOKEN_SYMBOL
    assert c.decimals() == TOKEN_DECIMALS
    assert c.balanceOf(a1) == 0
    assert c.balanceOf(a2) == 0
    assert c.balanceOf(a3) == 0
    assert c.allowance(a1, a1) == 0
    assert c.allowance(a1, a2) == 0
    assert c.allowance(a1, a3) == 0
    assert c.allowance(a2, a3) == 0

def test_mint_and_burn(c, w3, assert_tx_failed):
    if False:
        while True:
            i = 10
    (minter, a1, a2) = w3.eth.accounts[0:3]
    assert c.balanceOf(a1) == 0
    c.mint(a1, 2, transact={'from': minter})
    assert c.balanceOf(a1) == 2
    c.burn(2, transact={'from': a1})
    assert c.balanceOf(a1) == 0
    assert_tx_failed(lambda : c.burn(2, transact={'from': a1}))
    assert c.balanceOf(a1) == 0
    c.mint(a2, 0, transact={'from': minter})
    assert c.balanceOf(a2) == 0
    assert_tx_failed(lambda : c.burn(2, transact={'from': a2}))
    assert_tx_failed(lambda : c.burn(1, transact={'from': a1}))
    assert_tx_failed(lambda : c.mint(a1, 1, transact={'from': a1}))
    assert_tx_failed(lambda : c.mint(a2, 1, transact={'from': a2}))
    assert_tx_failed(lambda : c.mint(ZERO_ADDRESS, 1, transact={'from': a1}))
    assert_tx_failed(lambda : c.mint(ZERO_ADDRESS, 1, transact={'from': minter}))

def test_totalSupply(c, w3, assert_tx_failed):
    if False:
        i = 10
        return i + 15
    (minter, a1) = w3.eth.accounts[0:2]
    assert c.totalSupply() == 0
    c.mint(a1, 2, transact={'from': minter})
    assert c.totalSupply() == 2
    c.burn(1, transact={'from': a1})
    assert c.totalSupply() == 1
    c.burn(1, transact={'from': a1})
    assert c.totalSupply() == 0
    assert_tx_failed(lambda : c.burn(1, transact={'from': a1}))
    assert c.totalSupply() == 0
    c.mint(a1, 0, transact={'from': minter})
    assert c.totalSupply() == 0

def test_transfer(c, w3, assert_tx_failed):
    if False:
        print('Hello World!')
    (minter, a1, a2) = w3.eth.accounts[0:3]
    assert_tx_failed(lambda : c.burn(1, transact={'from': a2}))
    c.mint(a1, 2, transact={'from': minter})
    c.burn(1, transact={'from': a1})
    c.transfer(a2, 1, transact={'from': a1})
    assert_tx_failed(lambda : c.burn(1, transact={'from': a1}))
    c.burn(1, transact={'from': a2})
    assert_tx_failed(lambda : c.burn(1, transact={'from': a2}))
    assert_tx_failed(lambda : c.transfer(a1, 1, transact={'from': a2}))
    c.transfer(a1, 0, transact={'from': a2})

def test_maxInts(c, w3, assert_tx_failed):
    if False:
        while True:
            i = 10
    (minter, a1, a2) = w3.eth.accounts[0:3]
    c.mint(a1, MAX_UINT256, transact={'from': minter})
    assert c.balanceOf(a1) == MAX_UINT256
    assert_tx_failed(lambda : c.mint(a1, 1, transact={'from': a1}))
    assert_tx_failed(lambda : c.mint(a1, MAX_UINT256, transact={'from': a1}))
    assert_tx_failed(lambda : c.mint(a2, 1, transact={'from': minter}))
    c.burn(1, transact={'from': a1})
    c.mint(a2, 1, transact={'from': minter})
    assert_tx_failed(lambda : c.mint(a2, 1, transact={'from': minter}))
    c.transfer(a1, 1, transact={'from': a2})
    assert c.balanceOf(a1) == MAX_UINT256
    c.transfer(a2, MAX_UINT256, transact={'from': a1})
    assert c.balanceOf(a2) == MAX_UINT256
    assert c.balanceOf(a1) == 0
    with pytest.raises(ValidationError):
        c.transfer(a1, MAX_UINT256 + 1, transact={'from': a2})
    assert c.balanceOf(a2) == MAX_UINT256
    c.approve(a1, MAX_UINT256, transact={'from': a2})
    c.transferFrom(a2, a1, MAX_UINT256, transact={'from': a1})
    assert c.balanceOf(a1) == MAX_UINT256
    assert c.balanceOf(a2) == 0
    c.burn(MAX_UINT256, transact={'from': a1})
    assert c.balanceOf(a1) == 0

def test_transferFrom_and_Allowance(c, w3, assert_tx_failed):
    if False:
        i = 10
        return i + 15
    (minter, a1, a2, a3) = w3.eth.accounts[0:4]
    assert_tx_failed(lambda : c.burn(1, transact={'from': a2}))
    c.mint(a1, 1, transact={'from': minter})
    c.mint(a2, 1, transact={'from': minter})
    c.burn(1, transact={'from': a1})
    assert_tx_failed(lambda : c.transferFrom(a1, a3, 1, transact={'from': a2}))
    c.transferFrom(a1, a3, 0, transact={'from': a2})
    c.approve(a2, 1, transact={'from': a1})
    assert c.allowance(a1, a2) == 1
    assert c.allowance(a2, a1) == 0
    assert_tx_failed(lambda : c.transferFrom(a1, a3, 1, transact={'from': a3}))
    assert c.balanceOf(a2) == 1
    c.approve(a1, 1, transact={'from': a2})
    c.transferFrom(a2, a3, 1, transact={'from': a1})
    assert c.allowance(a2, a1) == 0
    c.approve(a1, 1, transact={'from': a2})
    assert c.allowance(a2, a1) == 1
    assert_tx_failed(lambda : c.transferFrom(a2, a3, 1, transact={'from': a1}))
    c.mint(a2, 1, transact={'from': minter})
    assert c.allowance(a2, a1) == 1
    c.approve(a1, 0, transact={'from': a2})
    assert c.allowance(a2, a1) == 0
    c.approve(a1, 0, transact={'from': a2})
    assert c.allowance(a2, a1) == 0
    assert_tx_failed(lambda : c.transferFrom(a2, a3, 1, transact={'from': a1}))
    assert c.allowance(a2, a1) == 0
    c.approve(a1, 1, transact={'from': a2})
    assert c.allowance(a2, a1) == 1
    c.approve(a1, 2, transact={'from': a2})
    assert c.allowance(a2, a1) == 2
    c.approve(a1, 0, transact={'from': a2})
    assert c.allowance(a2, a1) == 0
    c.approve(a1, 5, transact={'from': a2})
    assert c.allowance(a2, a1) == 5

def test_burnFrom_and_Allowance(c, w3, assert_tx_failed):
    if False:
        i = 10
        return i + 15
    (minter, a1, a2, a3) = w3.eth.accounts[0:4]
    assert_tx_failed(lambda : c.burn(1, transact={'from': a2}))
    c.mint(a1, 1, transact={'from': minter})
    c.mint(a2, 1, transact={'from': minter})
    c.burn(1, transact={'from': a1})
    assert_tx_failed(lambda : c.burnFrom(a1, 1, transact={'from': a2}))
    c.burnFrom(a1, 0, transact={'from': a2})
    c.approve(a2, 1, transact={'from': a1})
    assert c.allowance(a1, a2) == 1
    assert c.allowance(a2, a1) == 0
    assert_tx_failed(lambda : c.burnFrom(a2, 1, transact={'from': a3}))
    assert c.balanceOf(a2) == 1
    c.approve(a1, 1, transact={'from': a2})
    c.burnFrom(a2, 1, transact={'from': a1})
    assert c.allowance(a2, a1) == 0
    c.approve(a1, 1, transact={'from': a2})
    assert c.allowance(a2, a1) == 1
    assert_tx_failed(lambda : c.burnFrom(a2, 1, transact={'from': a1}))
    c.mint(a2, 1, transact={'from': minter})
    assert c.allowance(a2, a1) == 1
    c.approve(a1, 0, transact={'from': a2})
    assert c.allowance(a2, a1) == 0
    c.approve(a1, 0, transact={'from': a2})
    assert c.allowance(a2, a1) == 0
    assert_tx_failed(lambda : c.burnFrom(a2, 1, transact={'from': a1}))
    assert c.allowance(a2, a1) == 0
    c.approve(a1, 1, transact={'from': a2})
    assert c.allowance(a2, a1) == 1
    c.approve(a1, 2, transact={'from': a2})
    assert c.allowance(a2, a1) == 2
    c.approve(a1, 0, transact={'from': a2})
    assert c.allowance(a2, a1) == 0
    c.approve(a1, 5, transact={'from': a2})
    assert c.allowance(a2, a1) == 5
    assert_tx_failed(lambda : c.burnFrom(ZERO_ADDRESS, 0, transact={'from': a1}))

def test_raw_logs(c, w3, get_log_args):
    if False:
        i = 10
        return i + 15
    (minter, a1, a2, a3) = w3.eth.accounts[0:4]
    args = get_log_args(c.mint(a1, 2, transact={'from': minter}), c, 'Transfer')
    assert args.sender == ZERO_ADDRESS
    assert args.receiver == a1
    assert args.value == 2
    args = get_log_args(c.mint(a1, 0, transact={'from': minter}), c, 'Transfer')
    assert args.sender == ZERO_ADDRESS
    assert args.receiver == a1
    assert args.value == 0
    args = get_log_args(c.burn(1, transact={'from': a1}), c, 'Transfer')
    assert args.sender == a1
    assert args.receiver == ZERO_ADDRESS
    assert args.value == 1
    args = get_log_args(c.burn(0, transact={'from': a1}), c, 'Transfer')
    assert args.sender == a1
    assert args.receiver == ZERO_ADDRESS
    assert args.value == 0
    args = get_log_args(c.transfer(a2, 1, transact={'from': a1}), c, 'Transfer')
    assert args.sender == a1
    assert args.receiver == a2
    assert args.value == 1
    args = get_log_args(c.transfer(a2, 0, transact={'from': a1}), c, 'Transfer')
    assert args.sender == a1
    assert args.receiver == a2
    assert args.value == 0
    args = get_log_args(c.approve(a1, 1, transact={'from': a2}), c, 'Approval')
    assert args.owner == a2
    assert args.spender == a1
    assert args.value == 1
    args = get_log_args(c.approve(a2, 0, transact={'from': a3}), c, 'Approval')
    assert args.owner == a3
    assert args.spender == a2
    assert args.value == 0
    args = get_log_args(c.transferFrom(a2, a3, 1, transact={'from': a1}), c, 'Transfer')
    assert args.sender == a2
    assert args.receiver == a3
    assert args.value == 1
    args = get_log_args(c.transferFrom(a2, a3, 0, transact={'from': a1}), c, 'Transfer')
    assert args.sender == a2
    assert args.receiver == a3
    assert args.value == 0

def test_bad_transfer(c_bad, w3, assert_tx_failed):
    if False:
        while True:
            i = 10
    (minter, a1, a2) = w3.eth.accounts[0:3]
    c_bad.mint(a1, MAX_UINT256, transact={'from': minter})
    c_bad.mint(a2, 1, transact={'from': minter})
    assert_tx_failed(lambda : c_bad.transfer(a1, 1, transact={'from': a2}))
    c_bad.transfer(a2, MAX_UINT256 - 1, transact={'from': a1})
    assert c_bad.balanceOf(a1) == 1
    assert c_bad.balanceOf(a2) == MAX_UINT256

def test_bad_burn(c_bad, w3, assert_tx_failed):
    if False:
        while True:
            i = 10
    (minter, a1) = w3.eth.accounts[0:2]
    assert c_bad.balanceOf(a1) == 0
    c_bad.mint(a1, 2, transact={'from': minter})
    assert c_bad.balanceOf(a1) == 2
    assert_tx_failed(lambda : c_bad.burn(3, transact={'from': a1}))

def test_bad_transferFrom(c_bad, w3, assert_tx_failed):
    if False:
        for i in range(10):
            print('nop')
    (minter, a1, a2) = w3.eth.accounts[0:3]
    c_bad.mint(a1, MAX_UINT256, transact={'from': minter})
    c_bad.mint(a2, 1, transact={'from': minter})
    c_bad.approve(a1, 1, transact={'from': a2})
    assert_tx_failed(lambda : c_bad.transferFrom(a2, a1, 1, transact={'from': a1}))
    c_bad.approve(a2, MAX_UINT256 - 1, transact={'from': a1})
    assert c_bad.allowance(a1, a2) == MAX_UINT256 - 1
    c_bad.transferFrom(a1, a2, MAX_UINT256 - 1, transact={'from': a2})
    assert c_bad.balanceOf(a2) == MAX_UINT256