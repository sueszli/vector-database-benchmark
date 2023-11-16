import pytest

@pytest.fixture
def market_maker(get_contract):
    if False:
        i = 10
        return i + 15
    with open('examples/market_maker/on_chain_market_maker.vy') as f:
        contract_code = f.read()
    return get_contract(contract_code)
TOKEN_NAME = 'Vypercoin'
TOKEN_SYMBOL = 'FANG'
TOKEN_DECIMALS = 18
TOKEN_INITIAL_SUPPLY = 21 * 10 ** 6
TOKEN_TOTAL_SUPPLY = TOKEN_INITIAL_SUPPLY * 10 ** TOKEN_DECIMALS

@pytest.fixture
def erc20(get_contract):
    if False:
        return 10
    with open('examples/tokens/ERC20.vy') as f:
        contract_code = f.read()
    return get_contract(contract_code, *[TOKEN_NAME, TOKEN_SYMBOL, TOKEN_DECIMALS, TOKEN_INITIAL_SUPPLY])

def test_initial_state(market_maker):
    if False:
        while True:
            i = 10
    assert market_maker.totalEthQty() == 0
    assert market_maker.totalTokenQty() == 0
    assert market_maker.invariant() == 0
    assert market_maker.owner() is None

def test_initiate(w3, market_maker, erc20, assert_tx_failed):
    if False:
        while True:
            i = 10
    a0 = w3.eth.accounts[0]
    erc20.approve(market_maker.address, w3.to_wei(2, 'ether'), transact={})
    market_maker.initiate(erc20.address, w3.to_wei(1, 'ether'), transact={'value': w3.to_wei(2, 'ether')})
    assert market_maker.totalEthQty() == w3.to_wei(2, 'ether')
    assert market_maker.totalTokenQty() == w3.to_wei(1, 'ether')
    assert market_maker.invariant() == 2 * 10 ** 36
    assert market_maker.owner() == a0
    assert erc20.name() == TOKEN_NAME
    assert erc20.decimals() == TOKEN_DECIMALS
    assert_tx_failed(lambda : market_maker.initiate(erc20.address, w3.to_wei(1, 'ether'), transact={'value': w3.to_wei(2, 'ether')}))

def test_eth_to_tokens(w3, market_maker, erc20):
    if False:
        print('Hello World!')
    a1 = w3.eth.accounts[1]
    erc20.approve(market_maker.address, w3.to_wei(2, 'ether'), transact={})
    market_maker.initiate(erc20.address, w3.to_wei(1, 'ether'), transact={'value': w3.to_wei(2, 'ether')})
    assert erc20.balanceOf(market_maker.address) == w3.to_wei(1, 'ether')
    assert erc20.balanceOf(a1) == 0
    assert market_maker.totalTokenQty() == w3.to_wei(1, 'ether')
    assert market_maker.totalEthQty() == w3.to_wei(2, 'ether')
    market_maker.ethToTokens(transact={'value': 100, 'from': a1})
    assert erc20.balanceOf(market_maker.address) == 999999999999999950
    assert erc20.balanceOf(a1) == 50
    assert market_maker.totalTokenQty() == 999999999999999950
    assert market_maker.totalEthQty() == 2000000000000000100

def test_tokens_to_eth(w3, market_maker, erc20):
    if False:
        i = 10
        return i + 15
    a1 = w3.eth.accounts[1]
    a1_balance_before = w3.eth.get_balance(a1)
    erc20.transfer(a1, w3.to_wei(2, 'ether'), transact={})
    erc20.approve(market_maker.address, w3.to_wei(2, 'ether'), transact={'from': a1})
    market_maker.initiate(erc20.address, w3.to_wei(1, 'ether'), transact={'value': w3.to_wei(2, 'ether'), 'from': a1})
    assert w3.eth.get_balance(market_maker.address) == w3.to_wei(2, 'ether')
    assert w3.eth.get_balance(a1) == a1_balance_before - w3.to_wei(2, 'ether')
    assert market_maker.totalTokenQty() == w3.to_wei(1, 'ether')
    erc20.approve(market_maker.address, w3.to_wei(1, 'ether'), transact={'from': a1})
    market_maker.tokensToEth(w3.to_wei(1, 'ether'), transact={'from': a1})
    assert w3.eth.get_balance(market_maker.address) == w3.to_wei(1, 'ether')
    assert w3.eth.get_balance(a1) == a1_balance_before - w3.to_wei(1, 'ether')
    assert market_maker.totalTokenQty() == w3.to_wei(2, 'ether')
    assert market_maker.totalEthQty() == w3.to_wei(1, 'ether')

def test_owner_withdraw(w3, market_maker, erc20, assert_tx_failed):
    if False:
        for i in range(10):
            print('nop')
    (a0, a1) = w3.eth.accounts[:2]
    a0_balance_before = w3.eth.get_balance(a0)
    erc20.approve(market_maker.address, w3.to_wei(2, 'ether'), transact={})
    market_maker.initiate(erc20.address, w3.to_wei(1, 'ether'), transact={'value': w3.to_wei(2, 'ether')})
    assert w3.eth.get_balance(a0) == a0_balance_before - w3.to_wei(2, 'ether')
    assert erc20.balanceOf(a0) == TOKEN_TOTAL_SUPPLY - w3.to_wei(1, 'ether')
    assert_tx_failed(lambda : market_maker.ownerWithdraw(transact={'from': a1}))
    market_maker.ownerWithdraw(transact={})
    assert w3.eth.get_balance(a0) == a0_balance_before
    assert erc20.balanceOf(a0) == TOKEN_TOTAL_SUPPLY