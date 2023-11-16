import pytest
AMOUNT = 100 * 10 ** 18
TOKEN_NAME = 'Vypercoin'
TOKEN_SYMBOL = 'FANG'
TOKEN_DECIMALS = 18
TOKEN_INITIAL_SUPPLY = 0

@pytest.fixture
def token(get_contract):
    if False:
        print('Hello World!')
    with open('examples/tokens/ERC20.vy') as f:
        return get_contract(f.read(), TOKEN_NAME, TOKEN_SYMBOL, TOKEN_DECIMALS, TOKEN_INITIAL_SUPPLY)

@pytest.fixture
def vault(get_contract, token):
    if False:
        for i in range(10):
            print('nop')
    with open('examples/tokens/ERC4626.vy') as f:
        return get_contract(f.read(), token.address)

def test_asset(vault, token):
    if False:
        return 10
    assert vault.asset() == token.address

def test_max_methods(w3, vault):
    if False:
        for i in range(10):
            print('nop')
    a = w3.eth.accounts[0]
    assert vault.maxDeposit(a) == 2 ** 256 - 1
    assert vault.maxMint(a) == 2 ** 256 - 1
    assert vault.maxWithdraw(a) == 2 ** 256 - 1
    assert vault.maxRedeem(a) == 2 ** 256 - 1

def test_preview_methods(w3, token, vault):
    if False:
        return 10
    a = w3.eth.accounts[0]
    assert vault.totalAssets() == 0
    assert vault.convertToAssets(10 ** 18) == 0
    assert vault.convertToShares(10 ** 18) == 10 ** 18
    assert vault.previewDeposit(AMOUNT) == AMOUNT
    assert vault.previewMint(AMOUNT) == AMOUNT
    assert vault.previewWithdraw(AMOUNT) == 0
    assert vault.previewRedeem(AMOUNT) == 0
    token.mint(a, AMOUNT, transact={'from': a})
    token.approve(vault.address, AMOUNT, transact={'from': a})
    vault.deposit(AMOUNT, transact={'from': a})
    assert vault.totalAssets() == AMOUNT
    assert vault.convertToAssets(10 ** 18) == 10 ** 18
    assert vault.convertToShares(10 ** 18) == 10 ** 18
    assert vault.previewDeposit(AMOUNT) == AMOUNT
    assert vault.previewMint(AMOUNT) == AMOUNT
    assert vault.previewWithdraw(AMOUNT) == AMOUNT
    assert vault.previewRedeem(AMOUNT) == AMOUNT
    token.mint(vault.address, AMOUNT, transact={'from': a})
    assert vault.totalAssets() == 2 * AMOUNT
    assert vault.convertToAssets(10 ** 18) == 2 * 10 ** 18
    assert vault.convertToShares(2 * 10 ** 18) == 10 ** 18
    assert vault.previewDeposit(AMOUNT) == AMOUNT // 2
    assert vault.previewMint(AMOUNT // 2) == AMOUNT
    assert vault.previewWithdraw(AMOUNT) == AMOUNT // 2
    assert vault.previewRedeem(AMOUNT // 2) == AMOUNT
    vault.DEBUG_steal_tokens(AMOUNT, transact={'from': a})
    assert vault.totalAssets() == AMOUNT
    assert vault.convertToAssets(10 ** 18) == 10 ** 18
    assert vault.convertToShares(10 ** 18) == 10 ** 18
    assert vault.previewDeposit(AMOUNT) == AMOUNT
    assert vault.previewMint(AMOUNT) == AMOUNT
    assert vault.previewWithdraw(AMOUNT) == AMOUNT
    assert vault.previewRedeem(AMOUNT) == AMOUNT
    vault.DEBUG_steal_tokens(AMOUNT // 2, transact={'from': a})
    assert vault.totalAssets() == AMOUNT // 2
    assert vault.convertToAssets(10 ** 18) == 10 ** 18 // 2
    assert vault.convertToShares(10 ** 18 // 2) == 10 ** 18
    assert vault.previewDeposit(AMOUNT) == 2 * AMOUNT
    assert vault.previewMint(2 * AMOUNT) == AMOUNT
    assert vault.previewWithdraw(AMOUNT) == 2 * AMOUNT
    assert vault.previewRedeem(2 * AMOUNT) == AMOUNT