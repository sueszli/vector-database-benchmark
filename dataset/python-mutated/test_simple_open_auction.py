import pytest
EXPIRY = 16

@pytest.fixture
def auction_start(w3):
    if False:
        i = 10
        return i + 15
    return w3.eth.get_block('latest').timestamp + 1

@pytest.fixture
def auction_contract(w3, get_contract, auction_start):
    if False:
        while True:
            i = 10
    with open('examples/auctions/simple_open_auction.vy') as f:
        contract_code = f.read()
        contract = get_contract(contract_code, *[w3.eth.accounts[0], auction_start, EXPIRY])
    return contract

def test_initial_state(w3, tester, auction_contract, auction_start):
    if False:
        return 10
    assert auction_contract.beneficiary() == w3.eth.accounts[0]
    assert auction_contract.auctionStart() == auction_start
    assert auction_contract.auctionEnd() == auction_contract.auctionStart() + EXPIRY
    assert auction_contract.ended() is False
    assert auction_contract.highestBidder() is None
    assert auction_contract.highestBid() == 0
    assert auction_contract.auctionEnd() >= tester.get_block_by_number('latest')['timestamp']

def test_bid(w3, tester, auction_contract, assert_tx_failed):
    if False:
        while True:
            i = 10
    (k1, k2, k3, k4, k5) = w3.eth.accounts[:5]
    assert_tx_failed(lambda : auction_contract.bid(transact={'value': 0, 'from': k1}))
    auction_contract.bid(transact={'value': 1, 'from': k1})
    assert auction_contract.highestBidder() == k1
    assert auction_contract.highestBid() == 1
    assert_tx_failed(lambda : auction_contract.bid(transact={'value': 1, 'from': k1}))
    auction_contract.bid(transact={'value': 2, 'from': k2})
    assert auction_contract.highestBidder() == k2
    assert auction_contract.highestBid() == 2
    auction_contract.bid(transact={'value': 3, 'from': k3})
    auction_contract.bid(transact={'value': 4, 'from': k4})
    auction_contract.bid(transact={'value': 5, 'from': k5})
    assert auction_contract.highestBidder() == k5
    assert auction_contract.highestBid() == 5
    auction_contract.bid(transact={'value': 1 * 10 ** 10, 'from': k1})
    pending_return_before_outbid = auction_contract.pendingReturns(k1)
    auction_contract.bid(transact={'value': 2 * 10 ** 10, 'from': k2})
    pending_return_after_outbid = auction_contract.pendingReturns(k1)
    assert pending_return_after_outbid > pending_return_before_outbid
    balance_before_withdrawal = w3.eth.get_balance(k1)
    auction_contract.withdraw(transact={'from': k1})
    balance_after_withdrawal = w3.eth.get_balance(k1)
    assert balance_after_withdrawal > balance_before_withdrawal
    assert auction_contract.pendingReturns(k1) == 0

def test_end_auction(w3, tester, auction_contract, assert_tx_failed):
    if False:
        return 10
    (k1, k2, k3, k4, k5) = w3.eth.accounts[:5]
    assert_tx_failed(lambda : auction_contract.endAuction())
    auction_contract.bid(transact={'value': 1 * 10 ** 10, 'from': k2})
    w3.testing.mine(EXPIRY)
    balance_before_end = w3.eth.get_balance(k1)
    auction_contract.endAuction(transact={'from': k2})
    balance_after_end = w3.eth.get_balance(k1)
    assert balance_after_end == balance_before_end + 1 * 10 ** 10
    assert_tx_failed(lambda : auction_contract.bid(transact={'value': 10, 'from': k1}))
    assert_tx_failed(lambda : auction_contract.endAuction())