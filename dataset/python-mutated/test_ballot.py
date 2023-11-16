import pytest
PROPOSAL_1_NAME = b'Clinton' + b'\x00' * 25
PROPOSAL_2_NAME = b'Trump' + b'\x00' * 27

@pytest.fixture
def c(get_contract):
    if False:
        i = 10
        return i + 15
    with open('examples/voting/ballot.vy') as f:
        contract_code = f.read()
    return get_contract(contract_code, *[[PROPOSAL_1_NAME, PROPOSAL_2_NAME]])
z0 = '0x0000000000000000000000000000000000000000'

def test_initial_state(w3, c):
    if False:
        i = 10
        return i + 15
    a0 = w3.eth.accounts[0]
    assert c.chairperson() == a0
    assert c.proposals(0)[0][:7] == b'Clinton'
    assert c.proposals(1)[0][:5] == b'Trump'
    assert c.proposals(0)[1] == 0
    assert c.proposals(1)[1] == 0
    assert c.voterCount() == 0
    assert c.voters(z0)[2] is None
    assert c.voters(z0)[3] == 0
    assert c.voters(z0)[1] is False
    assert c.voters(z0)[0] == 0

def test_give_the_right_to_vote(w3, c, assert_tx_failed):
    if False:
        for i in range(10):
            print('nop')
    (a0, a1, a2, a3, a4, a5) = w3.eth.accounts[:6]
    c.giveRightToVote(a1, transact={})
    assert c.voters(a1)[0] == 1
    assert c.voters(a1)[2] is None
    assert c.voters(a1)[3] == 0
    assert c.voters(a1)[1] is False
    c.giveRightToVote(a0, transact={})
    assert c.voters(a0)[0] == 1
    assert c.voterCount() == 2
    c.giveRightToVote(a2, transact={})
    c.giveRightToVote(a3, transact={})
    c.giveRightToVote(a4, transact={})
    c.giveRightToVote(a5, transact={})
    assert c.voterCount() == 6
    assert_tx_failed(lambda : c.giveRightToVote(a5, transact={}))
    assert c.voters(a5)[0] == 1

def test_forward_weight(w3, c):
    if False:
        print('Hello World!')
    (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9) = w3.eth.accounts[:10]
    c.giveRightToVote(a0, transact={})
    c.giveRightToVote(a1, transact={})
    c.giveRightToVote(a2, transact={})
    c.giveRightToVote(a3, transact={})
    c.giveRightToVote(a4, transact={})
    c.giveRightToVote(a5, transact={})
    c.giveRightToVote(a6, transact={})
    c.giveRightToVote(a7, transact={})
    c.giveRightToVote(a8, transact={})
    c.giveRightToVote(a9, transact={})
    c.delegate(a2, transact={'from': a1})
    c.delegate(a3, transact={'from': a2})
    assert c.voters(a1)[0] == 0
    assert c.voters(a2)[0] == 0
    assert c.voters(a3)[0] == 3
    c.delegate(a9, transact={'from': a8})
    c.delegate(a8, transact={'from': a7})
    assert c.voters(a7)[0] == 0
    assert c.voters(a8)[0] == 0
    assert c.voters(a9)[0] == 3
    c.delegate(a7, transact={'from': a6})
    c.delegate(a6, transact={'from': a5})
    c.delegate(a5, transact={'from': a4})
    assert c.voters(a9)[0] == 6
    assert c.voters(a8)[0] == 0
    c.delegate(a4, transact={'from': a3})
    assert c.voters(a8)[0] == 3
    assert c.voters(a9)[0] == 6
    c.forwardWeight(a8, transact={})
    assert c.voters(a8)[0] == 0
    assert c.voters(a9)[0] == 9
    c.delegate(a1, transact={'from': a0})
    assert c.voters(a5)[0] == 1
    assert c.voters(a9)[0] == 9
    c.forwardWeight(a5, transact={})
    assert c.voters(a5)[0] == 0
    assert c.voters(a9)[0] == 10

def test_block_short_cycle(w3, c, assert_tx_failed):
    if False:
        i = 10
        return i + 15
    (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9) = w3.eth.accounts[:10]
    c.giveRightToVote(a0, transact={})
    c.giveRightToVote(a1, transact={})
    c.giveRightToVote(a2, transact={})
    c.giveRightToVote(a3, transact={})
    c.giveRightToVote(a4, transact={})
    c.giveRightToVote(a5, transact={})
    c.delegate(a1, transact={'from': a0})
    c.delegate(a2, transact={'from': a1})
    c.delegate(a3, transact={'from': a2})
    c.delegate(a4, transact={'from': a3})
    assert_tx_failed(lambda : c.delegate(a0, transact={'from': a4}))
    c.delegate(a5, transact={'from': a4})
    c.delegate(a0, transact={'from': a5})

def test_delegate(w3, c, assert_tx_failed):
    if False:
        i = 10
        return i + 15
    (a0, a1, a2, a3, a4, a5, a6) = w3.eth.accounts[:7]
    c.giveRightToVote(a0, transact={})
    c.giveRightToVote(a1, transact={})
    c.giveRightToVote(a2, transact={})
    c.giveRightToVote(a3, transact={})
    assert c.voters(a1)[0] == 1
    c.delegate(a0, transact={'from': a1})
    assert c.voters(a1)[0] == 0
    assert c.voters(a1)[1] is True
    assert c.voters(a0)[0] == 2
    assert_tx_failed(lambda : c.delegate(a2, transact={'from': a1}))
    assert_tx_failed(lambda : c.delegate(a2, transact={'from': a2}))
    c.delegate(a6, transact={'from': a2})
    c.delegate(a1, transact={'from': a3})
    assert c.voters(a0)[0] == 3

def test_vote(w3, c, assert_tx_failed):
    if False:
        i = 10
        return i + 15
    (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9) = w3.eth.accounts[:10]
    c.giveRightToVote(a0, transact={})
    c.giveRightToVote(a1, transact={})
    c.giveRightToVote(a2, transact={})
    c.giveRightToVote(a3, transact={})
    c.giveRightToVote(a4, transact={})
    c.giveRightToVote(a5, transact={})
    c.giveRightToVote(a6, transact={})
    c.giveRightToVote(a7, transact={})
    c.delegate(a0, transact={'from': a1})
    c.delegate(a1, transact={'from': a3})
    c.vote(0, transact={})
    assert c.proposals(0)[1] == 3
    assert_tx_failed(lambda : c.vote(0))
    assert_tx_failed(lambda : c.vote(0, transact={'from': a1}))
    c.vote(1, transact={'from': a4})
    c.vote(1, transact={'from': a2})
    c.vote(1, transact={'from': a5})
    c.vote(1, transact={'from': a6})
    assert c.proposals(1)[1] == 4
    assert_tx_failed(lambda : c.vote(2, transact={'from': a7}))

def test_winning_proposal(w3, c):
    if False:
        i = 10
        return i + 15
    (a0, a1, a2) = w3.eth.accounts[:3]
    c.giveRightToVote(a0, transact={})
    c.giveRightToVote(a1, transact={})
    c.giveRightToVote(a2, transact={})
    c.vote(0, transact={})
    assert c.winningProposal() == 0
    c.vote(1, transact={'from': a1})
    assert c.winningProposal() == 0
    c.vote(1, transact={'from': a2})
    assert c.winningProposal() == 1

def test_winner_namer(w3, c):
    if False:
        while True:
            i = 10
    (a0, a1, a2) = w3.eth.accounts[:3]
    c.giveRightToVote(a0, transact={})
    c.giveRightToVote(a1, transact={})
    c.giveRightToVote(a2, transact={})
    c.delegate(a1, transact={'from': a2})
    c.vote(0, transact={})
    assert c.winnerName()[:7] == b'Clinton'
    c.vote(1, transact={'from': a1})
    assert c.winnerName()[:5] == b'Trump'