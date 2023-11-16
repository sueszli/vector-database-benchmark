import pytest
SOMEONE_TOKEN_IDS = [1, 2, 3]
OPERATOR_TOKEN_ID = 10
NEW_TOKEN_ID = 20
INVALID_TOKEN_ID = 99
ZERO_ADDRESS = '0x0000000000000000000000000000000000000000'
ERC165_SIG = '0x01ffc9a7'
ERC165_INVALID_SIG = '0xffffffff'
ERC721_SIG = '0x80ac58cd'

@pytest.fixture
def c(get_contract, w3):
    if False:
        print('Hello World!')
    with open('examples/tokens/ERC721.vy') as f:
        code = f.read()
    c = get_contract(code)
    (minter, someone, operator) = w3.eth.accounts[:3]
    for i in SOMEONE_TOKEN_IDS:
        c.mint(someone, i, transact={'from': minter})
    c.mint(operator, OPERATOR_TOKEN_ID, transact={'from': minter})
    return c

def test_erc165(w3, c):
    if False:
        for i in range(10):
            print('nop')
    assert c.supportsInterface(ERC165_SIG)
    assert not c.supportsInterface(ERC165_INVALID_SIG)
    assert c.supportsInterface(ERC721_SIG)

def test_balanceOf(c, w3, assert_tx_failed):
    if False:
        print('Hello World!')
    someone = w3.eth.accounts[1]
    assert c.balanceOf(someone) == 3
    assert_tx_failed(lambda : c.balanceOf(ZERO_ADDRESS))

def test_ownerOf(c, w3, assert_tx_failed):
    if False:
        while True:
            i = 10
    someone = w3.eth.accounts[1]
    assert c.ownerOf(SOMEONE_TOKEN_IDS[0]) == someone
    assert_tx_failed(lambda : c.ownerOf(INVALID_TOKEN_ID))

def test_getApproved(c, w3):
    if False:
        return 10
    (someone, operator) = w3.eth.accounts[1:3]
    assert c.getApproved(SOMEONE_TOKEN_IDS[0]) is None
    c.approve(operator, SOMEONE_TOKEN_IDS[0], transact={'from': someone})
    assert c.getApproved(SOMEONE_TOKEN_IDS[0]) == operator

def test_isApprovedForAll(c, w3):
    if False:
        i = 10
        return i + 15
    (someone, operator) = w3.eth.accounts[1:3]
    assert c.isApprovedForAll(someone, operator) == 0
    c.setApprovalForAll(operator, True, transact={'from': someone})
    assert c.isApprovedForAll(someone, operator) == 1

def test_transferFrom_by_owner(c, w3, assert_tx_failed, get_logs):
    if False:
        i = 10
        return i + 15
    (someone, operator) = w3.eth.accounts[1:3]
    assert_tx_failed(lambda : c.transferFrom(ZERO_ADDRESS, operator, SOMEONE_TOKEN_IDS[0], transact={'from': someone}))
    assert_tx_failed(lambda : c.transferFrom(someone, ZERO_ADDRESS, SOMEONE_TOKEN_IDS[0], transact={'from': someone}))
    assert_tx_failed(lambda : c.transferFrom(someone, operator, OPERATOR_TOKEN_ID, transact={'from': someone}))
    assert_tx_failed(lambda : c.transferFrom(someone, operator, INVALID_TOKEN_ID, transact={'from': someone}))
    tx_hash = c.transferFrom(someone, operator, SOMEONE_TOKEN_IDS[0], transact={'from': someone})
    logs = get_logs(tx_hash, c, 'Transfer')
    assert len(logs) > 0
    args = logs[0].args
    assert args.sender == someone
    assert args.receiver == operator
    assert args.tokenId == SOMEONE_TOKEN_IDS[0]
    assert c.ownerOf(SOMEONE_TOKEN_IDS[0]) == operator
    assert c.balanceOf(someone) == 2
    assert c.balanceOf(operator) == 2

def test_transferFrom_by_approved(c, w3, get_logs):
    if False:
        return 10
    (someone, operator) = w3.eth.accounts[1:3]
    c.approve(operator, SOMEONE_TOKEN_IDS[1], transact={'from': someone})
    tx_hash = c.transferFrom(someone, operator, SOMEONE_TOKEN_IDS[1], transact={'from': operator})
    logs = get_logs(tx_hash, c, 'Transfer')
    assert len(logs) > 0
    args = logs[0].args
    assert args.sender == someone
    assert args.receiver == operator
    assert args.tokenId == SOMEONE_TOKEN_IDS[1]
    assert c.ownerOf(SOMEONE_TOKEN_IDS[1]) == operator
    assert c.balanceOf(someone) == 2
    assert c.balanceOf(operator) == 2

def test_transferFrom_by_operator(c, w3, get_logs):
    if False:
        print('Hello World!')
    (someone, operator) = w3.eth.accounts[1:3]
    c.setApprovalForAll(operator, True, transact={'from': someone})
    tx_hash = c.transferFrom(someone, operator, SOMEONE_TOKEN_IDS[2], transact={'from': operator})
    logs = get_logs(tx_hash, c, 'Transfer')
    assert len(logs) > 0
    args = logs[0].args
    assert args.sender == someone
    assert args.receiver == operator
    assert args.tokenId == SOMEONE_TOKEN_IDS[2]
    assert c.ownerOf(SOMEONE_TOKEN_IDS[2]) == operator
    assert c.balanceOf(someone) == 2
    assert c.balanceOf(operator) == 2

def test_safeTransferFrom_by_owner(c, w3, assert_tx_failed, get_logs):
    if False:
        print('Hello World!')
    (someone, operator) = w3.eth.accounts[1:3]
    assert_tx_failed(lambda : c.safeTransferFrom(ZERO_ADDRESS, operator, SOMEONE_TOKEN_IDS[0], transact={'from': someone}))
    assert_tx_failed(lambda : c.safeTransferFrom(someone, ZERO_ADDRESS, SOMEONE_TOKEN_IDS[0], transact={'from': someone}))
    assert_tx_failed(lambda : c.safeTransferFrom(someone, operator, OPERATOR_TOKEN_ID, transact={'from': someone}))
    assert_tx_failed(lambda : c.safeTransferFrom(someone, operator, INVALID_TOKEN_ID, transact={'from': someone}))
    tx_hash = c.safeTransferFrom(someone, operator, SOMEONE_TOKEN_IDS[0], transact={'from': someone})
    logs = get_logs(tx_hash, c, 'Transfer')
    assert len(logs) > 0
    args = logs[0].args
    assert args.sender == someone
    assert args.receiver == operator
    assert args.tokenId == SOMEONE_TOKEN_IDS[0]
    assert c.ownerOf(SOMEONE_TOKEN_IDS[0]) == operator
    assert c.balanceOf(someone) == 2
    assert c.balanceOf(operator) == 2

def test_safeTransferFrom_by_approved(c, w3, get_logs):
    if False:
        for i in range(10):
            print('nop')
    (someone, operator) = w3.eth.accounts[1:3]
    c.approve(operator, SOMEONE_TOKEN_IDS[1], transact={'from': someone})
    tx_hash = c.safeTransferFrom(someone, operator, SOMEONE_TOKEN_IDS[1], transact={'from': operator})
    logs = get_logs(tx_hash, c, 'Transfer')
    assert len(logs) > 0
    args = logs[0].args
    assert args.sender == someone
    assert args.receiver == operator
    assert args.tokenId == SOMEONE_TOKEN_IDS[1]
    assert c.ownerOf(SOMEONE_TOKEN_IDS[1]) == operator
    assert c.balanceOf(someone) == 2
    assert c.balanceOf(operator) == 2

def test_safeTransferFrom_by_operator(c, w3, get_logs):
    if False:
        return 10
    (someone, operator) = w3.eth.accounts[1:3]
    c.setApprovalForAll(operator, True, transact={'from': someone})
    tx_hash = c.safeTransferFrom(someone, operator, SOMEONE_TOKEN_IDS[2], transact={'from': operator})
    logs = get_logs(tx_hash, c, 'Transfer')
    assert len(logs) > 0
    args = logs[0].args
    assert args.sender == someone
    assert args.receiver == operator
    assert args.tokenId == SOMEONE_TOKEN_IDS[2]
    assert c.ownerOf(SOMEONE_TOKEN_IDS[2]) == operator
    assert c.balanceOf(someone) == 2
    assert c.balanceOf(operator) == 2

def test_safeTransferFrom_to_contract(c, w3, assert_tx_failed, get_logs, get_contract):
    if False:
        while True:
            i = 10
    someone = w3.eth.accounts[1]
    assert_tx_failed(lambda : c.safeTransferFrom(someone, c.address, SOMEONE_TOKEN_IDS[0], transact={'from': someone}))
    receiver = get_contract('\n@external\ndef onERC721Received(\n        _operator: address,\n        _from: address,\n        _tokenId: uint256,\n        _data: Bytes[1024]\n    ) -> bytes4:\n    return method_id("onERC721Received(address,address,uint256,bytes)", output_type=bytes4)\n    ')
    tx_hash = c.safeTransferFrom(someone, receiver.address, SOMEONE_TOKEN_IDS[0], transact={'from': someone})
    logs = get_logs(tx_hash, c, 'Transfer')
    assert len(logs) > 0
    args = logs[0].args
    assert args.sender == someone
    assert args.receiver == receiver.address
    assert args.tokenId == SOMEONE_TOKEN_IDS[0]
    assert c.ownerOf(SOMEONE_TOKEN_IDS[0]) == receiver.address
    assert c.balanceOf(someone) == 2
    assert c.balanceOf(receiver.address) == 1

def test_approve(c, w3, assert_tx_failed, get_logs):
    if False:
        while True:
            i = 10
    (someone, operator) = w3.eth.accounts[1:3]
    assert_tx_failed(lambda : c.approve(someone, SOMEONE_TOKEN_IDS[0], transact={'from': someone}))
    assert_tx_failed(lambda : c.approve(operator, OPERATOR_TOKEN_ID, transact={'from': someone}))
    assert_tx_failed(lambda : c.approve(operator, INVALID_TOKEN_ID, transact={'from': someone}))
    tx_hash = c.approve(operator, SOMEONE_TOKEN_IDS[0], transact={'from': someone})
    logs = get_logs(tx_hash, c, 'Approval')
    assert len(logs) > 0
    args = logs[0].args
    assert args.owner == someone
    assert args.approved == operator
    assert args.tokenId == SOMEONE_TOKEN_IDS[0]

def test_setApprovalForAll(c, w3, assert_tx_failed, get_logs):
    if False:
        while True:
            i = 10
    (someone, operator) = w3.eth.accounts[1:3]
    approved = True
    assert_tx_failed(lambda : c.setApprovalForAll(someone, approved, transact={'from': someone}))
    tx_hash = c.setApprovalForAll(operator, approved, transact={'from': someone})
    logs = get_logs(tx_hash, c, 'ApprovalForAll')
    assert len(logs) > 0
    args = logs[0].args
    assert args.owner == someone
    assert args.operator == operator
    assert args.approved == approved

def test_mint(c, w3, assert_tx_failed, get_logs):
    if False:
        while True:
            i = 10
    (minter, someone) = w3.eth.accounts[:2]
    assert_tx_failed(lambda : c.mint(someone, SOMEONE_TOKEN_IDS[0], transact={'from': someone}))
    assert_tx_failed(lambda : c.mint(ZERO_ADDRESS, SOMEONE_TOKEN_IDS[0], transact={'from': minter}))
    tx_hash = c.mint(someone, NEW_TOKEN_ID, transact={'from': minter})
    logs = get_logs(tx_hash, c, 'Transfer')
    assert len(logs) > 0
    args = logs[0].args
    assert args.sender == ZERO_ADDRESS
    assert args.receiver == someone
    assert args.tokenId == NEW_TOKEN_ID
    assert c.ownerOf(NEW_TOKEN_ID) == someone
    assert c.balanceOf(someone) == 4

def test_burn(c, w3, assert_tx_failed, get_logs):
    if False:
        i = 10
        return i + 15
    (someone, operator) = w3.eth.accounts[1:3]
    assert_tx_failed(lambda : c.burn(SOMEONE_TOKEN_IDS[0], transact={'from': operator}))
    tx_hash = c.burn(SOMEONE_TOKEN_IDS[0], transact={'from': someone})
    logs = get_logs(tx_hash, c, 'Transfer')
    assert len(logs) > 0
    args = logs[0].args
    assert args.sender == someone
    assert args.receiver == ZERO_ADDRESS
    assert args.tokenId == SOMEONE_TOKEN_IDS[0]
    assert_tx_failed(lambda : c.ownerOf(SOMEONE_TOKEN_IDS[0]))
    assert c.balanceOf(someone) == 2