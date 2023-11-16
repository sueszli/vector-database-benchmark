def test_send(assert_tx_failed, get_contract):
    if False:
        while True:
            i = 10
    send_test = '\n@external\ndef foo():\n    send(msg.sender, self.balance + 1)\n\n@external\ndef fop():\n    send(msg.sender, 10)\n    '
    c = get_contract(send_test, value=10)
    assert_tx_failed(lambda : c.foo(transact={}))
    c.fop(transact={})
    assert_tx_failed(lambda : c.fop(transact={}))

def test_default_gas(get_contract, w3):
    if False:
        print('Hello World!')
    '\n    Tests to verify that send to default function will send limited gas (2300),\n    but raw_call can send more.\n    '
    sender_code = '\n@external\ndef test_send(receiver: address):\n    send(receiver, 1)\n\n@external\ndef test_call(receiver: address):\n    raw_call(receiver, b"", gas=50000, max_outsize=0, value=1)\n    '
    receiver_code = '\nlast_sender: public(address)\n\n@external\n@payable\ndef __default__():\n    self.last_sender = msg.sender\n    '
    sender = get_contract(sender_code, value=1)
    receiver = get_contract(receiver_code)
    sender.test_send(receiver.address, transact={'gas': 100000})
    assert receiver.last_sender() is None
    assert w3.eth.get_balance(sender.address) == 1
    assert w3.eth.get_balance(receiver.address) == 0
    sender.test_call(receiver.address, transact={'gas': 100000})
    assert receiver.last_sender() == sender.address
    assert w3.eth.get_balance(sender.address) == 0
    assert w3.eth.get_balance(receiver.address) == 1

def test_send_gas_stipend(get_contract, w3):
    if False:
        print('Hello World!')
    '\n    Tests to verify that adding gas stipend to send() will send sufficient gas\n    '
    sender_code = '\n\n@external\ndef test_send_stipend(receiver: address):\n    send(receiver, 1, gas=50000)\n    '
    receiver_code = '\nlast_sender: public(address)\n\n@external\n@payable\ndef __default__():\n    self.last_sender = msg.sender\n    '
    sender = get_contract(sender_code, value=1)
    receiver = get_contract(receiver_code)
    sender.test_send_stipend(receiver.address, transact={'gas': 100000})
    assert receiver.last_sender() == sender.address
    assert w3.eth.get_balance(sender.address) == 0
    assert w3.eth.get_balance(receiver.address) == 1