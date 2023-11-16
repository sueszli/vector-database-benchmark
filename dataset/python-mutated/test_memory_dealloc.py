def test_memory_deallocation(get_contract):
    if False:
        print('Hello World!')
    code = '\nevent Shimmy:\n    a: indexed(address)\n    b: uint256\n\ninterface Other:\n    def sendit(): nonpayable\n\n@external\ndef foo(target: address) -> uint256[2]:\n    log Shimmy(empty(address), 3)\n    amount: uint256 = 1\n    flargen: uint256 = 42\n    Other(target).sendit()\n    return [amount, flargen]\n    '
    code2 = '\n\n@external\ndef sendit() -> bool:\n     return True\n    '
    c = get_contract(code)
    c2 = get_contract(code2)
    assert c.foo(c2.address) == [1, 42]