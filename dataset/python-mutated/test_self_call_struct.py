from decimal import Decimal

def test_call_to_self_struct(w3, get_contract):
    if False:
        print('Hello World!')
    code = '\nstruct MyStruct:\n    e1: decimal\n    e2: uint256\n\n@internal\n@view\ndef get_my_struct(_e1: decimal, _e2: uint256) -> MyStruct:\n    return MyStruct({e1: _e1, e2: _e2})\n\n@external\n@view\ndef wrap_get_my_struct_WORKING(_e1: decimal) -> MyStruct:\n    testing: MyStruct = self.get_my_struct(_e1, block.timestamp)\n    return testing\n\n@external\n@view\ndef wrap_get_my_struct_BROKEN(_e1: decimal) -> MyStruct:\n    return self.get_my_struct(_e1, block.timestamp)\n    '
    c = get_contract(code)
    assert c.wrap_get_my_struct_WORKING(Decimal('0.1')) == (Decimal('0.1'), w3.eth.get_block(w3.eth.block_number)['timestamp'])
    assert c.wrap_get_my_struct_BROKEN(Decimal('0.1')) == (Decimal('0.1'), w3.eth.get_block(w3.eth.block_number)['timestamp'])

def test_call_to_self_struct_2(get_contract):
    if False:
        i = 10
        return i + 15
    code = '\nstruct MyStruct:\n    e1: decimal\n\n@internal\n@view\ndef get_my_struct(_e1: decimal) -> MyStruct:\n    return MyStruct({e1: _e1})\n\n@external\n@view\ndef wrap_get_my_struct_WORKING(_e1: decimal) -> MyStruct:\n    testing: MyStruct = self.get_my_struct(_e1)\n    return testing\n\n@external\n@view\ndef wrap_get_my_struct_BROKEN(_e1: decimal) -> MyStruct:\n    return self.get_my_struct(_e1)\n    '
    c = get_contract(code)
    assert c.wrap_get_my_struct_WORKING(Decimal('0.1')) == (Decimal('0.1'),)
    assert c.wrap_get_my_struct_BROKEN(Decimal('0.1')) == (Decimal('0.1'),)