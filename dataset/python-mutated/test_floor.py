import math
from decimal import Decimal

def test_floor(get_contract_with_gas_estimation):
    if False:
        return 10
    code = '\nx: decimal\n\n@external\ndef __init__():\n    self.x = 504.0000000001\n\n@external\ndef x_floor() -> int256:\n    return floor(self.x)\n\n@external\ndef foo() -> int256:\n    return floor(1.9999999999)\n\n@external\ndef fop() -> int256:\n    return floor(1.0000000001)\n\n@external\ndef foq() -> int256:\n    return floor(18707220957835557353007165858768422651595.9365500927)\n\n@external\ndef fos() -> int256:\n    return floor(0.0)\n\n@external\ndef fot() -> int256:\n    return floor(0.0000000001)\n\n@external\ndef fou() -> int256:\n    a: decimal = 305.0\n    b: decimal = 100.0\n    c: decimal = a / b\n    return floor(c)\n'
    c = get_contract_with_gas_estimation(code)
    assert c.x_floor() == 504
    assert c.foo() == 1
    assert c.fop() == 1
    assert c.foq() == math.floor(Decimal(2 ** 167 - 1) / 10 ** 10)
    assert c.fos() == 0
    assert c.fot() == 0
    assert c.fou() == 3

def test_floor_negative(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    code = '\nx: decimal\n\n@external\ndef __init__():\n    self.x = -504.0000000001\n\n@external\ndef x_floor() -> int256:\n    return floor(self.x)\n\n@external\ndef foo() -> int256:\n    a: int128 = -65\n    b: decimal = convert(a, decimal) / 10.0\n    return floor(b)\n\n@external\ndef fop() -> int256:\n    return floor(-27.0)\n\n@external\ndef foq() -> int256:\n    return floor(-9000.0000000001)\n\n@external\ndef fos() -> int256:\n    return floor(-0.0000000001)\n\n@external\ndef fot() -> int256:\n    return floor(-18707220957835557353007165858768422651595.9365500928)\n\n@external\ndef fou() -> int256:\n    a: decimal = -305.0\n    b: decimal = 100.0\n    c: decimal = a / b\n    return floor(c)\n\n@external\ndef floor_param(p: decimal) -> int256:\n    return floor(p)\n'
    c = get_contract_with_gas_estimation(code)
    assert c.x_floor() == -505
    assert c.foo() == -7
    assert c.fop() == -27
    assert c.foq() == -9001
    assert c.fos() == -1
    assert c.fot() == math.floor(-Decimal(2 ** 167) / 10 ** 10)
    assert c.fou() == -4
    assert c.floor_param(Decimal('-5.6')) == -6
    assert c.floor_param(Decimal('-0.0000000001')) == -1

def test_floor_ext_call(w3, side_effects_contract, assert_side_effects_invoked, get_contract):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo(a: Foo) -> int256:\n    return floor(a.foo(2.5))\n\ninterface Foo:\n    def foo(x: decimal) -> decimal: nonpayable\n    '
    c1 = side_effects_contract('decimal')
    c2 = get_contract(code)
    assert c2.foo(c1.address) == 2
    assert_side_effects_invoked(c1, lambda : c2.foo(c1.address, transact={}))

def test_floor_internal_call(get_contract_with_gas_estimation):
    if False:
        return 10
    code = '\n@external\ndef foo() -> int256:\n    return floor(self.bar())\n\n@internal\ndef bar() -> decimal:\n    return 2.5\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.foo() == 2