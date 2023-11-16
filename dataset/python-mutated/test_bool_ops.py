def test_convert_from_bool(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo() -> bool:\n    val: bool = True and True and False\n    return val\n\n@external\ndef bar() -> bool:\n    val: bool = True or True or False\n    return val\n\n@external\ndef foobar() -> bool:\n    val: bool = False and True or False\n    return val\n\n@external\ndef oof() -> bool:\n    val: bool = False or False or False or False or False or True\n    return val\n\n@external\ndef rab() -> bool:\n    val: bool = True and True and True and True and True and False\n    return val\n\n@external\ndef oofrab() -> bool:\n    val: bool = False and True or False and True or False and False or True\n    return val\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.foo() is False
    assert c.bar() is True
    assert c.foobar() is False
    assert c.oof() is True
    assert c.rab() is False
    assert c.oofrab() is True