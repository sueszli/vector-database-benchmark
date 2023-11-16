from pytest import raises
from vyper.exceptions import UndeclaredDefinition

def test_permanent_variables_test(get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    permanent_variables_test = '\nstruct Var:\n    a: int128\n    b: int128\nvar: Var\n\n@external\ndef __init__(a: int128, b: int128):\n    self.var.a = a\n    self.var.b = b\n\n@external\ndef returnMoose() -> int128:\n    return self.var.a * 10 + self.var.b\n    '
    c = get_contract_with_gas_estimation(permanent_variables_test, *[5, 7])
    assert c.returnMoose() == 57
    print('Passed init argument and variable member test')

def test_missing_global(get_contract):
    if False:
        print('Hello World!')
    code = '\n@external\ndef a() -> int128:\n    return self.b\n    '
    with raises(UndeclaredDefinition):
        get_contract(code)