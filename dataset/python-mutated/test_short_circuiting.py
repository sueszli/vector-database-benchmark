import itertools
import pytest

def test_short_circuit_and_left_is_false(w3, get_contract):
    if False:
        for i in range(10):
            print('nop')
    code = '\n\ncalled_left: public(bool)\ncalled_right: public(bool)\n\n@internal\ndef left() -> bool:\n    self.called_left = True\n    return False\n\n@internal\ndef right() -> bool:\n    self.called_right = True\n    return False\n\n@external\ndef foo() -> bool:\n    return self.left() and self.right()\n'
    c = get_contract(code)
    assert not c.foo()
    c.foo(transact={})
    assert c.called_left()
    assert not c.called_right()

def test_short_circuit_and_left_is_true(w3, get_contract):
    if False:
        for i in range(10):
            print('nop')
    code = '\n\ncalled_left: public(bool)\ncalled_right: public(bool)\n\n@internal\ndef left() -> bool:\n    self.called_left = True\n    return True\n\n@internal\ndef right() -> bool:\n    self.called_right = True\n    return True\n\n@external\ndef foo() -> bool:\n    return self.left() and self.right()\n'
    c = get_contract(code)
    assert c.foo()
    c.foo(transact={})
    assert c.called_left()
    assert c.called_right()

def test_short_circuit_or_left_is_true(w3, get_contract):
    if False:
        i = 10
        return i + 15
    code = '\n\ncalled_left: public(bool)\ncalled_right: public(bool)\n\n@internal\ndef left() -> bool:\n    self.called_left = True\n    return True\n\n@internal\ndef right() -> bool:\n    self.called_right = True\n    return True\n\n@external\ndef foo() -> bool:\n    return self.left() or self.right()\n'
    c = get_contract(code)
    assert c.foo()
    c.foo(transact={})
    assert c.called_left()
    assert not c.called_right()

def test_short_circuit_or_left_is_false(w3, get_contract):
    if False:
        i = 10
        return i + 15
    code = '\n\ncalled_left: public(bool)\ncalled_right: public(bool)\n\n@internal\ndef left() -> bool:\n    self.called_left = True\n    return False\n\n@internal\ndef right() -> bool:\n    self.called_right = True\n    return False\n\n@external\ndef foo() -> bool:\n    return self.left() or self.right()\n'
    c = get_contract(code)
    assert not c.foo()
    c.foo(transact={})
    assert c.called_left()
    assert c.called_right()

@pytest.mark.parametrize('op', ['and', 'or'])
@pytest.mark.parametrize('a, b', itertools.product([True, False], repeat=2))
def test_from_memory(w3, get_contract, a, b, op):
    if False:
        i = 10
        return i + 15
    code = f'\n@external\ndef foo(a: bool, b: bool) -> bool:\n    c: bool = a\n    d: bool = b\n    return c {op} d\n'
    c = get_contract(code)
    assert c.foo(a, b) == eval(f'{a} {op} {b}')

@pytest.mark.parametrize('op', ['and', 'or'])
@pytest.mark.parametrize('a, b', itertools.product([True, False], repeat=2))
def test_from_storage(w3, get_contract, a, b, op):
    if False:
        for i in range(10):
            print('nop')
    code = f'\nc: bool\nd: bool\n\n@external\ndef foo(a: bool, b: bool) -> bool:\n    self.c = a\n    self.d = b\n    return self.c {op} self.d\n'
    c = get_contract(code)
    assert c.foo(a, b) == eval(f'{a} {op} {b}')

@pytest.mark.parametrize('op', ['and', 'or'])
@pytest.mark.parametrize('a, b', itertools.product([True, False], repeat=2))
def test_from_calldata(w3, get_contract, a, b, op):
    if False:
        return 10
    code = f'\n@external\ndef foo(a: bool, b: bool) -> bool:\n    return a {op} b\n'
    c = get_contract(code)
    assert c.foo(a, b) == eval(f'{a} {op} {b}')

@pytest.mark.parametrize('a, b, c, d', itertools.product([True, False], repeat=4))
@pytest.mark.parametrize('ops', itertools.product(['and', 'or'], repeat=3))
def test_complex_combination(w3, get_contract, a, b, c, d, ops):
    if False:
        return 10
    boolop = f'a {ops[0]} b {ops[1]} c {ops[2]} d'
    code = f'\n@external\ndef foo(a: bool, b: bool, c: bool, d: bool) -> bool:\n    return {boolop}\n'
    contract = get_contract(code)
    if eval(boolop):
        assert contract.foo(a, b, c, d)
    else:
        assert not contract.foo(a, b, c, d)