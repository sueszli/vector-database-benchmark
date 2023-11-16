from decimal import ROUND_FLOOR, Decimal
import hypothesis
import pytest
from eth_tester.exceptions import TransactionFailed
from vyper.utils import SizeLimits
DECIMAL_PLACES = 10
DECIMAL_RANGE = [Decimal('0.' + '0' * d + '2') for d in range(0, DECIMAL_PLACES)]

def decimal_truncate(val, decimal_places=DECIMAL_PLACES, rounding=ROUND_FLOOR):
    if False:
        while True:
            i = 10
    q = '0'
    if decimal_places != 0:
        q += '.' + '0' * decimal_places
    return val.quantize(Decimal(q), rounding=rounding)

def decimal_sqrt(val):
    if False:
        for i in range(10):
            print('nop')
    return decimal_truncate(val.sqrt())

def test_sqrt_literal(get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    code = '\n@external\ndef test() -> decimal:\n    return sqrt(2.0)\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.test() == decimal_sqrt(Decimal('2'))

def test_sqrt_variable(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef test(a: decimal) -> decimal:\n    return sqrt(a)\n\n@external\ndef test2() -> decimal:\n    a: decimal = 44.001\n    return sqrt(a)\n    '
    c = get_contract_with_gas_estimation(code)
    val = Decimal('33.33')
    assert c.test(val) == decimal_sqrt(val)
    val = Decimal('0.1')
    assert c.test(val) == decimal_sqrt(val)
    assert c.test(Decimal('0.0')) == Decimal('0.0')
    assert c.test2() == decimal_sqrt(Decimal('44.001'))

def test_sqrt_storage(get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    code = '\ns_var: decimal\n\n@external\ndef test(a: decimal) -> decimal:\n    self.s_var = a + 1.0\n    return sqrt(self.s_var)\n\n@external\ndef test2() -> decimal:\n    self.s_var = 444.44\n    return sqrt(self.s_var)\n    '
    c = get_contract_with_gas_estimation(code)
    val = Decimal('12.21')
    assert c.test(val) == decimal_sqrt(val + 1)
    val = Decimal('100.01')
    assert c.test(val) == decimal_sqrt(val + 1)
    assert c.test2() == decimal_sqrt(Decimal('444.44'))

def test_sqrt_inline_memory_correct(get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    code = "\n@external\ndef test(a: decimal) -> (decimal, decimal, decimal, decimal, decimal, String[100]):\n    x: decimal = 1.0\n    y: decimal = 2.0\n    z: decimal = 3.0\n    e: decimal = sqrt(a)\n    f: String[100] = 'hello world'\n    return a, x, y, z, e, f\n    "
    c = get_contract_with_gas_estimation(code)
    val = Decimal('2.1')
    assert c.test(val) == [val, Decimal('1'), Decimal('2'), Decimal('3'), decimal_sqrt(val), 'hello world']

@pytest.mark.parametrize('value', DECIMAL_RANGE)
def test_sqrt_sub_decimal_places(value, get_contract):
    if False:
        return 10
    code = '\n@external\ndef test(a: decimal) -> decimal:\n    return sqrt(a)\n    '
    c = get_contract(code)
    vyper_sqrt = c.test(value)
    actual_sqrt = decimal_sqrt(value)
    assert vyper_sqrt == actual_sqrt

@pytest.fixture(scope='module')
def sqrt_contract(get_contract_module):
    if False:
        print('Hello World!')
    code = '\n@external\ndef test(a: decimal) -> decimal:\n    return sqrt(a)\n    '
    c = get_contract_module(code)
    return c

@pytest.mark.parametrize('value', [Decimal(0), Decimal(SizeLimits.MAX_INT128)])
def test_sqrt_bounds(sqrt_contract, value):
    if False:
        i = 10
        return i + 15
    vyper_sqrt = sqrt_contract.test(value)
    actual_sqrt = decimal_sqrt(value)
    assert vyper_sqrt == actual_sqrt

@pytest.mark.fuzzing
@hypothesis.given(value=hypothesis.strategies.decimals(min_value=Decimal(0), max_value=Decimal(SizeLimits.MAX_INT128), places=DECIMAL_PLACES))
@hypothesis.example(value=Decimal(SizeLimits.MAX_INT128))
@hypothesis.example(value=Decimal(0))
def test_sqrt_valid_range(sqrt_contract, value):
    if False:
        for i in range(10):
            print('nop')
    vyper_sqrt = sqrt_contract.test(value)
    actual_sqrt = decimal_sqrt(value)
    assert vyper_sqrt == actual_sqrt

@pytest.mark.fuzzing
@hypothesis.given(value=hypothesis.strategies.decimals(min_value=Decimal(SizeLimits.MIN_INT128), max_value=Decimal('-1E10'), places=DECIMAL_PLACES))
@hypothesis.example(value=Decimal(SizeLimits.MIN_INT128))
@hypothesis.example(value=Decimal('-1E10'))
def test_sqrt_invalid_range(sqrt_contract, value):
    if False:
        print('Hello World!')
    with pytest.raises(TransactionFailed):
        sqrt_contract.test(value)