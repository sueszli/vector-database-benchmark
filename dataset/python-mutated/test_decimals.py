import warnings
from decimal import ROUND_DOWN, Decimal, getcontext
import pytest
from vyper.exceptions import DecimalOverrideException, InvalidOperation, TypeMismatch
from vyper.utils import DECIMAL_EPSILON, SizeLimits

def test_decimal_override():
    if False:
        while True:
            i = 10
    getcontext().prec = 78
    with pytest.raises(DecimalOverrideException):
        getcontext().prec = 77
    with warnings.catch_warnings(record=True) as w:
        getcontext().prec = 79
        assert len(w) == 1
        assert str(w[-1].message) == 'Changing decimals precision could have unintended side effects!'

@pytest.mark.parametrize('op', ['**', '&', '|', '^'])
def test_invalid_ops(get_contract, assert_compile_failed, op):
    if False:
        i = 10
        return i + 15
    code = f'\n@external\ndef foo(x: decimal, y: decimal) -> decimal:\n    return x {op} y\n    '
    assert_compile_failed(lambda : get_contract(code), InvalidOperation)

@pytest.mark.parametrize('op', ['not'])
def test_invalid_unary_ops(get_contract, assert_compile_failed, op):
    if False:
        i = 10
        return i + 15
    code = f'\n@external\ndef foo(x: decimal) -> decimal:\n    return {op} x\n    '
    assert_compile_failed(lambda : get_contract(code), InvalidOperation)

def quantize(x: Decimal) -> Decimal:
    if False:
        return 10
    return x.quantize(DECIMAL_EPSILON, rounding=ROUND_DOWN)

def test_decimal_test(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    decimal_test = '\n@external\ndef foo() -> int256:\n    return(floor(999.0))\n\n@external\ndef fop() -> int256:\n    return(floor(333.0 + 666.0))\n\n@external\ndef foq() -> int256:\n    return(floor(1332.1 - 333.1))\n\n@external\ndef bar() -> int256:\n    return(floor(27.0 * 37.0))\n\n@external\ndef baz() -> int256:\n    x: decimal = 27.0\n    return(floor(x * 37.0))\n\n@external\ndef mok() -> int256:\n    return(floor(999999.0 / 7.0 / 11.0 / 13.0))\n\n@external\ndef mol() -> int256:\n    return(floor(499.5 / 0.5))\n\n@external\ndef mom() -> int256:\n    return(floor(1498.5 / 1.5))\n\n@external\ndef moo() -> int256:\n    return(floor(2997.0 / 3.0))\n\n@external\ndef foom() -> int256:\n    return(floor(1999.0 % 1000.0))\n\n@external\ndef foop() -> int256:\n    return(floor(1999.0 % 1000.0))\n    '
    c = get_contract_with_gas_estimation(decimal_test)
    assert c.foo() == 999
    assert c.fop() == 999
    assert c.foq() == 999
    assert c.bar() == 999
    assert c.baz() == 999
    assert c.mok() == 999
    assert c.mol() == 999
    assert c.mom() == 999
    assert c.moo() == 999
    assert c.foom() == 999
    assert c.foop() == 999
    print('Passed basic addition, subtraction and multiplication tests')

def test_harder_decimal_test(get_contract_with_gas_estimation):
    if False:
        return 10
    harder_decimal_test = '\n@external\ndef phooey(inp: decimal) -> decimal:\n    x: decimal = 10000.0\n    for i in range(4):\n        x = x * inp\n    return x\n\n@external\ndef arg(inp: decimal) -> decimal:\n    return inp\n\n@external\ndef garg() -> decimal:\n    x: decimal = 4.5\n    x *= 1.5\n    return x\n\n@external\ndef harg() -> decimal:\n    x: decimal = 4.5\n    x *= 2.0\n    return x\n\n@external\ndef iarg() -> uint256:\n    x: uint256 = as_wei_value(7, "wei")\n    x *= 2\n    return x\n    '
    c = get_contract_with_gas_estimation(harder_decimal_test)
    assert c.phooey(Decimal('1.2')) == Decimal('20736.0')
    assert c.phooey(Decimal('-1.2')) == Decimal('20736.0')
    assert c.arg(Decimal('-3.7')) == Decimal('-3.7')
    assert c.arg(Decimal('3.7')) == Decimal('3.7')
    assert c.garg() == Decimal('6.75')
    assert c.harg() == Decimal('9.0')
    assert c.iarg() == Decimal('14')
    print('Passed fractional multiplication test')

def test_mul_overflow(assert_tx_failed, get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    mul_code = '\n\n@external\ndef _num_mul(x: decimal, y: decimal) -> decimal:\n    return x * y\n\n    '
    c = get_contract_with_gas_estimation(mul_code)
    x = Decimal('85070591730234615865843651857942052864')
    y = Decimal('136112946768375385385349842973')
    assert_tx_failed(lambda : c._num_mul(x, y))
    x = SizeLimits.MAX_AST_DECIMAL
    y = 1 + DECIMAL_EPSILON
    assert_tx_failed(lambda : c._num_mul(x, y))
    assert c._num_mul(x, Decimal(1)) == x
    assert c._num_mul(x, 1 - DECIMAL_EPSILON) == quantize(x * (1 - DECIMAL_EPSILON))
    x = SizeLimits.MIN_AST_DECIMAL
    assert c._num_mul(x, 1 - DECIMAL_EPSILON) == quantize(x * (1 - DECIMAL_EPSILON))

def test_div_overflow(get_contract, assert_tx_failed):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo(x: decimal, y: decimal) -> decimal:\n    return x / y\n    '
    c = get_contract(code)
    x = SizeLimits.MIN_AST_DECIMAL
    y = -DECIMAL_EPSILON
    assert_tx_failed(lambda : c.foo(x, y))
    assert_tx_failed(lambda : c.foo(x, Decimal(0)))
    assert_tx_failed(lambda : c.foo(y, Decimal(0)))
    y = Decimal(1) - DECIMAL_EPSILON
    assert_tx_failed(lambda : c.foo(x, y))
    y = Decimal(-1)
    assert_tx_failed(lambda : c.foo(x, y))
    assert c.foo(x, Decimal(1)) == x
    assert c.foo(x, 1 + DECIMAL_EPSILON) == quantize(x / (1 + DECIMAL_EPSILON))
    x = SizeLimits.MAX_AST_DECIMAL
    assert_tx_failed(lambda : c.foo(x, DECIMAL_EPSILON))
    y = Decimal(1) - DECIMAL_EPSILON
    assert_tx_failed(lambda : c.foo(x, y))
    assert c.foo(x, Decimal(1)) == x
    assert c.foo(x, 1 + DECIMAL_EPSILON) == quantize(x / (1 + DECIMAL_EPSILON))

def test_decimal_min_max_literals(assert_tx_failed, get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef maximum():\n    a: decimal = 18707220957835557353007165858768422651595.9365500927\n@external\ndef minimum():\n    a: decimal = -18707220957835557353007165858768422651595.9365500928\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.maximum() == []
    assert c.minimum() == []

def test_scientific_notation(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\ndef foo() -> decimal:\n    return 1e-10\n\n@external\ndef bar(num: decimal) -> decimal:\n    return num + -1e38\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.foo() == Decimal('1e-10')
    assert c.bar(Decimal('1e37')) == Decimal('-9e37')

def test_exponents(assert_compile_failed, get_contract):
    if False:
        print('Hello World!')
    code = '\n@external\ndef foo() -> decimal:\n    return 2.2 ** 2.0\n    '
    assert_compile_failed(lambda : get_contract(code), TypeMismatch)