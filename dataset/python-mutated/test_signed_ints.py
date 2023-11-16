import itertools
import operator
import random
import pytest
from vyper.exceptions import InvalidOperation, InvalidType, OverflowException, ZeroDivisionException
from vyper.semantics.types import IntegerT
from vyper.utils import evm_div, evm_mod
types = sorted(IntegerT.signeds())

@pytest.mark.parametrize('typ', types)
def test_exponent_base_zero(get_contract, assert_tx_failed, typ):
    if False:
        for i in range(10):
            print('nop')
    code = f'\n@external\ndef foo(x: {typ}) -> {typ}:\n    return 0 ** x\n    '
    (lo, hi) = typ.ast_bounds
    c = get_contract(code)
    assert c.foo(0) == 1
    assert c.foo(1) == 0
    assert c.foo(hi) == 0
    assert_tx_failed(lambda : c.foo(-1))
    assert_tx_failed(lambda : c.foo(lo))

@pytest.mark.parametrize('typ', types)
def test_exponent_base_one(get_contract, assert_tx_failed, typ):
    if False:
        for i in range(10):
            print('nop')
    code = f'\n@external\ndef foo(x: {typ}) -> {typ}:\n    return 1 ** x\n    '
    (lo, hi) = typ.ast_bounds
    c = get_contract(code)
    assert c.foo(0) == 1
    assert c.foo(1) == 1
    assert c.foo(hi) == 1
    assert_tx_failed(lambda : c.foo(-1))
    assert_tx_failed(lambda : c.foo(lo))

def test_exponent_base_minus_one(get_contract):
    if False:
        return 10
    code = '\n@external\ndef foo(x: int256) -> int256:\n    y: int256 = (-1) ** x\n    return y\n    '
    c = get_contract(code)
    for x in range(5):
        assert c.foo(x) == (-1) ** x

@pytest.mark.parametrize('base', (0, 1))
def test_exponent_negative_power(get_contract, assert_tx_failed, base):
    if False:
        for i in range(10):
            print('nop')
    code = f'\n@external\ndef bar() -> int16:\n    x: int16 = -2\n    return {base} ** x\n    '
    c = get_contract(code)
    assert_tx_failed(lambda : c.bar())

def test_exponent_min_int16(get_contract):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo() -> int16:\n    x: int16 = -8\n    y: int16 = x ** 5\n    return y\n    '
    c = get_contract(code)
    assert c.foo() == -2 ** 15

@pytest.mark.parametrize('base,power', itertools.product((-2, -1, 0, 1, 2), (0, 1)))
def test_exponent_power_zero_one(get_contract, base, power):
    if False:
        for i in range(10):
            print('nop')
    code = f'\n@external\ndef foo() -> int256:\n    x: int256 = {base}\n    return x ** {power}\n    '
    c = get_contract(code)
    assert c.foo() == base ** power

@pytest.mark.parametrize('typ', types)
def test_exponent(get_contract, assert_tx_failed, typ):
    if False:
        for i in range(10):
            print('nop')
    code = f'\n@external\ndef foo(x: {typ}) -> {typ}:\n    return 4 ** x\n    '
    (lo, hi) = typ.ast_bounds
    c = get_contract(code)
    test_cases = [0, 1, 3, 4, 126, 127, -1, lo, hi]
    for x in test_cases:
        if x * 2 >= typ.bits or x < 0:
            assert_tx_failed(lambda : c.foo(x))
        else:
            assert c.foo(x) == 4 ** x

@pytest.mark.parametrize('typ', types)
def test_negative_nums(get_contract_with_gas_estimation, typ):
    if False:
        print('Hello World!')
    negative_nums_code = f'\n@external\ndef negative_one() -> {typ}:\n    return -1\n\n@external\ndef negative_three() -> {typ}:\n    return -(1+2)\n\n@external\ndef negative_four() -> {typ}:\n    a: {typ} = 2\n    return -(a+2)\n    '
    c = get_contract_with_gas_estimation(negative_nums_code)
    assert c.negative_one() == -1
    assert c.negative_three() == -3
    assert c.negative_four() == -4

@pytest.mark.parametrize('typ', types)
def test_num_bound(assert_tx_failed, get_contract_with_gas_estimation, typ):
    if False:
        i = 10
        return i + 15
    (lo, hi) = typ.ast_bounds
    num_bound_code = f'\n@external\ndef _num(x: {typ}) -> {typ}:\n    return x\n\n@external\ndef _num_add(x: {typ}, y: {typ}) -> {typ}:\n    return x + y\n\n@external\ndef _num_sub(x: {typ}, y: {typ}) -> {typ}:\n    return x - y\n\n@external\ndef _num_add3(x: {typ}, y: {typ}, z: {typ}) -> {typ}:\n    return x + y + z\n\n@external\ndef _num_max() -> {typ}:\n    return {hi}\n\n@external\ndef _num_min() -> {typ}:\n    return {lo}\n    '
    c = get_contract_with_gas_estimation(num_bound_code)
    assert c._num_add(hi, 0) == hi
    assert c._num_sub(lo, 0) == lo
    assert c._num_add(hi - 1, 1) == hi
    assert c._num_sub(lo + 1, 1) == lo
    assert_tx_failed(lambda : c._num_add(hi, 1))
    assert_tx_failed(lambda : c._num_sub(lo, 1))
    assert_tx_failed(lambda : c._num_add(hi - 1, 2))
    assert_tx_failed(lambda : c._num_sub(lo + 1, 2))
    assert c._num_max() == hi
    assert c._num_min() == lo
    assert_tx_failed(lambda : c._num_add3(hi, 1, -1))
    assert c._num_add3(hi, -1, 1) == hi - 1 + 1
    assert_tx_failed(lambda : c._num_add3(lo, -1, 1))
    assert c._num_add3(lo, 1, -1) == lo + 1 - 1

@pytest.mark.parametrize('typ', types)
def test_overflow_out_of_range(get_contract, assert_compile_failed, typ):
    if False:
        return 10
    code = f'\n@external\ndef num_sub() -> {typ}:\n    return 1-2**{typ.bits}\n    '
    if typ.bits == 256:
        assert_compile_failed(lambda : get_contract(code), OverflowException)
    else:
        assert_compile_failed(lambda : get_contract(code), InvalidType)
ARITHMETIC_OPS = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': evm_div, '%': evm_mod}

@pytest.mark.parametrize('op', sorted(ARITHMETIC_OPS.keys()))
@pytest.mark.parametrize('typ', types)
@pytest.mark.fuzzing
def test_arithmetic_thorough(get_contract, assert_tx_failed, assert_compile_failed, op, typ):
    if False:
        i = 10
        return i + 15
    code_1 = f'\n@external\ndef foo(x: {typ}, y: {typ}) -> {typ}:\n    return x {op} y\n    '
    code_2_template = '\n@external\ndef foo(x: {typ}) -> {typ}:\n    return x {op} {y}\n    '
    code_3_template = '\n@external\ndef foo(y: {typ}) -> {typ}:\n    return {x} {op} y\n    '
    code_4_template = '\n@external\ndef foo() -> {typ}:\n    return {x} {op} {y}\n    '
    (lo, hi) = typ.ast_bounds
    fns = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': evm_div, '%': evm_mod}
    fn = fns[op]
    c = get_contract(code_1)
    special_cases = [lo, lo + 1, lo // 2, lo // 2 - 1, lo // 2 + 1, -3, -2, -1, 0, 1, 2, 3, hi // 2 - 1, hi // 2, hi // 2 + 1, hi - 1, hi]
    xs = special_cases.copy()
    ys = special_cases.copy()
    NUM_CASES = 5
    xs += [random.randrange(lo, hi) for _ in range(NUM_CASES)]
    ys += [random.randrange(lo, hi) for _ in range(NUM_CASES)]
    assert lo in xs and -1 in ys
    for (x, y) in itertools.product(xs, ys):
        expected = fn(x, y)
        in_bounds = lo <= expected <= hi
        div_by_zero = y == 0 and op in ('/', '%')
        ok = in_bounds and (not div_by_zero)
        code_2 = code_2_template.format(typ=typ, op=op, y=y)
        code_3 = code_3_template.format(typ=typ, op=op, x=x)
        code_4 = code_4_template.format(typ=typ, op=op, x=x, y=y)
        if ok:
            assert c.foo(x, y) == expected
            assert get_contract(code_2).foo(x) == expected
            assert get_contract(code_3).foo(y) == expected
            assert get_contract(code_4).foo() == expected
        elif div_by_zero:
            assert_tx_failed(lambda : c.foo(x, y))
            assert_compile_failed(lambda : get_contract(code_2), ZeroDivisionException)
            assert_tx_failed(lambda : get_contract(code_3).foo(y))
            assert_compile_failed(lambda : get_contract(code_4), ZeroDivisionException)
        else:
            assert_tx_failed(lambda : c.foo(x, y))
            assert_tx_failed(lambda : get_contract(code_2).foo(x))
            assert_tx_failed(lambda : get_contract(code_3).foo(y))
            assert_compile_failed(lambda : get_contract(code_4), (InvalidType, OverflowException))
COMPARISON_OPS = {'==': operator.eq, '!=': operator.ne, '>': operator.gt, '>=': operator.ge, '<': operator.lt, '<=': operator.le}

@pytest.mark.parametrize('op', sorted(COMPARISON_OPS.keys()))
@pytest.mark.parametrize('typ', types)
@pytest.mark.fuzzing
def test_comparators(get_contract, op, typ):
    if False:
        print('Hello World!')
    code_1 = f'\n@external\ndef foo(x: {typ}, y: {typ}) -> bool:\n    return x {op} y\n    '
    (lo, hi) = typ.ast_bounds
    fn = COMPARISON_OPS[op]
    c = get_contract(code_1)
    special_cases = [lo, lo + 1, lo // 2, lo // 2 - 1, lo // 2 + 1, -3, -2, -1, 0, 1, 2, 3, hi // 2 - 1, hi // 2, hi // 2 + 1, hi - 1, hi]
    xs = special_cases.copy()
    ys = special_cases.copy()
    for (x, y) in itertools.product(xs, ys):
        expected = fn(x, y)
        assert c.foo(x, y) is expected

@pytest.mark.parametrize('typ', types)
def test_negation(get_contract, assert_tx_failed, typ):
    if False:
        for i in range(10):
            print('nop')
    code = f'\n@external\ndef foo(a: {typ}) -> {typ}:\n    return -a\n    '
    (lo, hi) = typ.ast_bounds
    c = get_contract(code)
    assert c.foo(hi) == lo + 1
    assert c.foo(-1) == 1
    assert c.foo(1) == -1
    assert c.foo(0) == 0
    assert c.foo(2) == -2
    assert c.foo(-2) == 2
    assert_tx_failed(lambda : c.foo(lo))

@pytest.mark.parametrize('typ', types)
@pytest.mark.parametrize('op', ['not'])
def test_invalid_unary_ops(get_contract, assert_compile_failed, typ, op):
    if False:
        for i in range(10):
            print('nop')
    code = f'\n@external\ndef foo(a: {typ}) -> {typ}:\n    return {op} a\n    '
    assert_compile_failed(lambda : get_contract(code), InvalidOperation)