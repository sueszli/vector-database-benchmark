import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from vyper.codegen.arithmetic import calculate_largest_base, calculate_largest_power

@pytest.mark.fuzzing
@pytest.mark.parametrize('power', range(2, 255))
def test_exp_uint256(get_contract, assert_tx_failed, power):
    if False:
        for i in range(10):
            print('nop')
    code = f'\n@external\ndef foo(a: uint256) -> uint256:\n    return a ** {power}\n    '
    (_min_base, max_base) = calculate_largest_base(power, 256, False)
    assert max_base ** power < 2 ** 256
    assert (max_base + 1) ** power >= 2 ** 256
    c = get_contract(code)
    c.foo(max_base)
    assert_tx_failed(lambda : c.foo(max_base + 1))

@pytest.mark.fuzzing
@pytest.mark.parametrize('power', range(2, 127))
def test_exp_int128(get_contract, assert_tx_failed, power):
    if False:
        while True:
            i = 10
    code = f'\n@external\ndef foo(a: int128) -> int128:\n    return a ** {power}\n    '
    (min_base, max_base) = calculate_largest_base(power, 128, True)
    assert -2 ** 127 <= max_base ** power < 2 ** 127
    assert -2 ** 127 <= min_base ** power < 2 ** 127
    assert not -2 ** 127 <= (max_base + 1) ** power < 2 ** 127
    assert not -2 ** 127 <= (min_base - 1) ** power < 2 ** 127
    c = get_contract(code)
    c.foo(max_base)
    c.foo(min_base)
    assert_tx_failed(lambda : c.foo(max_base + 1))
    assert_tx_failed(lambda : c.foo(min_base - 1))

@pytest.mark.fuzzing
@pytest.mark.parametrize('power', range(2, 15))
def test_exp_int16(get_contract, assert_tx_failed, power):
    if False:
        i = 10
        return i + 15
    code = f'\n@external\ndef foo(a: int16) -> int16:\n    return a ** {power}\n    '
    (min_base, max_base) = calculate_largest_base(power, 16, True)
    assert -2 ** 15 <= max_base ** power < 2 ** 15
    assert -2 ** 15 <= min_base ** power < 2 ** 15
    assert not -2 ** 15 <= (max_base + 1) ** power < 2 ** 15
    assert not -2 ** 15 <= (min_base - 1) ** power < 2 ** 15
    c = get_contract(code)
    c.foo(max_base)
    c.foo(min_base)
    assert_tx_failed(lambda : c.foo(max_base + 1))
    assert_tx_failed(lambda : c.foo(min_base - 1))

@pytest.mark.fuzzing
@given(a=st.integers(min_value=2, max_value=2 ** 256 - 1))
@example(a=2 ** 7)
@example(a=2 ** 7 - 1)
@example(a=2 ** 15)
@example(a=2 ** 15 - 1)
@example(a=2 ** 31)
@example(a=2 ** 31 - 1)
@example(a=2 ** 63)
@example(a=2 ** 63 - 1)
@example(a=2 ** 127)
@example(a=2 ** 127 - 1)
@example(a=2 ** 256 - 1)
@settings(max_examples=200)
def test_max_exp(get_contract, assert_tx_failed, a):
    if False:
        print('Hello World!')
    code = f'\n@external\ndef foo(b: uint256) -> uint256:\n    return {a} ** b\n    '
    c = get_contract(code)
    max_power = calculate_largest_power(a, 256, False)
    assert a ** max_power < 2 ** 256
    assert a ** (max_power + 1) >= 2 ** 256
    c.foo(max_power)
    assert_tx_failed(lambda : c.foo(max_power + 1))

@pytest.mark.fuzzing
@given(a=st.integers(min_value=2, max_value=2 ** 127 - 1))
@example(a=2 ** 7)
@example(a=2 ** 7 - 1)
@example(a=2 ** 15)
@example(a=2 ** 15 - 1)
@example(a=2 ** 31)
@example(a=2 ** 31 - 1)
@example(a=2 ** 63)
@example(a=2 ** 63 - 1)
@example(a=2 ** 127 - 1)
@settings(max_examples=200)
def test_max_exp_int128(get_contract, assert_tx_failed, a):
    if False:
        while True:
            i = 10
    code = f'\n@external\ndef foo(b: int128) -> int128:\n    return {a} ** b\n    '
    c = get_contract(code)
    max_power = calculate_largest_power(a, 128, True)
    assert -2 ** 127 <= a ** max_power < 2 ** 127
    assert not -2 ** 127 <= a ** (max_power + 1) < 2 ** 127
    c.foo(max_power)
    assert_tx_failed(lambda : c.foo(max_power + 1))