import sys
import pytest
from hypothesis import HealthCheck, assume, example, given, settings, strategies as st
from hypothesis.internal.compat import ceil, floor, int_from_bytes, int_to_bytes
from hypothesis.internal.conjecture import floats as flt
from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.internal.conjecture.engine import ConjectureRunner
from hypothesis.internal.floats import float_to_int
EXPONENTS = list(range(flt.MAX_EXPONENT + 1))
assert len(EXPONENTS) == 2 ** 11

def assert_reordered_exponents(res):
    if False:
        return 10
    res = list(res)
    assert len(res) == len(EXPONENTS)
    for x in res:
        assert res.count(x) == 1
        assert 0 <= x <= flt.MAX_EXPONENT

def test_encode_permutes_elements():
    if False:
        while True:
            i = 10
    assert_reordered_exponents(map(flt.encode_exponent, EXPONENTS))

def test_decode_permutes_elements():
    if False:
        i = 10
        return i + 15
    assert_reordered_exponents(map(flt.decode_exponent, EXPONENTS))

def test_decode_encode():
    if False:
        return 10
    for e in EXPONENTS:
        assert flt.decode_exponent(flt.encode_exponent(e)) == e

def test_encode_decode():
    if False:
        i = 10
        return i + 15
    for e in EXPONENTS:
        assert flt.decode_exponent(flt.encode_exponent(e)) == e

@given(st.data())
def test_double_reverse_bounded(data):
    if False:
        while True:
            i = 10
    n = data.draw(st.integers(1, 64))
    i = data.draw(st.integers(0, 2 ** n - 1))
    j = flt.reverse_bits(i, n)
    assert flt.reverse_bits(j, n) == i

@given(st.integers(0, 2 ** 64 - 1))
def test_double_reverse(i):
    if False:
        return 10
    j = flt.reverse64(i)
    assert flt.reverse64(j) == i

@example(1.25)
@example(1.0)
@given(st.floats())
def test_draw_write_round_trip(f):
    if False:
        while True:
            i = 10
    d = ConjectureData.for_buffer(bytes(10))
    flt.write_float(d, f)
    d2 = ConjectureData.for_buffer(d.buffer)
    g = flt.draw_float(d2)
    if f == f:
        assert f == g
    assert float_to_int(f) == float_to_int(g)
    d3 = ConjectureData.for_buffer(d2.buffer)
    flt.draw_float(d3)
    assert d3.buffer == d2.buffer

@example(0.0)
@example(2.5)
@example(8.000000000000007)
@example(3.0)
@example(2.0)
@example(1.9999999999999998)
@example(1.0)
@given(st.floats(min_value=0.0))
def test_floats_round_trip(f):
    if False:
        for i in range(10):
            print('nop')
    i = flt.float_to_lex(f)
    g = flt.lex_to_float(i)
    assert float_to_int(f) == float_to_int(g)

@settings(suppress_health_check=[HealthCheck.too_slow])
@example(1, 0.5)
@given(st.integers(1, 2 ** 53), st.floats(0, 1).filter(lambda x: x not in (0, 1)))
def test_floats_order_worse_than_their_integral_part(n, g):
    if False:
        i = 10
        return i + 15
    f = n + g
    assume(int(f) != f)
    assume(int(f) != 0)
    i = flt.float_to_lex(f)
    if f < 0:
        g = ceil(f)
    else:
        g = floor(f)
    assert flt.float_to_lex(float(g)) < i
integral_floats = st.floats(allow_infinity=False, allow_nan=False, min_value=0.0).map(lambda x: abs(float(int(x))))

@given(integral_floats, integral_floats)
def test_integral_floats_order_as_integers(x, y):
    if False:
        for i in range(10):
            print('nop')
    assume(x != y)
    (x, y) = sorted((x, y))
    assert flt.float_to_lex(x) < flt.float_to_lex(y)

@given(st.floats(0, 1))
def test_fractional_floats_are_worse_than_one(f):
    if False:
        print('Hello World!')
    assume(0 < f < 1)
    assert flt.float_to_lex(f) > flt.float_to_lex(1)

def test_reverse_bits_table_reverses_bits():
    if False:
        return 10

    def bits(x):
        if False:
            print('Hello World!')
        result = []
        for _ in range(8):
            result.append(x & 1)
            x >>= 1
        result.reverse()
        return result
    for (i, b) in enumerate(flt.REVERSE_BITS_TABLE):
        assert bits(i) == list(reversed(bits(b)))

def test_reverse_bits_table_has_right_elements():
    if False:
        print('Hello World!')
    assert sorted(flt.REVERSE_BITS_TABLE) == list(range(256))

def float_runner(start, condition):
    if False:
        while True:
            i = 10

    def parse_buf(b):
        if False:
            print('Hello World!')
        return flt.lex_to_float(int_from_bytes(b))

    def test_function(data):
        if False:
            while True:
                i = 10
        f = flt.draw_float(data)
        if condition(f):
            data.mark_interesting()
    runner = ConjectureRunner(test_function)
    runner.cached_test_function(bytes(1) + int_to_bytes(flt.float_to_lex(start), 8))
    assert runner.interesting_examples
    return runner

def minimal_from(start, condition):
    if False:
        return 10
    runner = float_runner(start, condition)
    runner.shrink_interesting_examples()
    (v,) = runner.interesting_examples.values()
    result = flt.draw_float(ConjectureData.for_buffer(v.buffer))
    assert condition(result)
    return result
INTERESTING_FLOATS = [0.0, 1.0, 2.0, sys.float_info.max, float('inf'), float('nan')]

@pytest.mark.parametrize(('start', 'end'), [(a, b) for a in INTERESTING_FLOATS for b in INTERESTING_FLOATS if flt.float_to_lex(a) > flt.float_to_lex(b)])
def test_can_shrink_downwards(start, end):
    if False:
        i = 10
        return i + 15
    assert minimal_from(start, lambda x: not x < end) == end

@pytest.mark.parametrize('f', [1, 2, 4, 8, 10, 16, 32, 64, 100, 128, 256, 500, 512, 1000, 1024])
@pytest.mark.parametrize('mul', [1.1, 1.5, 9.99, 10])
def test_shrinks_downwards_to_integers(f, mul):
    if False:
        i = 10
        return i + 15
    g = minimal_from(f * mul, lambda x: x >= f)
    assert g == f

def test_shrink_to_integer_upper_bound():
    if False:
        print('Hello World!')
    assert minimal_from(1.1, lambda x: 1 < x <= 2) == 2

def test_shrink_up_to_one():
    if False:
        while True:
            i = 10
    assert minimal_from(0.5, lambda x: 0.5 <= x <= 1.5) == 1

def test_shrink_down_to_half():
    if False:
        return 10
    assert minimal_from(0.75, lambda x: 0 < x < 1) == 0.5

def test_shrink_fractional_part():
    if False:
        i = 10
        return i + 15
    assert minimal_from(2.5, lambda x: divmod(x, 1)[1] == 0.5) == 1.5

def test_does_not_shrink_across_one():
    if False:
        while True:
            i = 10
    assert minimal_from(1.1, lambda x: x == 1.1 or 0 < x < 1) == 1.1

@pytest.mark.parametrize('f', [2.0, 10000000.0])
def test_converts_floats_to_integer_form(f):
    if False:
        for i in range(10):
            print('nop')
    assert flt.is_simple(f)
    buf = int_to_bytes(flt.base_float_to_lex(f), 8)
    runner = float_runner(f, lambda g: g == f)
    runner.shrink_interesting_examples()
    (v,) = runner.interesting_examples.values()
    assert v.buffer[:-1] < buf