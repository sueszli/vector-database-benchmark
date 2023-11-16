import sys
from fractions import Fraction
from hypothesis import assume, example, given, strategies as st, target
from hypothesis.internal.compat import int_to_bytes
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.conjecture.data import ConjectureData, StopTest
from hypothesis.internal.conjecture.engine import BUFFER_SIZE

def test_gives_the_correct_probabilities():
    if False:
        print('Hello World!')
    weights = [Fraction(1), Fraction(9)]
    total = sum(weights)
    probabilities = [w / total for w in weights]
    sampler = cu.Sampler(probabilities)
    assert cu.Sampler(weights).table == sampler.table
    counts = [0] * len(weights)
    i = 0
    while i < 2 ** 16:
        data = ConjectureData.for_buffer(int_to_bytes(i, 2))
        try:
            c = sampler.sample(data)
            counts[c] += 1
            assert probabilities[c] >= Fraction(counts[c], 2 ** 16)
        except StopTest:
            pass
        if 1 in data.forced_indices:
            i += 256
        else:
            i += 1

@example(0, 1)
@example(0, float('inf'))
@example(cu.SMALLEST_POSITIVE_FLOAT, 2 * cu.SMALLEST_POSITIVE_FLOAT)
@example(cu.SMALLEST_POSITIVE_FLOAT, 1)
@example(cu.SMALLEST_POSITIVE_FLOAT, float('inf'))
@example(sys.float_info.min, 1)
@example(sys.float_info.min, float('inf'))
@example(10, 10)
@example(10, float('inf'))
@given(st.floats(0, BUFFER_SIZE // 2), st.integers(0, BUFFER_SIZE // 2))
def test_p_continue(average_size, max_size):
    if False:
        return 10
    assume(average_size <= max_size)
    p = cu._calc_p_continue(average_size, max_size)
    assert 0 <= target(p, label='p') <= 1
    assert 0 < target(p, label='-p') or average_size < 1e-05
    abs_err = abs(average_size - cu._p_continue_to_avg(p, max_size))
    assert target(abs_err, label='abs_err') < 0.01

@example(1.1, 10)
@given(st.floats(0, 1), st.integers(0, BUFFER_SIZE // 2))
def test_p_continue_to_average(p_continue, max_size):
    if False:
        print('Hello World!')
    average = cu._p_continue_to_avg(p_continue, max_size)
    assert 0 <= average <= max_size