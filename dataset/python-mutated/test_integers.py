from random import Random
from hypothesis import HealthCheck, Phase, Verbosity, assume, example, given, reject, settings, strategies as st
from hypothesis.internal.conjecture.data import ConjectureData, Status, StopTest
from hypothesis.internal.conjecture.engine import ConjectureRunner
from hypothesis.internal.conjecture.utils import INT_SIZES

@st.composite
def problems(draw):
    if False:
        return 10
    while True:
        buf = bytearray(draw(st.binary(min_size=16, max_size=16)))
        while buf and (not buf[-1]):
            buf.pop()
        try:
            d = ConjectureData.for_buffer(buf)
            k = d.draw(st.integers())
            stop = d.draw_bits(8)
        except (StopTest, IndexError):
            pass
        else:
            if stop > 0 and k > 0:
                return (draw(st.integers(0, k - 1)), bytes(d.buffer))

@example((2, b'\x00\x00\n\x01'))
@example((1, b'\x00\x00\x06\x01'))
@example(problem=(32768, b'\x03\x01\x00\x00\x00\x00\x00\x01\x00\x02\x01'))
@settings(suppress_health_check=list(HealthCheck), deadline=None, max_examples=10, verbosity=Verbosity.normal)
@given(problems())
def test_always_reduces_integers_to_smallest_suitable_sizes(problem):
    if False:
        return 10
    (n, blob) = problem
    try:
        d = ConjectureData.for_buffer(blob)
        k = d.draw(st.integers())
        stop = blob[len(d.buffer)]
    except (StopTest, IndexError):
        reject()
    assume(k > n)
    assume(stop > 0)

    def f(data):
        if False:
            i = 10
            return i + 15
        k = data.draw(st.integers())
        data.output = repr(k)
        if data.draw_bits(8) == stop and k >= n:
            data.mark_interesting()
    runner = ConjectureRunner(f, random=Random(0), settings=settings(suppress_health_check=list(HealthCheck), phases=(Phase.shrink,), database=None, verbosity=Verbosity.debug), database_key=None)
    runner.cached_test_function(blob)
    assert runner.interesting_examples
    (v,) = runner.interesting_examples.values()
    shrinker = runner.new_shrinker(v, lambda x: x.status == Status.INTERESTING)
    shrinker.fixate_shrink_passes(['minimize_individual_blocks'])
    v = shrinker.shrink_target
    m = ConjectureData.for_buffer(v.buffer).draw(st.integers())
    assert m == n
    bits_needed = 1 + n.bit_length()
    actual_bits_needed = min((s for s in INT_SIZES if s >= bits_needed))
    bytes_needed = actual_bits_needed // 8
    assert len(v.buffer) == 3 + bytes_needed

def test_generates_boundary_values_even_when_unlikely():
    if False:
        for i in range(10):
            print('nop')
    r = Random()
    trillion = 10 ** 12
    strat = st.integers(-trillion, trillion)
    boundary_vals = {-trillion, -trillion + 1, trillion - 1, trillion}
    for _ in range(10000):
        buffer = bytes((r.randrange(0, 255) for _ in range(1000)))
        val = ConjectureData.for_buffer(buffer).draw(strat)
        boundary_vals.discard(val)
        if not boundary_vals:
            break
    else:
        raise AssertionError(f'Expected to see all boundary vals, but still have {boundary_vals}')