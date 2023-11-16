from hypothesis import given, settings, strategies as st
from hypothesis.database import InMemoryExampleDatabase
from hypothesis.internal.compat import int_from_bytes
from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.internal.conjecture.engine import ConjectureRunner
from hypothesis.internal.conjecture.shrinker import Shrinker, block_program
from tests.common.utils import counts_calls, non_covering_examples
from tests.conjecture.common import run_to_buffer, shrinking_from

def test_lot_of_dead_nodes():
    if False:
        print('Hello World!')

    @run_to_buffer
    def x(data):
        if False:
            while True:
                i = 10
        for i in range(4):
            if data.draw_bytes(1)[0] != i:
                data.mark_invalid()
        data.mark_interesting()
    assert x == bytes([0, 1, 2, 3])

def test_saves_data_while_shrinking(monkeypatch):
    if False:
        while True:
            i = 10
    key = b'hi there'
    n = 5
    db = InMemoryExampleDatabase()
    assert list(db.fetch(key)) == []
    seen = set()
    monkeypatch.setattr(ConjectureRunner, 'generate_new_examples', lambda runner: runner.cached_test_function([255] * 10))

    def f(data):
        if False:
            while True:
                i = 10
        x = data.draw_bytes(10)
        if sum(x) >= 2000 and len(seen) < n:
            seen.add(x)
        if x in seen:
            data.mark_interesting()
    runner = ConjectureRunner(f, settings=settings(database=db), database_key=key)
    runner.run()
    assert runner.interesting_examples
    assert len(seen) == n
    in_db = non_covering_examples(db)
    assert in_db.issubset(seen)
    assert in_db == seen

def test_can_discard(monkeypatch):
    if False:
        i = 10
        return i + 15
    n = 8
    monkeypatch.setattr(ConjectureRunner, 'generate_new_examples', lambda runner: runner.cached_test_function([v for i in range(n) for v in [i, i]]))

    @run_to_buffer
    def x(data):
        if False:
            for i in range(10):
                print('nop')
        seen = set()
        while len(seen) < n:
            seen.add(bytes(data.draw_bytes(1)))
        data.mark_interesting()
    assert len(x) == n

def test_regression_1():
    if False:
        return 10

    @run_to_buffer
    def x(data):
        if False:
            return 10
        data.write(b'\x01\x02')
        data.write(b'\x01\x00')
        v = data.draw_bits(41)
        if v >= 512 or v == 254:
            data.mark_interesting()
    assert list(x)[:-2] == [1, 2, 1, 0, 0, 0, 0, 0]
    assert int_from_bytes(x[-2:]) in (254, 512)

@given(st.integers(0, 255), st.integers(0, 255))
def test_cached_with_masked_byte_agrees_with_results(byte_a, byte_b):
    if False:
        for i in range(10):
            print('nop')

    def f(data):
        if False:
            for i in range(10):
                print('nop')
        data.draw_bits(2)
    runner = ConjectureRunner(f)
    cached_a = runner.cached_test_function(bytes([byte_a]))
    cached_b = runner.cached_test_function(bytes([byte_b]))
    data_b = ConjectureData.for_buffer(bytes([byte_b]), observer=runner.tree.new_observer())
    runner.test_function(data_b)
    assert (cached_a is cached_b) == (cached_a.buffer == data_b.buffer)

def test_block_programs_fail_efficiently(monkeypatch):
    if False:
        i = 10
        return i + 15

    @shrinking_from(bytes(range(256)))
    def shrinker(data):
        if False:
            while True:
                i = 10
        values = set()
        for _ in range(256):
            v = data.draw_bits(8)
            values.add(v)
        if len(values) == 256:
            data.mark_interesting()
    monkeypatch.setattr(Shrinker, 'run_block_program', counts_calls(Shrinker.run_block_program))
    shrinker.max_stall = 500
    shrinker.fixate_shrink_passes([block_program('XX')])
    assert shrinker.shrinks == 0
    assert 250 <= shrinker.calls <= 260
    assert 250 <= Shrinker.run_block_program.calls <= 260