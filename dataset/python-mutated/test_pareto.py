import pytest
from hypothesis import HealthCheck, Phase, settings
from hypothesis.database import InMemoryExampleDatabase
from hypothesis.internal.compat import int_to_bytes
from hypothesis.internal.conjecture.data import Status
from hypothesis.internal.conjecture.engine import ConjectureRunner, RunIsComplete
from hypothesis.internal.entropy import deterministic_PRNG

def test_pareto_front_contains_different_interesting_reasons():
    if False:
        return 10
    with deterministic_PRNG():

        def test(data):
            if False:
                print('Hello World!')
            data.mark_interesting(data.draw_bits(4))
        runner = ConjectureRunner(test, settings=settings(max_examples=5000, database=InMemoryExampleDatabase(), suppress_health_check=list(HealthCheck)), database_key=b'stuff')
        runner.run()
        assert len(runner.pareto_front) == 2 ** 4

def test_database_contains_only_pareto_front():
    if False:
        i = 10
        return i + 15
    with deterministic_PRNG():

        def test(data):
            if False:
                while True:
                    i = 10
            data.target_observations['1'] = data.draw_bits(4)
            data.draw_bits(64)
            data.target_observations['2'] = data.draw_bits(8)
        db = InMemoryExampleDatabase()
        runner = ConjectureRunner(test, settings=settings(max_examples=500, database=db, suppress_health_check=list(HealthCheck)), database_key=b'stuff')
        runner.run()
        assert len(runner.pareto_front) <= 500
        for v in runner.pareto_front:
            assert v.status >= Status.VALID
        assert len(db.data) == 1
        (values,) = db.data.values()
        values = set(values)
        assert len(values) == len(runner.pareto_front)
        for data in runner.pareto_front:
            assert data.buffer in values
            assert data in runner.pareto_front
        for k in values:
            assert runner.cached_test_function(k) in runner.pareto_front

def test_clears_defunct_pareto_front():
    if False:
        return 10
    with deterministic_PRNG():

        def test(data):
            if False:
                i = 10
                return i + 15
            data.draw_bits(8)
            data.draw_bits(8)
        db = InMemoryExampleDatabase()
        runner = ConjectureRunner(test, settings=settings(max_examples=10000, database=db, suppress_health_check=list(HealthCheck), phases=[Phase.reuse]), database_key=b'stuff')
        for i in range(256):
            db.save(runner.pareto_key, bytes([i, 0]))
        runner.run()
        assert len(list(db.fetch(runner.pareto_key))) == 1

def test_down_samples_the_pareto_front():
    if False:
        print('Hello World!')
    with deterministic_PRNG():

        def test(data):
            if False:
                return 10
            data.draw_bits(8)
            data.draw_bits(8)
        db = InMemoryExampleDatabase()
        runner = ConjectureRunner(test, settings=settings(max_examples=1000, database=db, suppress_health_check=list(HealthCheck), phases=[Phase.reuse]), database_key=b'stuff')
        for i in range(10000):
            db.save(runner.pareto_key, int_to_bytes(i, 2))
        with pytest.raises(RunIsComplete):
            runner.reuse_existing_examples()
        assert runner.valid_examples == 1000

def test_stops_loading_pareto_front_if_interesting():
    if False:
        i = 10
        return i + 15
    with deterministic_PRNG():

        def test(data):
            if False:
                return 10
            data.draw_bits(8)
            data.draw_bits(8)
            data.mark_interesting()
        db = InMemoryExampleDatabase()
        runner = ConjectureRunner(test, settings=settings(max_examples=1000, database=db, suppress_health_check=list(HealthCheck), phases=[Phase.reuse]), database_key=b'stuff')
        for i in range(10000):
            db.save(runner.pareto_key, int_to_bytes(i, 2))
        runner.reuse_existing_examples()
        assert runner.call_count == 1

def test_uses_tags_in_calculating_pareto_front():
    if False:
        print('Hello World!')
    with deterministic_PRNG():

        def test(data):
            if False:
                i = 10
                return i + 15
            if data.draw_bits(1):
                data.start_example(11)
                data.draw_bits(8)
                data.stop_example()
        runner = ConjectureRunner(test, settings=settings(max_examples=10, database=InMemoryExampleDatabase()), database_key=b'stuff')
        runner.run()
        assert len(runner.pareto_front) == 2

def test_optimises_the_pareto_front():
    if False:
        while True:
            i = 10

    def test(data):
        if False:
            while True:
                i = 10
        count = 0
        while data.draw_bits(8):
            count += 1
        data.target_observations[''] = min(count, 5)
    runner = ConjectureRunner(test, settings=settings(max_examples=10000, database=InMemoryExampleDatabase()), database_key=b'stuff')
    runner.cached_test_function([255] * 20 + [0])
    runner.pareto_optimise()
    assert len(runner.pareto_front) == 6
    for (i, data) in enumerate(runner.pareto_front):
        assert list(data.buffer) == [1] * i + [0]

def test_does_not_optimise_the_pareto_front_if_interesting():
    if False:
        for i in range(10):
            print('nop')

    def test(data):
        if False:
            i = 10
            return i + 15
        n = data.draw_bits(8)
        data.target_observations[''] = n
        if n == 255:
            data.mark_interesting()
    runner = ConjectureRunner(test, settings=settings(max_examples=10000, database=InMemoryExampleDatabase()), database_key=b'stuff')
    runner.cached_test_function([0])
    runner.pareto_optimise = None
    runner.optimise_targets()
    assert runner.interesting_examples

def test_stops_optimising_once_interesting():
    if False:
        return 10
    hi = 2 ** 16 - 1

    def test(data):
        if False:
            for i in range(10):
                print('nop')
        n = data.draw_bits(16)
        data.target_observations[''] = n
        if n < hi:
            data.mark_interesting()
    runner = ConjectureRunner(test, settings=settings(max_examples=10000, database=InMemoryExampleDatabase()), database_key=b'stuff')
    data = runner.cached_test_function([255] * 2)
    assert data.status == Status.VALID
    runner.pareto_optimise()
    assert runner.call_count <= 20
    assert runner.interesting_examples