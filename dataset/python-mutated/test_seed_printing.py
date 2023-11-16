import time
import pytest
from hypothesis import Verbosity, assume, core, given, settings, strategies as st
from hypothesis.database import InMemoryExampleDatabase
from hypothesis.errors import FailedHealthCheck
from tests.common.utils import all_values, capture_out

@pytest.mark.parametrize('in_pytest', [False, True])
@pytest.mark.parametrize('fail_healthcheck', [False, True])
@pytest.mark.parametrize('verbosity', [Verbosity.normal, Verbosity.quiet])
def test_prints_seed_only_on_healthcheck(monkeypatch, in_pytest, fail_healthcheck, verbosity):
    if False:
        return 10
    monkeypatch.setattr(core, 'running_under_pytest', in_pytest)
    strategy = st.integers()
    if fail_healthcheck:

        def slow_map(i):
            if False:
                print('Hello World!')
            time.sleep(10)
            return i
        strategy = strategy.map(slow_map)
        expected_exc = FailedHealthCheck
    else:
        expected_exc = AssertionError

    @settings(database=None, verbosity=verbosity)
    @given(strategy)
    def test(i):
        if False:
            print('Hello World!')
        assert fail_healthcheck
    with capture_out() as o:
        with pytest.raises(expected_exc):
            test()
    output = o.getvalue()
    seed = test._hypothesis_internal_use_generated_seed
    assert seed is not None
    if fail_healthcheck and verbosity != Verbosity.quiet:
        assert f'@seed({seed})' in output
        contains_pytest_instruction = f'--hypothesis-seed={seed}' in output
        assert contains_pytest_instruction == in_pytest
    else:
        assert '@seed' not in output
        assert f'--hypothesis-seed={seed}' not in output

def test_uses_global_force(monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.setattr(core, 'global_force_seed', 42)

    @given(st.integers())
    def test(i):
        if False:
            for i in range(10):
                print('nop')
        raise ValueError
    output = []
    for _ in range(2):
        with capture_out() as o:
            with pytest.raises(ValueError):
                test()
        output.append(o.getvalue())
    assert output[0] == output[1]
    assert '@seed' not in output[0]

def test_does_print_on_reuse_from_database():
    if False:
        print('Hello World!')
    passes_healthcheck = False
    database = InMemoryExampleDatabase()

    @settings(database=database)
    @given(st.integers())
    def test(i):
        if False:
            print('Hello World!')
        assume(passes_healthcheck)
        raise ValueError
    with capture_out() as o:
        with pytest.raises(FailedHealthCheck):
            test()
    assert '@seed' in o.getvalue()
    passes_healthcheck = True
    with capture_out() as o:
        with pytest.raises(ValueError):
            test()
    assert all_values(database)
    assert '@seed' not in o.getvalue()
    passes_healthcheck = False
    with capture_out() as o:
        with pytest.raises(FailedHealthCheck):
            test()
    assert '@seed' in o.getvalue()