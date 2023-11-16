import pytest
from hypothesis import HealthCheck, Verbosity, assume, example, given, reject, settings
from hypothesis.errors import Flaky, Unsatisfiable, UnsatisfiedAssumption
from hypothesis.internal.conjecture.engine import MIN_TEST_CALLS
from hypothesis.strategies import booleans, composite, integers, lists, random_module
from tests.common.utils import no_shrink

class Nope(Exception):
    pass

def test_fails_only_once_is_flaky():
    if False:
        return 10
    first_call = [True]

    @given(integers())
    def rude(x):
        if False:
            while True:
                i = 10
        if first_call[0]:
            first_call[0] = False
            raise Nope
    with pytest.raises(Flaky):
        rude()

def test_gives_flaky_error_if_assumption_is_flaky():
    if False:
        return 10
    seen = set()

    @given(integers())
    @settings(verbosity=Verbosity.quiet)
    def oops(s):
        if False:
            return 10
        assume(s not in seen)
        seen.add(s)
        raise AssertionError
    with pytest.raises(Flaky):
        oops()

def test_does_not_attempt_to_shrink_flaky_errors():
    if False:
        print('Hello World!')
    values = []

    @settings(database=None)
    @given(integers())
    def test(x):
        if False:
            return 10
        values.append(x)
        assert len(values) != 1
    with pytest.raises(Flaky):
        test()
    assert 1 < len(set(values)) <= MIN_TEST_CALLS
    assert set(values) == set(values[:-2])

class SatisfyMe(Exception):
    pass

@composite
def single_bool_lists(draw):
    if False:
        while True:
            i = 10
    n = draw(integers(0, 20))
    result = [False] * (n + 1)
    result[n] = True
    return result

@example([True, False, False, False], [3], None)
@example([False, True, False, False], [3], None)
@example([False, False, True, False], [3], None)
@example([False, False, False, True], [3], None)
@settings(deadline=None)
@given(lists(booleans()) | single_bool_lists(), lists(integers(1, 3)), random_module())
def test_failure_sequence_inducing(building, testing, rnd):
    if False:
        return 10
    buildit = iter(building)
    testit = iter(testing)

    def build(x):
        if False:
            print('Hello World!')
        try:
            assume(not next(buildit))
        except StopIteration:
            pass
        return x

    @given(integers().map(build))
    @settings(verbosity=Verbosity.quiet, database=None, suppress_health_check=list(HealthCheck), phases=no_shrink)
    def test(x):
        if False:
            for i in range(10):
                print('nop')
        try:
            i = next(testit)
        except StopIteration:
            return
        if i == 1:
            return
        elif i == 2:
            reject()
        else:
            raise Nope
    try:
        test()
    except (Nope, Unsatisfiable, Flaky):
        pass
    except UnsatisfiedAssumption:
        raise SatisfyMe from None