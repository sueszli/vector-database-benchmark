import time
import pytest
from pytest import raises
from hypothesis import HealthCheck, Phase, given, settings, strategies as st
from hypothesis.control import assume
from hypothesis.errors import FailedHealthCheck, InvalidArgument
from hypothesis.internal.compat import int_from_bytes
from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.internal.entropy import deterministic_PRNG
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule, run_state_machine_as_test
from hypothesis.strategies._internal.lazy import LazyStrategy
from hypothesis.strategies._internal.strategies import SearchStrategy
from tests.common.utils import no_shrink
HEALTH_CHECK_SETTINGS = settings(max_examples=11, database=None)

def test_slow_generation_fails_a_health_check():
    if False:
        for i in range(10):
            print('nop')

    @HEALTH_CHECK_SETTINGS
    @given(st.integers().map(lambda x: time.sleep(0.2)))
    def test(x):
        if False:
            for i in range(10):
                print('nop')
        pass
    with raises(FailedHealthCheck):
        test()

def test_slow_generation_inline_fails_a_health_check():
    if False:
        print('Hello World!')

    @HEALTH_CHECK_SETTINGS
    @given(st.data())
    def test(data):
        if False:
            print('Hello World!')
        data.draw(st.integers().map(lambda x: time.sleep(0.2)))
    with raises(FailedHealthCheck):
        test()

def test_default_health_check_can_weaken_specific():
    if False:
        i = 10
        return i + 15
    import random

    @settings(HEALTH_CHECK_SETTINGS, suppress_health_check=list(HealthCheck))
    @given(st.lists(st.integers(), min_size=1))
    def test(x):
        if False:
            return 10
        random.choice(x)
    test()

def test_suppressing_filtering_health_check():
    if False:
        for i in range(10):
            print('nop')
    forbidden = set()

    def unhealthy_filter(x):
        if False:
            for i in range(10):
                print('nop')
        if len(forbidden) < 200:
            forbidden.add(x)
        return x not in forbidden

    @HEALTH_CHECK_SETTINGS
    @given(st.integers().filter(unhealthy_filter))
    def test1(x):
        if False:
            return 10
        raise ValueError
    with raises(FailedHealthCheck):
        test1()
    forbidden = set()

    @settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
    @given(st.integers().filter(unhealthy_filter))
    def test2(x):
        if False:
            return 10
        raise ValueError
    with raises(ValueError):
        test2()

def test_filtering_everything_fails_a_health_check():
    if False:
        while True:
            i = 10

    @given(st.integers().filter(lambda x: False))
    @settings(database=None)
    def test(x):
        if False:
            for i in range(10):
                print('nop')
        pass
    with raises(FailedHealthCheck) as e:
        test()
    assert 'filter' in e.value.args[0]

class fails_regularly(SearchStrategy):

    def do_draw(self, data):
        if False:
            while True:
                i = 10
        b = int_from_bytes(data.draw_bytes(2))
        assume(b == 3)
        print('ohai')

def test_filtering_most_things_fails_a_health_check():
    if False:
        for i in range(10):
            print('nop')

    @given(fails_regularly())
    @settings(database=None, phases=no_shrink)
    def test(x):
        if False:
            return 10
        pass
    with raises(FailedHealthCheck) as e:
        test()
    assert 'filter' in e.value.args[0]

def test_large_data_will_fail_a_health_check():
    if False:
        return 10

    @given(st.none() | st.binary(min_size=10 ** 5, max_size=10 ** 5))
    @settings(database=None)
    def test(x):
        if False:
            for i in range(10):
                print('nop')
        pass
    with raises(FailedHealthCheck) as e:
        test()
    assert 'allowable size' in e.value.args[0]

def test_returning_non_none_is_forbidden():
    if False:
        i = 10
        return i + 15

    @given(st.integers())
    def a(x):
        if False:
            return 10
        return 1
    with raises(FailedHealthCheck):
        a()

def test_the_slow_test_health_check_can_be_disabled():
    if False:
        while True:
            i = 10

    @given(st.integers())
    @settings(deadline=None)
    def a(x):
        if False:
            while True:
                i = 10
        time.sleep(1000)
    a()

def test_the_slow_test_health_only_runs_if_health_checks_are_on():
    if False:
        i = 10
        return i + 15

    @given(st.integers())
    @settings(suppress_health_check=list(HealthCheck), deadline=None)
    def a(x):
        if False:
            return 10
        time.sleep(1000)
    a()

def test_large_base_example_fails_health_check():
    if False:
        while True:
            i = 10

    @given(st.binary(min_size=7000, max_size=7000))
    def test(b):
        if False:
            return 10
        pass
    with pytest.raises(FailedHealthCheck) as exc:
        test()
    assert str(HealthCheck.large_base_example) in str(exc.value)

def test_example_that_shrinks_to_overrun_fails_health_check():
    if False:
        return 10

    @given(st.binary(min_size=9000, max_size=9000) | st.none())
    def test(b):
        if False:
            while True:
                i = 10
        pass
    with pytest.raises(FailedHealthCheck) as exc:
        test()
    assert str(HealthCheck.large_base_example) in str(exc.value)

class sample_test_runner:

    @given(st.none())
    def test(self, _):
        if False:
            while True:
                i = 10
        pass

def test_differing_executors_fails_health_check():
    if False:
        return 10
    sample_test_runner().test()
    with pytest.raises(FailedHealthCheck) as exc:
        sample_test_runner().test()
    assert str(HealthCheck.differing_executors) in str(exc.value)

def test_it_is_an_error_to_suppress_non_iterables():
    if False:
        return 10
    with raises(InvalidArgument):
        settings(suppress_health_check=1)

def test_it_is_an_error_to_suppress_non_healthchecks():
    if False:
        i = 10
        return i + 15
    with raises(InvalidArgument):
        settings(suppress_health_check=[1])
slow_down_init = True

def slow_init_integers(*args, **kwargs):
    if False:
        while True:
            i = 10
    global slow_down_init
    if slow_down_init:
        time.sleep(0.5)
        slow_down_init = False
    return st.integers(*args, **kwargs)

@given(st.data())
def test_lazy_slow_initialization_issue_2108_regression(data):
    if False:
        i = 10
        return i + 15
    data.draw(LazyStrategy(slow_init_integers, (), {}))

def test_does_not_trigger_health_check_on_simple_strategies(monkeypatch):
    if False:
        return 10
    existing_draw_bits = ConjectureData.draw_bits

    def draw_bits(self, n, forced=None):
        if False:
            print('Hello World!')
        time.sleep(0.001)
        return existing_draw_bits(self, n, forced=forced)
    monkeypatch.setattr(ConjectureData, 'draw_bits', draw_bits)
    with deterministic_PRNG():
        for _ in range(100):

            @settings(database=None, max_examples=11, phases=[Phase.generate])
            @given(st.binary())
            def test(b):
                if False:
                    for i in range(10):
                        print('nop')
                pass
            test()

def test_does_not_trigger_health_check_when_most_examples_are_small(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    with deterministic_PRNG():
        for _ in range(100):

            @settings(database=None, max_examples=11, phases=[Phase.generate])
            @given(st.integers(0, 100).flatmap(lambda n: st.binary(min_size=n * 100, max_size=n * 100)))
            def test(b):
                if False:
                    while True:
                        i = 10
                pass
            test()

class ReturningRuleMachine(RuleBasedStateMachine):

    @rule()
    def r(self):
        if False:
            i = 10
            return i + 15
        return 'any non-None value'

class ReturningInitializeMachine(RuleBasedStateMachine):
    _ = rule()(lambda self: None)

    @initialize()
    def r(self):
        if False:
            i = 10
            return i + 15
        return 'any non-None value'

class ReturningInvariantMachine(RuleBasedStateMachine):
    _ = rule()(lambda self: None)

    @invariant(check_during_init=True)
    def r(self):
        if False:
            i = 10
            return i + 15
        return 'any non-None value'

@pytest.mark.parametrize('cls', [ReturningRuleMachine, ReturningInitializeMachine, ReturningInvariantMachine])
def test_stateful_returnvalue_healthcheck(cls):
    if False:
        print('Hello World!')
    with pytest.raises(FailedHealthCheck):
        run_state_machine_as_test(cls, settings=settings())