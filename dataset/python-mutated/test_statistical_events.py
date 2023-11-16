import re
import time
import traceback
import pytest
from hypothesis import HealthCheck, assume, event, example, given, settings, stateful, strategies as st, target
from hypothesis.statistics import collector, describe_statistics

def call_for_statistics(test_function):
    if False:
        i = 10
        return i + 15
    result = []
    with collector.with_value(result.append):
        try:
            test_function()
        except Exception:
            traceback.print_exc()
    assert len(result) == 1, result
    return result[0]

def unique_events(stats):
    if False:
        for i in range(10):
            print('nop')
    return set(sum((t['events'] for t in stats['generate-phase']['test-cases']), []))

def test_notes_hard_to_satisfy():
    if False:
        for i in range(10):
            print('nop')

    @given(st.integers())
    @settings(suppress_health_check=list(HealthCheck))
    def test(i):
        if False:
            print('Hello World!')
        assume(i == 13)
    stats = call_for_statistics(test)
    assert 'satisfied assumptions' in stats['stopped-because']

def test_can_callback_with_a_string():
    if False:
        i = 10
        return i + 15

    @given(st.integers())
    def test(i):
        if False:
            while True:
                i = 10
        event('hi')
    stats = call_for_statistics(test)
    assert any(('hi' in s for s in unique_events(stats)))
counter = 0
seen = []

class Foo:

    def __eq__(self, other):
        if False:
            return 10
        return True

    def __ne__(self, other):
        if False:
            return 10
        return False

    def __hash__(self):
        if False:
            print('Hello World!')
        return 0

    def __str__(self):
        if False:
            return 10
        seen.append(self)
        global counter
        counter += 1
        return f'COUNTER {counter}'

def test_formats_are_evaluated_only_once():
    if False:
        for i in range(10):
            print('nop')
    global counter
    counter = 0

    @given(st.integers())
    def test(i):
        if False:
            i = 10
            return i + 15
        event(Foo())
    stats = call_for_statistics(test)
    assert 'COUNTER 1' in unique_events(stats)
    assert 'COUNTER 2' not in unique_events(stats)

def test_does_not_report_on_examples():
    if False:
        return 10

    @example('hi')
    @given(st.integers())
    def test(i):
        if False:
            return 10
        if isinstance(i, str):
            event('boo')
    stats = call_for_statistics(test)
    assert not unique_events(stats)

def test_exact_timing():
    if False:
        i = 10
        return i + 15

    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
    @given(st.integers())
    def test(i):
        if False:
            for i in range(10):
                print('nop')
        time.sleep(0.5)
    stats = describe_statistics(call_for_statistics(test))
    assert '~ 529ms' in stats

def test_apparently_instantaneous_tests():
    if False:
        i = 10
        return i + 15
    time.freeze()

    @given(st.integers())
    def test(i):
        if False:
            i = 10
            return i + 15
        pass
    stats = describe_statistics(call_for_statistics(test))
    assert '< 1ms' in stats

def test_flaky_exit():
    if False:
        for i in range(10):
            print('nop')
    first = [True]

    @settings(derandomize=True)
    @given(st.integers())
    def test(i):
        if False:
            print('Hello World!')
        if i > 1001:
            if first[0]:
                first[0] = False
                print('Hi')
                raise AssertionError
    stats = call_for_statistics(test)
    assert stats['stopped-because'] == 'test was flaky'

@pytest.mark.parametrize('draw_delay', [False, True])
@pytest.mark.parametrize('test_delay', [False, True])
def test_draw_timing(draw_delay, test_delay):
    if False:
        for i in range(10):
            print('nop')
    time.freeze()

    @st.composite
    def s(draw):
        if False:
            for i in range(10):
                print('nop')
        if draw_delay:
            time.sleep(0.05)
        draw(st.integers())

    @given(s())
    def test(_):
        if False:
            while True:
                i = 10
        if test_delay:
            time.sleep(0.05)
    stats = describe_statistics(call_for_statistics(test))
    if not draw_delay:
        assert '< 1ms' in stats
    else:
        match = re.search('of which ~ (?P<gentime>\\d+)', stats)
        assert 49 <= int(match.group('gentime')) <= 51

def test_has_lambdas_in_output():
    if False:
        print('Hello World!')

    @settings(max_examples=100, database=None)
    @given(st.integers().filter(lambda x: x % 2 == 0))
    def test(i):
        if False:
            print('Hello World!')
        pass
    stats = call_for_statistics(test)
    assert any(('lambda x: x % 2 == 0' in e for e in unique_events(stats)))

def test_stops_after_x_shrinks(monkeypatch):
    if False:
        while True:
            i = 10
    from hypothesis.internal.conjecture import engine
    monkeypatch.setattr(engine, 'MAX_SHRINKS', 0)

    @given(st.integers(min_value=0))
    def test(n):
        if False:
            for i in range(10):
                print('nop')
        assert n < 10
    stats = call_for_statistics(test)
    assert 'shrunk example' in stats['stopped-because']

def test_stateful_states_are_deduped():
    if False:
        while True:
            i = 10

    class DemoStateMachine(stateful.RuleBasedStateMachine):
        Stuff = stateful.Bundle('stuff')

        @stateful.rule(target=Stuff, name=st.text())
        def create_stuff(self, name):
            if False:
                return 10
            return name

        @stateful.rule(item=Stuff)
        def do(self, item):
            if False:
                return 10
            return
    stats = call_for_statistics(DemoStateMachine.TestCase().runTest)
    assert len(unique_events(stats)) <= 2

def test_stateful_with_one_of_bundles_states_are_deduped():
    if False:
        for i in range(10):
            print('nop')

    class DemoStateMachine(stateful.RuleBasedStateMachine):
        Things = stateful.Bundle('things')
        Stuff = stateful.Bundle('stuff')
        StuffAndThings = Things | Stuff

        @stateful.rule(target=Things, name=st.text())
        def create_thing(self, name):
            if False:
                print('Hello World!')
            return name

        @stateful.rule(target=Stuff, name=st.text())
        def create_stuff(self, name):
            if False:
                print('Hello World!')
            return name

        @stateful.rule(item=StuffAndThings)
        def do(self, item):
            if False:
                i = 10
                return i + 15
            return
    stats = call_for_statistics(DemoStateMachine.TestCase().runTest)
    assert len(unique_events(stats)) <= 4

def test_statistics_for_threshold_problem():
    if False:
        return 10

    @settings(max_examples=100)
    @given(st.floats(min_value=0, allow_infinity=False))
    def threshold(error):
        if False:
            while True:
                i = 10
        target(error, label='error')
        assert error <= 10
        target(0.0, label='never in failing example')
    stats = call_for_statistics(threshold)
    assert '  - Highest target scores:' in describe_statistics(stats)
    assert 'never in failing example' in describe_statistics(stats)
    assert stats['targets']['error'] > 10

def test_statistics_with_events_and_target():
    if False:
        return 10

    @given(st.integers(0, 10000))
    def test(value):
        if False:
            print('Hello World!')
        event(value)
        target(float(value), label='a target')
    stats = describe_statistics(call_for_statistics(test))
    assert '- Events:' in stats
    assert '- Highest target score: ' in stats

@given(st.booleans())
def test_event_with_non_weakrefable_keys(b):
    if False:
        i = 10
        return i + 15
    event((b,))