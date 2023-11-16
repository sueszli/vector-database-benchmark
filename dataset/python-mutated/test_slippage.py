import pytest
from hypothesis import Phase, assume, given, settings, strategies as st, target
from hypothesis.database import InMemoryExampleDatabase
from hypothesis.errors import Flaky
from hypothesis.internal.compat import ExceptionGroup
from hypothesis.internal.conjecture.engine import MIN_TEST_CALLS
from tests.common.utils import assert_output_contains_failure, capture_out, non_covering_examples

def capture_reports(test):
    if False:
        for i in range(10):
            print('nop')
    with capture_out() as o, pytest.raises(ExceptionGroup) as err:
        test()
    return o.getvalue() + '\n\n'.join((f'{e!r}\n' + '\n'.join(getattr(e, '__notes__', [])) for e in (err.value, *err.value.exceptions)))

def test_raises_multiple_failures_with_varying_type():
    if False:
        i = 10
        return i + 15
    target = [None]

    @settings(database=None, max_examples=100)
    @given(st.integers())
    def test(i):
        if False:
            while True:
                i = 10
        if abs(i) < 1000:
            return
        if target[0] is None:
            assume(1003 < abs(i))
            target[0] = i
        exc_class = TypeError if target[0] == i else ValueError
        raise exc_class
    output = capture_reports(test)
    assert 'TypeError' in output
    assert 'ValueError' in output

def test_shows_target_scores_with_multiple_failures():
    if False:
        print('Hello World!')

    @settings(derandomize=True, max_examples=10000)
    @given(st.integers())
    def test(i):
        if False:
            i = 10
            return i + 15
        target(i)
        assert i > 0
        assert i < 0
    assert 'Highest target score:' in capture_reports(test)

def test_raises_multiple_failures_when_position_varies():
    if False:
        i = 10
        return i + 15
    target = [None]

    @settings(max_examples=100)
    @given(st.integers())
    def test(i):
        if False:
            i = 10
            return i + 15
        if abs(i) < 1000:
            return
        if target[0] is None:
            target[0] = i
        if target[0] == i:
            raise ValueError('loc 1')
        else:
            raise ValueError('loc 2')
    output = capture_reports(test)
    assert 'loc 1' in output
    assert 'loc 2' in output

def test_replays_both_failing_values():
    if False:
        i = 10
        return i + 15
    target = [None]

    @settings(database=InMemoryExampleDatabase(), max_examples=500)
    @given(st.integers())
    def test(i):
        if False:
            while True:
                i = 10
        if abs(i) < 1000:
            return
        if target[0] is None:
            target[0] = i
        exc_class = TypeError if target[0] == i else ValueError
        raise exc_class
    with pytest.raises(ExceptionGroup):
        test()
    with pytest.raises(ExceptionGroup):
        test()

@pytest.mark.parametrize('fix', [TypeError, ValueError])
def test_replays_slipped_examples_once_initial_bug_is_fixed(fix):
    if False:
        while True:
            i = 10
    target = []
    bug_fixed = False

    @settings(database=InMemoryExampleDatabase(), max_examples=500)
    @given(st.integers())
    def test(i):
        if False:
            i = 10
            return i + 15
        if abs(i) < 1000:
            return
        if not target:
            target.append(i)
        if i == target[0]:
            if bug_fixed and fix == TypeError:
                return
            raise TypeError
        if len(target) == 1:
            target.append(i)
        if bug_fixed and fix == ValueError:
            return
        if i == target[1]:
            raise ValueError
    with pytest.raises(ExceptionGroup):
        test()
    bug_fixed = True
    with pytest.raises(ValueError if fix == TypeError else TypeError):
        test()

def test_garbage_collects_the_secondary_key():
    if False:
        while True:
            i = 10
    target = []
    bug_fixed = False
    db = InMemoryExampleDatabase()

    @settings(database=db, max_examples=500)
    @given(st.integers())
    def test(i):
        if False:
            for i in range(10):
                print('nop')
        if bug_fixed:
            return
        if abs(i) < 1000:
            return
        if not target:
            target.append(i)
        if i == target[0]:
            raise TypeError
        if len(target) == 1:
            target.append(i)
        if i == target[1]:
            raise ValueError
    with pytest.raises(ExceptionGroup):
        test()
    bug_fixed = True

    def count():
        if False:
            return 10
        return len(non_covering_examples(db))
    prev = count()
    while prev > 0:
        test()
        current = count()
        assert current < prev
        prev = current

def test_shrinks_both_failures():
    if False:
        return 10
    first_has_failed = [False]
    duds = set()
    second_target = [None]

    @settings(database=None, max_examples=1000)
    @given(st.integers(min_value=0).map(int))
    def test(i):
        if False:
            for i in range(10):
                print('nop')
        if i >= 10000:
            first_has_failed[0] = True
            raise AssertionError
        assert i < 10000
        if first_has_failed[0]:
            if second_target[0] is None:
                for j in range(10000):
                    if j not in duds:
                        second_target[0] = j
                        break
            assert i < second_target[0]
        else:
            duds.add(i)
    output = capture_reports(test)
    assert_output_contains_failure(output, test, i=10000)
    assert_output_contains_failure(output, test, i=second_target[0])

def test_handles_flaky_tests_where_only_one_is_flaky():
    if False:
        return 10
    flaky_fixed = False
    target = []
    flaky_failed_once = [False]

    @settings(database=InMemoryExampleDatabase(), max_examples=1000)
    @given(st.integers())
    def test(i):
        if False:
            print('Hello World!')
        if abs(i) < 1000:
            return
        if not target:
            target.append(i)
        if i == target[0]:
            raise TypeError
        if flaky_failed_once[0] and (not flaky_fixed):
            return
        if len(target) == 1:
            target.append(i)
        if i == target[1]:
            flaky_failed_once[0] = True
            raise ValueError
    with pytest.raises(ExceptionGroup) as err:
        test()
    assert any((isinstance(e, Flaky) for e in err.value.exceptions))
    flaky_fixed = True
    with pytest.raises(ExceptionGroup) as err:
        test()
    assert not any((isinstance(e, Flaky) for e in err.value.exceptions))

@pytest.mark.parametrize('allow_multi', [True, False])
def test_can_disable_multiple_error_reporting(allow_multi):
    if False:
        print('Hello World!')
    seen = set()

    @settings(database=None, derandomize=True, report_multiple_bugs=allow_multi)
    @given(st.integers(min_value=0))
    def test(i):
        if False:
            return 10
        if i == 1:
            seen.add(TypeError)
            raise TypeError
        elif i >= 2:
            seen.add(ValueError)
            raise ValueError
    with pytest.raises(ExceptionGroup if allow_multi else TypeError):
        test()
    assert seen == {TypeError, ValueError}

def test_finds_multiple_failures_in_generation():
    if False:
        return 10
    special = []
    seen = set()

    @settings(phases=[Phase.generate, Phase.shrink], max_examples=100)
    @given(st.integers(min_value=0))
    def test(x):
        if False:
            print('Hello World!')
        "Constructs a test so the 10th largeish example we've seen is a\n        special failure, and anything new we see after that point that\n        is larger than it is a different failure. This demonstrates that we\n        can keep generating larger examples and still find new bugs after that\n        point."
        if not special:
            if len(seen) >= 10 and x <= 1000:
                special.append(x)
            else:
                seen.add(x)
        if special:
            assert x in seen or x <= special[0]
        assert x not in special
    with pytest.raises(ExceptionGroup):
        test()

def test_stops_immediately_if_not_report_multiple_bugs():
    if False:
        return 10
    seen = set()

    @settings(phases=[Phase.generate], report_multiple_bugs=False)
    @given(st.integers())
    def test(x):
        if False:
            i = 10
            return i + 15
        seen.add(x)
        raise AssertionError
    with pytest.raises(AssertionError):
        test()
    assert len(seen) == 1

def test_stops_immediately_on_replay():
    if False:
        i = 10
        return i + 15
    seen = set()

    @settings(database=InMemoryExampleDatabase(), phases=tuple(Phase)[:-1])
    @given(st.integers())
    def test(x):
        if False:
            return 10
        seen.add(x)
        assert x
    with pytest.raises(AssertionError):
        test()
    assert 1 < len(seen) <= MIN_TEST_CALLS
    seen.clear()
    with pytest.raises(AssertionError):
        test()
    assert len(seen) == 1