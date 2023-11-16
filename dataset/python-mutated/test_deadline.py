import time
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.errors import DeadlineExceeded, Flaky, InvalidArgument
from tests.common.utils import assert_falsifying_output, fails_with

def test_raises_deadline_on_slow_test():
    if False:
        print('Hello World!')

    @settings(deadline=500)
    @given(st.integers())
    def slow(i):
        if False:
            i = 10
            return i + 15
        time.sleep(1)
    with pytest.raises(DeadlineExceeded):
        slow()

@fails_with(DeadlineExceeded)
@given(st.integers())
def test_slow_tests_are_errors_by_default(i):
    if False:
        return 10
    time.sleep(1)

def test_non_numeric_deadline_is_an_error():
    if False:
        print('Hello World!')
    with pytest.raises(InvalidArgument):
        settings(deadline='3 seconds')

@given(st.integers())
@settings(deadline=None)
def test_slow_with_none_deadline(i):
    if False:
        print('Hello World!')
    time.sleep(1)

def test_raises_flaky_if_a_test_becomes_fast_on_rerun():
    if False:
        print('Hello World!')
    once = [True]

    @settings(deadline=500)
    @given(st.integers())
    def test_flaky_slow(i):
        if False:
            print('Hello World!')
        if once[0]:
            once[0] = False
            time.sleep(1)
    with pytest.raises(Flaky):
        test_flaky_slow()

def test_deadlines_participate_in_shrinking():
    if False:
        i = 10
        return i + 15

    @settings(deadline=500, max_examples=1000)
    @given(st.integers(min_value=0))
    def slow_if_large(i):
        if False:
            for i in range(10):
                print('nop')
        if i >= 1000:
            time.sleep(1)
    assert_falsifying_output(slow_if_large, expected_exception=DeadlineExceeded, i=1000)

def test_keeps_you_well_above_the_deadline():
    if False:
        print('Hello World!')
    seen = set()
    failed_once = [False]

    @settings(deadline=100)
    @given(st.integers(0, 2000))
    def slow(i):
        if False:
            for i in range(10):
                print('nop')
        if not failed_once[0]:
            if i * 0.9 <= 100:
                return
            else:
                failed_once[0] = True
        t = i / 1000
        if i in seen:
            time.sleep(0.9 * t)
        else:
            seen.add(i)
            time.sleep(t)
    with pytest.raises(DeadlineExceeded):
        slow()

def test_gives_a_deadline_specific_flaky_error_message():
    if False:
        for i in range(10):
            print('nop')
    once = [True]

    @settings(deadline=100)
    @given(st.integers())
    def slow_once(i):
        if False:
            for i in range(10):
                print('nop')
        if once[0]:
            once[0] = False
            time.sleep(0.2)
    with pytest.raises(Flaky) as err:
        slow_once()
    assert 'Unreliable test timing' in '\n'.join(err.value.__notes__)
    assert 'took 2' in '\n'.join(err.value.__notes__)

@pytest.mark.parametrize('slow_strategy', [False, True])
@pytest.mark.parametrize('slow_test', [False, True])
def test_should_only_fail_a_deadline_if_the_test_is_slow(slow_strategy, slow_test):
    if False:
        while True:
            i = 10
    s = st.integers()
    if slow_strategy:
        s = s.map(lambda x: time.sleep(0.08))

    @settings(deadline=50)
    @given(st.data())
    def test(data):
        if False:
            i = 10
            return i + 15
        data.draw(s)
        if slow_test:
            time.sleep(0.1)
    if slow_test:
        with pytest.raises(DeadlineExceeded):
            test()
    else:
        test()