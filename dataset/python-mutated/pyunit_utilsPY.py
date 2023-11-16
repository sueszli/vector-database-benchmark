from tests import assert_equals
import numpy as np

def should_fail(*args, **kvargs):
    if False:
        i = 10
        return i + 15
    try:
        assert_equals(*args, **kvargs)
    except AssertionError:
        assert True, 'Should fail'
        return
    assert False, 'Should fail'

def should_pass(*args, **kvargs):
    if False:
        return 10
    try:
        assert_equals(*args, **kvargs)
    except AssertionError:
        assert False, 'Should pass'

def test_fail_is_failing_properly():
    if False:
        i = 10
        return i + 15
    try:
        should_fail('HELLO', 'HELLO')
    except AssertionError as e:
        assert 'Should fail' in str(e)
    should_fail('DUNNO', 'HELLO')

def test_pass_is_passing_properly():
    if False:
        while True:
            i = 10
    try:
        should_pass('ID', 'HELLO')
    except AssertionError as e:
        assert 'Should pass' in str(e)
    should_pass('HELLO', 'HELLO')

def test_assert_equals():
    if False:
        i = 10
        return i + 15
    should_pass('HI', 'HI')
    should_pass('10', '10')
    should_pass('10', '10', delta=0.001)
    should_fail('HI', 'HELLO')
    should_fail('HI', 'HELLO', delta=0.001)
    should_fail('HI', 3)
    should_fail('HI', 3, delta=0.001)
    should_pass(3, 3)
    should_pass(-3, -3)
    should_pass(3.0, 3.0, delta=10000000000.0)
    should_pass(-3.0, -3.0, delta=10000000000.0)
    should_pass(3.2335541321, 3.2339856985, delta=0.001)
    should_pass(-3.2335541321, -3.2339856985, delta=0.001)
    should_fail(3.2335541321, 3.2339856985, delta=0.0001)
    should_fail(-3.2335541321, -3.2339856985, delta=0.0001)
    should_fail(3.2335541321, 3.2339856985)
    should_fail(-3.2335541321, -3.2339856985)
    should_pass('nan', 'nan')
    should_pass('nan', 'nan', delta=0.001)
    should_fail(np.nan, np.nan)
    should_pass(np.nan, np.nan, delta=0.0001)
    should_fail('nan', np.nan, delta=0.001)
    should_fail(np.nan, 'nan', delta=0.001)
    should_fail('nan', np.nan)
    should_fail(np.nan, 'nan')
    should_pass('inf', 'inf')
    should_pass('inf', 'inf', delta=0.001)
    should_pass(np.inf, np.inf)
    should_pass(np.inf, np.inf, delta=0.0001)
    should_fail('inf', np.inf, delta=0.001)
    should_fail(np.inf, 'inf', delta=0.001)
    should_fail('inf', np.inf)
    should_fail(np.inf, 'inf')
test_fail_is_failing_properly()
test_pass_is_passing_properly()
test_assert_equals()