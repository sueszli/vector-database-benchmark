import time
from sentry.dynamic_sampling.tasks.task_context import DynamicSamplingLogState, TaskContext, Timers
from sentry.testutils.helpers.datetime import freeze_time

def test_task_context_expiration_time():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests that the TaskContext properly initialises the expiration_time\n    '
    with freeze_time('2023-07-12 10:00:00'):
        context = TaskContext('my-task', 3)
        assert context.expiration_time == time.monotonic() + 3

def test_task_context_data():
    if False:
        print('Hello World!')
    '\n    Tests that TaskContext properly handles function contexts\n\n    * it deals with defaults and missing values\n    * it sets and retrieves values correctly\n    * it keeps various function contexts separated from each other\n    '
    context = TaskContext('my-task', 3)
    assert context.get_function_state('func1') == DynamicSamplingLogState()
    context.set_function_state('func1', DynamicSamplingLogState(num_rows_total=1, num_db_calls=2))
    assert context.get_function_state('func1') == DynamicSamplingLogState(num_rows_total=1, num_db_calls=2)
    assert context.get_function_state('func2') == DynamicSamplingLogState()
    context.set_function_state('func2', DynamicSamplingLogState(num_rows_total=1, num_db_calls=2, num_iterations=3, num_projects=4, num_orgs=5, execution_time=2.3))
    assert context.get_function_state('func1') == DynamicSamplingLogState(num_rows_total=1, num_db_calls=2)
    assert context.get_function_state('func2') == DynamicSamplingLogState(num_rows_total=1, num_db_calls=2, num_iterations=3, num_projects=4, num_orgs=5, execution_time=2.3)

def test_timer_raw():
    if False:
        i = 10
        return i + 15
    '\n    Tests the direct functionality of Timer (i.e. not as a context manager)\n    '
    t = Timers().get_timer('a')
    with freeze_time('2023-07-12 10:00:00') as frozen_time:
        assert t.current() == 0
        t.start()
        t.stop()
        assert t.current() == 0
        t.start()
        assert t.current() == 0
        t.stop()
        frozen_time.shift(1)
        assert t.current() == 0
        t.start()
        assert t.current() == 0
        t.stop()
        assert t.current() == 0
        t.start()
        frozen_time.shift(1)
        assert t.current() == 1.0
        t.start()
        assert t.current() == 1.0
        t.stop()
        assert t.current() == 1.0
        t.start()
        assert t.current() == 1.0
        t.stop()
        assert t.current() == 1.0
        frozen_time.shift(1)
        assert t.current() == 1.0
        t.start()
        assert t.current() == 1.0
        frozen_time.shift(1)
        assert t.current() == 2.0
        t.stop()
        assert t.current() == 2.0
        t.start()
        assert t.current() == 2.0
        t.stop()
        assert t.current() == 2.0

def test_named_timer_raw():
    if False:
        while True:
            i = 10
    '\n    Tests the direct functionality of Timer (i.e. not as a context manager)\n    with named timers\n    '
    t = Timers()
    with freeze_time('2023-07-12 10:00:00') as frozen_time:
        ta = t.get_timer('a')
        tb = t.get_timer('b')
        tc = t.get_timer('c')
        assert ta.current() == 0
        assert tb.current() == 0
        assert tc.current() == 0
        ta.start()
        ta.stop()
        tc.start()
        assert ta.current() == 0
        assert tb.current() == 0
        assert tc.current() == 0
        ta.start()
        assert ta.current() == 0
        assert tb.current() == 0
        assert tc.current() == 0
        ta.stop()
        frozen_time.shift(1)
        assert ta.current() == 0
        assert tb.current() == 0
        assert tc.current() == 1
        ta.start()
        assert ta.current() == 0
        assert tb.current() == 0
        assert tc.current() == 1
        ta.stop()
        assert ta.current() == 0
        assert tb.current() == 0
        assert tc.current() == 1
        ta.start()
        frozen_time.shift(1)
        assert ta.current() == 1.0
        assert tb.current() == 0.0
        assert tc.current() == 2.0
        ta.start()
        assert ta.current() == 1.0
        assert tb.current() == 0.0
        assert tc.current() == 2.0
        ta.stop()
        assert ta.current() == 1.0
        assert tb.current() == 0.0
        assert tc.current() == 2.0
        tb.start()
        frozen_time.shift(1)
        assert ta.current() == 1.0
        assert tb.current() == 1.0
        assert tc.current() == 3.0
        ta.start()
        assert ta.current() == 1.0
        assert tb.current() == 1.0
        assert tc.current() == 3.0
        frozen_time.shift(1)
        assert ta.current() == 2.0
        assert tb.current() == 2.0
        assert tc.current() == 4.0
        ta.stop()
        assert ta.current() == 2.0
        assert tb.current() == 2.0
        assert tc.current() == 4.0
        frozen_time.shift(1)
        assert ta.current() == 2.0
        assert tb.current() == 3.0
        assert tc.current() == 5.0

def test_timer_context_manager():
    if False:
        while True:
            i = 10
    '\n    Tests the context manager functionality of the timer\n    '
    with freeze_time('2023-07-12 10:00:00') as frozen_time:
        t = Timers()
        for i in range(3):
            with t.get_timer('a'):
                assert t.current('a') == i
                frozen_time.shift(1)
            frozen_time.shift(1)
        assert t.current('a') == 3

def test_named_timer_context_manager():
    if False:
        return 10
    '\n    Tests the context manager functionality of the timer\n    '
    with freeze_time('2023-07-12 10:00:00') as frozen_time:
        t = Timers()
        for i in range(3):
            with t.get_timer('global') as t_global:
                assert t_global.current() == i * 7
                assert t.current('global') == i * 7
                with t.get_timer('a') as ta:
                    assert ta.current() == i
                    frozen_time.shift(1)
                with t.get_timer('b') as tb:
                    assert tb.current() == i * 2
                    frozen_time.shift(2)
                with t.get_timer('c') as tc:
                    assert tc.current() == i * 3
                    frozen_time.shift(3)
                frozen_time.shift(1)
            frozen_time.shift(1)
        frozen_time.shift(100)
        assert t.current('a') == 3
        assert t.current('b') == 3 * 2
        assert t.current('c') == 3 * 3
        assert t.current('global') == 3 * 7

def test_task_context_serialisation():
    if False:
        return 10
    task = TaskContext('my-task', 100)
    with freeze_time('2023-07-12 10:00:00') as frozen_time:
        with task.get_timer('a'):
            frozen_time.shift(1)
        with task.get_timer('b'):
            frozen_time.shift(2)
            state = task.get_function_state('b')
            state.num_iterations = 1
            state.num_orgs = 2
            state.num_projects = 3
            state.num_db_calls = 4
            state.num_rows_total = 5
            task.set_function_state('b', state)
        state = task.get_function_state('c')
        state.num_iterations = 1
        task.set_function_state('c', state)
    result = task.to_dict()
    del result['seconds']
    assert result == {'maxSeconds': 100, 'taskName': 'my-task', 'taskData': {'a': {'executionTime': 1.0, 'numDbCalls': 0, 'numIterations': 0, 'numOrgs': 0, 'numProjects': 0, 'numRowsTotal': 0}, 'b': {'executionTime': 2.0, 'numDbCalls': 4, 'numIterations': 1, 'numOrgs': 2, 'numProjects': 3, 'numRowsTotal': 5}, 'c': {'executionTime': 0, 'numDbCalls': 0, 'numIterations': 1, 'numOrgs': 0, 'numProjects': 0, 'numRowsTotal': 0}}}