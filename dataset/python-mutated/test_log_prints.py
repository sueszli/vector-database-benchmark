import builtins
import pytest
from prefect import flow, task
from prefect.context import get_run_context
from prefect.logging.loggers import print_as_log
from prefect.settings import PREFECT_LOGGING_LOG_PRINTS, temporary_settings

def test_log_prints_patch_is_scoped_to_task():
    if False:
        i = 10
        return i + 15

    @task(log_prints=True)
    def get_builtin_print():
        if False:
            print('Hello World!')
        return builtins.print

    @flow
    def wrapper():
        if False:
            while True:
                i = 10
        return (builtins.print, get_builtin_print())
    (caller_builtin_print, user_builtin_print) = wrapper()
    assert caller_builtin_print is builtins.print
    assert user_builtin_print is print_as_log

def test_log_prints_patch_is_scoped_to_subflow():
    if False:
        while True:
            i = 10

    @flow(log_prints=True)
    def get_builtin_print():
        if False:
            print('Hello World!')
        return builtins.print

    @flow
    def wrapper():
        if False:
            print('Hello World!')
        return (builtins.print, get_builtin_print())
    (caller_builtin_print, user_builtin_print) = wrapper()
    assert caller_builtin_print is builtins.print
    assert user_builtin_print is print_as_log

@pytest.mark.parametrize('setting_value', [True, False])
def test_root_flow_log_prints_defaults_to_setting_value(caplog, setting_value):
    if False:
        return 10

    @flow
    def test_flow():
        if False:
            for i in range(10):
                print('nop')
        print('hello world!')
    with temporary_settings({PREFECT_LOGGING_LOG_PRINTS: setting_value}):
        test_flow()
    assert ('hello world!' in caplog.text) is setting_value

@pytest.mark.parametrize('setting_value', [True, False])
def test_task_log_prints_defaults_to_setting_value(caplog, setting_value):
    if False:
        i = 10
        return i + 15

    @task
    def test_task():
        if False:
            while True:
                i = 10
        print('hello world!')

    @flow
    def parent_flow():
        if False:
            for i in range(10):
                print('nop')
        test_task()
    with temporary_settings({PREFECT_LOGGING_LOG_PRINTS: setting_value}):
        parent_flow()
    assert ('hello world!' in caplog.text) is setting_value

@pytest.mark.parametrize('setting_value', [True, False])
def test_subflow_log_prints_defaults_to_setting_value(caplog, setting_value):
    if False:
        while True:
            i = 10

    @flow
    def test_flow():
        if False:
            print('Hello World!')
        print('hello world!')

    @flow
    def parent_flow():
        if False:
            return 10
        test_flow()
    with temporary_settings({PREFECT_LOGGING_LOG_PRINTS: setting_value}):
        parent_flow()
    assert ('hello world!' in caplog.text) is setting_value

@pytest.mark.parametrize('setting_value', [True, False])
@pytest.mark.parametrize('parent_value', [True, False])
def test_task_log_prints_inherits_parent_value(caplog, setting_value, parent_value):
    if False:
        print('Hello World!')

    @task
    def test_task():
        if False:
            print('Hello World!')
        print('hello world!')

    @flow(log_prints=parent_value)
    def parent_flow():
        if False:
            for i in range(10):
                print('nop')
        test_task()
    with temporary_settings({PREFECT_LOGGING_LOG_PRINTS: setting_value}):
        parent_flow()
    assert ('hello world!' in caplog.text) is parent_value

@pytest.mark.parametrize('setting_value', [True, False])
@pytest.mark.parametrize('parent_value', [True, False])
def test_subflow_log_prints_inherits_parent_value(caplog, setting_value, parent_value):
    if False:
        i = 10
        return i + 15

    @flow
    def test_subflow():
        if False:
            while True:
                i = 10
        print('hello world!')

    @flow(log_prints=parent_value)
    def parent_flow():
        if False:
            i = 10
            return i + 15
        return test_subflow()
    with temporary_settings({PREFECT_LOGGING_LOG_PRINTS: setting_value}):
        parent_flow()
    assert ('hello world!' in caplog.text) is parent_value

@pytest.mark.parametrize('parent_value', [True, False, None])
def test_nested_subflow_log_prints_inherits_parent_value(caplog, parent_value):
    if False:
        print('Hello World!')

    @flow
    def three():
        if False:
            i = 10
            return i + 15
        print('hello world!')

    @flow(log_prints=parent_value)
    def two():
        if False:
            return 10
        return three()

    @flow(log_prints=True)
    def one():
        if False:
            while True:
                i = 10
        return two()
    one()
    if parent_value is not False:
        assert 'hello world!' in caplog.text
    else:
        assert 'hello world!' not in caplog.text

@pytest.mark.parametrize('parent_value', [False, None])
def test_task_can_opt_in_to_log_prints(caplog, parent_value):
    if False:
        while True:
            i = 10

    @task(log_prints=True)
    def test_task():
        if False:
            i = 10
            return i + 15
        task_run_name = get_run_context().task_run.name
        print(f'test print from {task_run_name}')
        return task_run_name

    @flow(log_prints=parent_value)
    def parent_flow():
        if False:
            return 10
        return test_task()
    printing_task_name = parent_flow()
    assert f'test print from {printing_task_name}' in caplog.text

@pytest.mark.parametrize('parent_value', [False, None])
def test_subflow_can_opt_in_to_log_prints(caplog, parent_value):
    if False:
        print('Hello World!')

    @flow(log_prints=True)
    def test_flow():
        if False:
            i = 10
            return i + 15
        print('hello world!')

    @flow(log_prints=parent_value)
    def parent_flow():
        if False:
            i = 10
            return i + 15
        return test_flow()
    parent_flow()
    assert 'hello world!' in caplog.text

def test_task_can_opt_out_of_log_prints(caplog, capsys):
    if False:
        print('Hello World!')

    @task(log_prints=False)
    def test_task():
        if False:
            i = 10
            return i + 15
        task_run_name = get_run_context().task_run.name
        print(f'test print from {task_run_name}')
        return task_run_name

    @flow(log_prints=True)
    def parent_flow():
        if False:
            print('Hello World!')
        return test_task()
    printing_task_name = parent_flow()
    assert f'test print from {printing_task_name}' not in caplog.text
    assert f'test print from {printing_task_name}' in capsys.readouterr().out

def test_subflow_can_opt_out_of_log_prints(caplog, capsys):
    if False:
        while True:
            i = 10

    @flow(log_prints=False)
    def test_flow():
        if False:
            while True:
                i = 10
        print('hello world!')

    @flow(log_prints=True)
    def parent_flow():
        if False:
            i = 10
            return i + 15
        return test_flow()
    parent_flow()
    assert 'hello world!' not in caplog.text
    assert 'hello world!' in capsys.readouterr().out

@pytest.mark.parametrize('value', [True, False, None])
def test_task_log_prints_updated_by_with_options(value):
    if False:
        i = 10
        return i + 15

    @task
    def test_task():
        if False:
            i = 10
            return i + 15
        task_run_name = get_run_context().task_run.name
        print(f'test print from {task_run_name}')
        return task_run_name
    new_task = test_task.with_options(log_prints=value)
    assert new_task.log_prints is value

@pytest.mark.parametrize('value', [True, False, None])
def test_flow_log_prints_updated_by_with_options(value):
    if False:
        while True:
            i = 10

    @flow
    def test_flow():
        if False:
            while True:
                i = 10
        print('hello world!')
    new_flow = test_flow.with_options(log_prints=value)
    assert new_flow.log_prints is value