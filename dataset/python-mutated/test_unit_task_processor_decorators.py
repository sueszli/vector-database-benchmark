import json
import logging
from datetime import timedelta
from unittest.mock import MagicMock
import pytest
from django_capture_on_commit_callbacks import capture_on_commit_callbacks
from pytest_django.fixtures import SettingsWrapper
from pytest_mock import MockerFixture
from task_processor.decorators import register_recurring_task, register_task_handler
from task_processor.exceptions import InvalidArgumentsError
from task_processor.models import RecurringTask, Task, TaskPriority
from task_processor.task_registry import get_task
from task_processor.task_run_method import TaskRunMethod

@pytest.fixture
def capture_task_processor_logger(caplog: pytest.LogCaptureFixture) -> None:
    if False:
        print('Hello World!')
    task_processor_logger = logging.getLogger('task_processor')
    task_processor_logger.propagate = True
    task_processor_logger.setLevel(logging.INFO)
    caplog.set_level(logging.INFO)

@pytest.fixture
def mock_thread_class(mocker: MockerFixture) -> MagicMock:
    if False:
        i = 10
        return i + 15
    mock_thread_class = mocker.patch('task_processor.decorators.Thread', return_value=mocker.MagicMock())
    return mock_thread_class

@pytest.mark.django_db
def test_register_task_handler_run_in_thread__transaction_commit__true__default(capture_task_processor_logger: None, caplog: pytest.LogCaptureFixture, mock_thread_class: MagicMock) -> None:
    if False:
        i = 10
        return i + 15

    @register_task_handler()
    def my_function(*args: str, **kwargs: str) -> None:
        if False:
            i = 10
            return i + 15
        pass
    mock_thread = mock_thread_class.return_value
    args = ('foo',)
    kwargs = {'bar': 'baz'}
    with capture_on_commit_callbacks(execute=True):
        my_function.run_in_thread(args=args, kwargs=kwargs)
    mock_thread_class.assert_called_once_with(target=my_function.unwrapped, args=args, kwargs=kwargs, daemon=True)
    mock_thread.start.assert_called_once()
    assert len(caplog.records) == 1
    assert caplog.records[0].message == 'Running function my_function in unmanaged thread.'

def test_register_task_handler_run_in_thread__transaction_commit__false(capture_task_processor_logger: None, caplog: pytest.LogCaptureFixture, mock_thread_class: MagicMock) -> None:
    if False:
        i = 10
        return i + 15

    @register_task_handler(transaction_on_commit=False)
    def my_function(*args, **kwargs):
        if False:
            while True:
                i = 10
        pass
    mock_thread = mock_thread_class.return_value
    args = ('foo',)
    kwargs = {'bar': 'baz'}
    my_function.run_in_thread(args=args, kwargs=kwargs)
    mock_thread_class.assert_called_once_with(target=my_function.unwrapped, args=args, kwargs=kwargs, daemon=True)
    mock_thread.start.assert_called_once()
    assert len(caplog.records) == 1
    assert caplog.records[0].message == 'Running function my_function in unmanaged thread.'

def test_register_recurring_task(mocker, db, run_by_processor):
    if False:
        while True:
            i = 10
    task_kwargs = {'first_arg': 'foo', 'second_arg': 'bar'}
    run_every = timedelta(minutes=10)
    task_identifier = 'test_unit_task_processor_decorators.a_function'

    @register_recurring_task(run_every=run_every, kwargs=task_kwargs)
    def a_function(first_arg, second_arg):
        if False:
            return 10
        return first_arg + second_arg
    task = RecurringTask.objects.get(task_identifier=task_identifier)
    assert task.serialized_kwargs == json.dumps(task_kwargs)
    assert task.run_every == run_every
    assert get_task(task_identifier)
    assert task.run() == 'foobar'

def test_register_recurring_task_does_nothing_if_not_run_by_processor(mocker, db):
    if False:
        return 10
    task_kwargs = {'first_arg': 'foo', 'second_arg': 'bar'}
    run_every = timedelta(minutes=10)
    task_identifier = 'test_unit_task_processor_decorators.some_function'

    @register_recurring_task(run_every=run_every, kwargs=task_kwargs)
    def some_function(first_arg, second_arg):
        if False:
            for i in range(10):
                print('nop')
        return first_arg + second_arg
    assert not RecurringTask.objects.filter(task_identifier=task_identifier).exists()
    with pytest.raises(KeyError):
        assert get_task(task_identifier)

def test_register_task_handler_validates_inputs() -> None:
    if False:
        while True:
            i = 10

    @register_task_handler()
    def my_function(*args, **kwargs):
        if False:
            while True:
                i = 10
        pass

    class NonSerializableObj:
        pass
    with pytest.raises(InvalidArgumentsError):
        my_function(NonSerializableObj())

@pytest.mark.parametrize('task_run_method', (TaskRunMethod.SEPARATE_THREAD, TaskRunMethod.SYNCHRONOUSLY))
def test_inputs_are_validated_when_run_without_task_processor(settings: SettingsWrapper, task_run_method: TaskRunMethod) -> None:
    if False:
        i = 10
        return i + 15
    settings.TASK_RUN_METHOD = task_run_method

    @register_task_handler()
    def my_function(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        pass

    class NonSerializableObj:
        pass
    with pytest.raises(InvalidArgumentsError):
        my_function.delay(args=(NonSerializableObj(),))

def test_delay_returns_none_if_task_queue_is_full(settings, db):
    if False:
        print('Hello World!')
    settings.TASK_RUN_METHOD = TaskRunMethod.TASK_PROCESSOR

    @register_task_handler(queue_size=1)
    def my_function(*args, **kwargs):
        if False:
            while True:
                i = 10
        pass
    for _ in range(10):
        Task.objects.create(task_identifier='test_unit_task_processor_decorators.my_function')
    task = my_function.delay()
    assert task is None

def test_can_create_task_with_priority(settings, db):
    if False:
        while True:
            i = 10
    settings.TASK_RUN_METHOD = TaskRunMethod.TASK_PROCESSOR

    @register_task_handler(priority=TaskPriority.HIGH)
    def my_function(*args, **kwargs):
        if False:
            return 10
        pass
    for _ in range(10):
        Task.objects.create(task_identifier='test_unit_task_processor_decorators.my_function')
    task = my_function.delay()
    assert task.priority == TaskPriority.HIGH