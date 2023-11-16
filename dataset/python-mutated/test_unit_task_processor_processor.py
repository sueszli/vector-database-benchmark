import time
import uuid
from datetime import timedelta
from threading import Thread
import pytest
from django.utils import timezone
from organisations.models import Organisation
from task_processor.decorators import register_recurring_task, register_task_handler
from task_processor.models import RecurringTask, RecurringTaskRun, Task, TaskPriority, TaskResult, TaskRun
from task_processor.processor import run_recurring_tasks, run_tasks
from task_processor.task_registry import registered_tasks

def test_run_task_runs_task_and_creates_task_run_object_when_success(db):
    if False:
        while True:
            i = 10
    organisation_name = f'test-org-{uuid.uuid4()}'
    task = Task.create(_create_organisation.task_identifier, scheduled_for=timezone.now(), args=(organisation_name,))
    task.save()
    task_runs = run_tasks()
    assert Organisation.objects.filter(name=organisation_name).exists()
    assert len(task_runs) == TaskRun.objects.filter(task=task).count() == 1
    task_run = task_runs[0]
    assert task_run.result == TaskResult.SUCCESS
    assert task_run.started_at
    assert task_run.finished_at
    assert task_run.error_details is None
    task.refresh_from_db()
    assert task.completed

def test_run_recurring_tasks_runs_task_and_creates_recurring_task_run_object_when_success(db, monkeypatch):
    if False:
        while True:
            i = 10
    monkeypatch.setenv('RUN_BY_PROCESSOR', 'True')
    organisation_name = f'test-org-{uuid.uuid4()}'
    task_identifier = 'test_unit_task_processor_processor._create_organisation'

    @register_recurring_task(run_every=timedelta(seconds=1), args=(organisation_name,))
    def _create_organisation(organisation_name):
        if False:
            return 10
        Organisation.objects.create(name=organisation_name)
    task = RecurringTask.objects.get(task_identifier=task_identifier)
    task_runs = run_recurring_tasks()
    assert Organisation.objects.filter(name=organisation_name).count() == 1
    assert len(task_runs) == RecurringTaskRun.objects.filter(task=task).count() == 1
    task_run = task_runs[0]
    assert task_run.result == TaskResult.SUCCESS
    assert task_run.started_at
    assert task_run.finished_at
    assert task_run.error_details is None

@pytest.mark.django_db(transaction=True)
def test_run_recurring_tasks_multiple_runs(db, run_by_processor):
    if False:
        print('Hello World!')
    organisation_name = 'test-org'
    task_identifier = 'test_unit_task_processor_processor._create_organisation'

    @register_recurring_task(run_every=timedelta(milliseconds=200), args=(organisation_name,))
    def _create_organisation(organisation_name):
        if False:
            while True:
                i = 10
        Organisation.objects.create(name=organisation_name)
    task = RecurringTask.objects.get(task_identifier=task_identifier)
    first_task_runs = run_recurring_tasks()
    no_task_runs = run_recurring_tasks()
    time.sleep(0.3)
    second_task_runs = run_recurring_tasks()
    assert len(first_task_runs) == 1
    assert len(no_task_runs) == 0
    assert len(second_task_runs) == 1
    assert Organisation.objects.filter(name=organisation_name).count() == 2
    all_task_runs = first_task_runs + second_task_runs
    assert len(all_task_runs) == RecurringTaskRun.objects.filter(task=task).count() == 2
    for task_run in all_task_runs:
        assert task_run.result == TaskResult.SUCCESS
        assert task_run.started_at
        assert task_run.finished_at
        assert task_run.error_details is None

def test_run_recurring_tasks_only_executes_tasks_after_interval_set_by_run_every(db, run_by_processor):
    if False:
        print('Hello World!')
    organisation_name = 'test-org'
    task_identifier = 'test_unit_task_processor_processor._create_organisation'

    @register_recurring_task(run_every=timedelta(milliseconds=100), args=(organisation_name,))
    def _create_organisation(organisation_name):
        if False:
            print('Hello World!')
        Organisation.objects.create(name=organisation_name)
    task = RecurringTask.objects.get(task_identifier=task_identifier)
    run_recurring_tasks()
    run_recurring_tasks()
    assert Organisation.objects.filter(name=organisation_name).count() == 1
    assert RecurringTaskRun.objects.filter(task=task).count() == 1

def test_run_recurring_tasks_deletes_the_task_if_it_is_not_registered(db, run_by_processor):
    if False:
        i = 10
        return i + 15
    task_identifier = 'test_unit_task_processor_processor._a_task'

    @register_recurring_task(run_every=timedelta(milliseconds=100))
    def _a_task():
        if False:
            return 10
        pass
    registered_tasks.pop(task_identifier)
    task_runs = run_recurring_tasks()
    assert len(task_runs) == 0
    assert not RecurringTask.objects.filter(task_identifier=task_identifier).exists()

def test_run_task_runs_task_and_creates_task_run_object_when_failure(db):
    if False:
        for i in range(10):
            print('nop')
    task = Task.create(_raise_exception.task_identifier, scheduled_for=timezone.now())
    task.save()
    task_runs = run_tasks()
    assert len(task_runs) == TaskRun.objects.filter(task=task).count() == 1
    task_run = task_runs[0]
    assert task_run.result == TaskResult.FAILURE
    assert task_run.started_at
    assert task_run.finished_at is None
    assert task_run.error_details is not None
    task.refresh_from_db()
    assert not task.completed

def test_run_task_runs_failed_task_again(db):
    if False:
        while True:
            i = 10
    task = Task.create(_raise_exception.task_identifier, scheduled_for=timezone.now())
    task.save()
    first_task_runs = run_tasks()
    second_task_runs = run_tasks()
    task_runs = first_task_runs + second_task_runs
    assert len(task_runs) == TaskRun.objects.filter(task=task).count() == 2
    for task_run in task_runs:
        assert task_run.result == TaskResult.FAILURE
        assert task_run.started_at
        assert task_run.finished_at is None
        assert task_run.error_details is not None
    task.refresh_from_db()
    assert task.completed is False
    assert task.is_locked is False

def test_run_recurring_task_runs_task_and_creates_recurring_task_run_object_when_failure(db, run_by_processor):
    if False:
        for i in range(10):
            print('nop')
    task_identifier = 'test_unit_task_processor_processor._raise_exception'

    @register_recurring_task(run_every=timedelta(seconds=1))
    def _raise_exception(organisation_name):
        if False:
            for i in range(10):
                print('nop')
        raise RuntimeError('test exception')
    task = RecurringTask.objects.get(task_identifier=task_identifier)
    task_runs = run_recurring_tasks()
    assert len(task_runs) == RecurringTaskRun.objects.filter(task=task).count() == 1
    task_run = task_runs[0]
    assert task_run.result == TaskResult.FAILURE
    assert task_run.started_at
    assert task_run.finished_at is None
    assert task_run.error_details is not None

def test_run_task_does_nothing_if_no_tasks(db):
    if False:
        print('Hello World!')
    result = run_tasks()
    assert result == []
    assert not TaskRun.objects.exists()

@pytest.mark.django_db(transaction=True)
def test_run_task_runs_tasks_in_correct_priority():
    if False:
        for i in range(10):
            print('nop')
    task_1 = Task.create(_create_organisation.task_identifier, scheduled_for=timezone.now(), args=('task 1 organisation',), priority=TaskPriority.HIGH)
    task_1.save()
    task_2 = Task.create(_create_organisation.task_identifier, scheduled_for=timezone.now(), args=('task 2 organisation',), priority=TaskPriority.HIGH)
    task_2.save()
    task_3 = Task.create(_create_organisation.task_identifier, scheduled_for=timezone.now(), args=('task 3 organisation',), priority=TaskPriority.HIGHEST)
    task_3.save()
    task_runs_1 = run_tasks()
    task_runs_2 = run_tasks()
    task_runs_3 = run_tasks()
    assert task_runs_1[0].task == task_3
    assert task_runs_2[0].task == task_1
    assert task_runs_3[0].task == task_2

@pytest.mark.django_db(transaction=True)
def test_run_tasks_skips_locked_tasks():
    if False:
        for i in range(10):
            print('nop')
    "\n    This test verifies that tasks are locked while being executed, and hence\n    new task runners are not able to pick up 'in progress' tasks.\n    "
    task_1 = Task.create(_sleep.task_identifier, scheduled_for=timezone.now(), args=(3,))
    task_1.save()
    task_2 = Task.create(_create_organisation.task_identifier, scheduled_for=timezone.now(), args=('task 2 organisation',))
    task_2.save()
    task_runner_thread = Thread(target=run_tasks)
    task_runner_thread.start()
    time.sleep(1)
    task_runs = run_tasks()
    assert task_runs[0].task == task_2
    task_runner_thread.join()

def test_run_more_than_one_task(db):
    if False:
        i = 10
        return i + 15
    num_tasks = 5
    tasks = []
    for _ in range(num_tasks):
        organisation_name = f'test-org-{uuid.uuid4()}'
        tasks.append(Task.create(_create_organisation.task_identifier, scheduled_for=timezone.now(), args=(organisation_name,)))
    Task.objects.bulk_create(tasks)
    task_runs = run_tasks(5)
    assert len(task_runs) == num_tasks
    for task_run in task_runs:
        assert task_run.result == TaskResult.SUCCESS
        assert task_run.started_at
        assert task_run.finished_at
        assert task_run.error_details is None
    for task in tasks:
        task.refresh_from_db()
        assert task.completed

def test_recurring_tasks_are_unlocked_if_picked_up_but_not_executed(db, run_by_processor):
    if False:
        return 10

    @register_recurring_task(run_every=timedelta(days=1))
    def my_task():
        if False:
            for i in range(10):
                print('nop')
        pass
    recurring_task = RecurringTask.objects.get(task_identifier='test_unit_task_processor_processor.my_task')
    now = timezone.now()
    one_minute_ago = now - timedelta(minutes=1)
    RecurringTaskRun.objects.create(task=recurring_task, started_at=one_minute_ago, finished_at=now, result=TaskResult.SUCCESS.name)
    run_recurring_tasks()
    recurring_task.refresh_from_db()
    assert recurring_task.is_locked is False

@register_task_handler()
def _create_organisation(name: str):
    if False:
        return 10
    'function used to test that task is being run successfully'
    Organisation.objects.create(name=name)

@register_task_handler()
def _raise_exception():
    if False:
        i = 10
        return i + 15
    raise Exception()

@register_task_handler()
def _sleep(seconds: int):
    if False:
        i = 10
        return i + 15
    time.sleep(seconds)