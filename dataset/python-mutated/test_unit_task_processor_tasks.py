from datetime import timedelta
from django.utils import timezone
from task_processor.models import Task
from task_processor.tasks import clean_up_old_tasks
now = timezone.now()
three_days_ago = now - timedelta(days=3)
one_day_ago = now - timedelta(days=1)
one_hour_from_now = now + timedelta(hours=1)
sixty_days_ago = now - timedelta(days=60)

def test_clean_up_old_tasks_does_nothing_when_no_tasks(db):
    if False:
        while True:
            i = 10
    assert Task.objects.count() == 0
    clean_up_old_tasks()
    assert Task.objects.count() == 0

def test_clean_up_old_tasks(settings, django_assert_num_queries, db):
    if False:
        print('Hello World!')
    settings.TASK_DELETE_RETENTION_DAYS = 2
    settings.TASK_DELETE_BATCH_SIZE = 1
    for _ in range(2):
        Task.objects.create(task_identifier='some.identifier', scheduled_for=three_days_ago, completed=True)
    task_in_retention_period = Task.objects.create(task_identifier='some.identifier', scheduled_for=one_day_ago, completed=True)
    future_task = Task.objects.create(task_identifier='some.identifier', scheduled_for=one_hour_from_now)
    failed_task = Task.objects.create(task_identifier='some.identifier', scheduled_for=three_days_ago, num_failures=3)
    with django_assert_num_queries(9):
        clean_up_old_tasks()
    assert list(Task.objects.all()) == [task_in_retention_period, future_task, failed_task]

def test_clean_up_old_tasks_include_failed_tasks(settings, django_assert_num_queries, db):
    if False:
        return 10
    settings.TASK_DELETE_RETENTION_DAYS = 2
    settings.TASK_DELETE_INCLUDE_FAILED_TASKS = True
    Task.objects.create(task_identifier='some.identifier', scheduled_for=three_days_ago, num_failures=3)
    clean_up_old_tasks()
    assert not Task.objects.exists()

def test_clean_up_old_tasks_does_not_run_if_disabled(settings, django_assert_num_queries, db):
    if False:
        for i in range(10):
            print('nop')
    settings.ENABLE_CLEAN_UP_OLD_TASKS = False
    task = Task.objects.create(task_identifier='some.identifier', scheduled_for=sixty_days_ago)
    with django_assert_num_queries(0):
        clean_up_old_tasks()
    assert Task.objects.filter(id=task.id).exists()