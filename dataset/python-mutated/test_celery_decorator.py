from django.core.management.base import BaseCommand
from dojo.models import Finding, Notes
from dojo.celery import app
from functools import wraps
from dojo.utils import test_valentijn

class Command(BaseCommand):
    help = "Command to do some tests with celery and decorators. Just committing it so 'we never forget'"

    def handle(self, *args, **options):
        if False:
            print('Hello World!')
        finding = Finding.objects.all().first()
        test_valentijn(finding, Notes.objects.all().first())

def test2(clazz, id):
    if False:
        for i in range(10):
            print('nop')
    model = clazz.objects.get(id=id)
    print(model)

def my_decorator_outside(func):
    if False:
        print('Hello World!')

    @wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            return 10
        print('outside before')
        func(*args, **kwargs)
        print('outside after')
    if getattr(func, 'delay', None):
        wrapper.delay = my_decorator_outside(func.delay)
    return wrapper

def my_decorator_inside(func):
    if False:
        print('Hello World!')

    @wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            return 10
        print('inside before')
        func(*args, **kwargs)
        print('inside after')
    return wrapper

@my_decorator_outside
@app.task
@my_decorator_inside
def my_test_task(new_finding, *args, **kwargs):
    if False:
        while True:
            i = 10
    print('oh la la what a nice task')

@dojo_model_to_id(parameter=1)
@dojo_model_to_id
@dojo_async_task
@app.task
@dojo_model_from_id(model=Notes, parameter=1)
@dojo_model_from_id
def test_valentijn_task(new_finding, note, **kwargs):
    if False:
        while True:
            i = 10
    logger.debug('test_valentijn:')
    logger.debug(new_finding)
    logger.debug(note)