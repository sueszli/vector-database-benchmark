from xyzzy import plugh
from celery import Celery, shared_task
app = Celery()

@app.task
def bar():
    if False:
        for i in range(10):
            print('nop')
    'Task.\n\n    This is a sample Task.\n    '

@shared_task
def baz():
    if False:
        return 10
    'Shared Task.\n\n    This is a sample Shared Task.\n    '