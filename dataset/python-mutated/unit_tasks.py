from celery import shared_task

@shared_task
def mul(x, y):
    if False:
        print('Hello World!')
    return x * y