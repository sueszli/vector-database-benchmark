from dagster_celery import celery_executor
from dagster import job, op

@op
def not_much():
    if False:
        print('Hello World!')
    return

@job(executor_def=celery_executor)
def parallel_job():
    if False:
        for i in range(10):
            print('nop')
    for i in range(50):
        not_much.alias('not_much_' + str(i))()