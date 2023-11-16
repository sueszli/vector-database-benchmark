from dagster import job, op, repository

@op
def my_op():
    if False:
        while True:
            i = 10
    pass

@job
def my_job():
    if False:
        print('Hello World!')
    my_op()

@repository
def my_repo():
    if False:
        for i in range(10):
            print('nop')
    return [my_job, my_job]