from dagster import job, op, repository

@op
def my_op():
    if False:
        return 10
    pass

@job
def my_job():
    if False:
        return 10
    my_op()

@repository
def my_other_repo():
    if False:
        print('Hello World!')
    return [my_job]