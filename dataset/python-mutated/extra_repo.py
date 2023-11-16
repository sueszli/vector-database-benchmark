from dagster import job, op, repository

@op
def do_something():
    if False:
        return 10
    return 1

@job
def extra_job():
    if False:
        return 10
    do_something()

@repository
def extra():
    if False:
        while True:
            i = 10
    return {'jobs': {'extra_job': extra_job}}