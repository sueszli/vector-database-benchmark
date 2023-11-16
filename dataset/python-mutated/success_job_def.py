from dagster import Definitions, job, op

@op
def an_op():
    if False:
        print('Hello World!')
    pass

@job
def success_job():
    if False:
        print('Hello World!')
    an_op()

@job
def another_success_job():
    if False:
        print('Hello World!')
    an_op()
defs = Definitions(jobs=[success_job, another_success_job])