from dagster import job, op, repository

@op
def extra_op(_):
    if False:
        while True:
            i = 10
    pass

@job
def extra_job():
    if False:
        return 10
    extra_op()

@repository
def extra_repository():
    if False:
        for i in range(10):
            print('nop')
    return [extra_job]