from dagster import job, op, repository, schedule

@op
def the_op():
    if False:
        for i in range(10):
            print('nop')
    pass

@job
def the_job():
    if False:
        while True:
            i = 10
    the_op()

@schedule(cron_schedule='0 2 * * *', job_name='the_job', execution_timezone='UTC')
def simple_schedule():
    if False:
        print('Hello World!')
    return {}

@repository
def the_repo():
    if False:
        while True:
            i = 10
    return [the_job, simple_schedule]