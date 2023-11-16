from dagster import ScheduleDefinition, job, op, repository

@op
def do_something():
    if False:
        i = 10
        return i + 15
    return 1

@op
def do_input(x):
    if False:
        while True:
            i = 10
    return x

@job(name='foo')
def foo_job():
    if False:
        i = 10
        return i + 15
    do_input(do_something())

def define_foo_job():
    if False:
        print('Hello World!')
    return foo_job

@job(name='baz', description='Not much tbh')
def baz_job():
    if False:
        return 10
    do_input()

def define_bar_schedules():
    if False:
        print('Hello World!')
    return {'foo_schedule': ScheduleDefinition('foo_schedule', cron_schedule='* * * * *', job_name='foo', run_config={})}

@repository
def bar():
    if False:
        return 10
    return {'jobs': {'foo': foo_job, 'baz': baz_job}, 'schedules': define_bar_schedules()}