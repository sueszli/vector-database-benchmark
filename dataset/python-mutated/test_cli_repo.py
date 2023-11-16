from dagster import ScheduleDefinition, job, op, repository

@op
def add_one(num: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    return num + 1

@op
def mult_two(num: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    return num * 2

@job
def math():
    if False:
        print('Hello World!')
    mult_two(add_one())

@op(config_schema={'gimme': str})
def needs_config(context):
    if False:
        return 10
    return context.op_config['gimme']

@op
def no_config():
    if False:
        while True:
            i = 10
    return 'ok'

@job
def subset_test():
    if False:
        for i in range(10):
            print('nop')
    no_config()
    needs_config()

def define_schedules():
    if False:
        for i in range(10):
            print('nop')
    math_hourly_schedule = ScheduleDefinition(name='math_hourly_schedule', cron_schedule='0 0 * * *', job_name='math', run_config={'ops': {'add_one': {'inputs': {'num': {'value': 123}}}}})
    return [math_hourly_schedule]

@repository
def test():
    if False:
        for i in range(10):
            print('nop')
    return [math, subset_test] + define_schedules()