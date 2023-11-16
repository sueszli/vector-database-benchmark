from dagster import job, op, sensor

@op
def foo_op(_):
    if False:
        i = 10
        return i + 15
    return

@job
def foo_job():
    if False:
        while True:
            i = 10
    foo_op()

@sensor(job=foo_job)
def foo_sensor(_context):
    if False:
        for i in range(10):
            print('nop')
    return

def test_sensor_typechecks_when_defined_on_job_decorated_function():
    if False:
        while True:
            i = 10
    assert foo_sensor is not None