from dagster import In, String, dagster_type_loader, job, op, resource, usable_as_dagster_type

class UserError(Exception):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(UserError, self).__init__('The user has errored')

def test_user_error_boundary_op_compute():
    if False:
        print('Hello World!')

    @op
    def throws_user_error(_):
        if False:
            while True:
                i = 10
        raise UserError()

    @job
    def job_def():
        if False:
            while True:
                i = 10
        throws_user_error()
    result = job_def.execute_in_process(raise_on_error=False)
    assert not result.success

def test_user_error_boundary_input_hydration():
    if False:
        while True:
            i = 10

    @dagster_type_loader(String)
    def InputHydration(context, hello):
        if False:
            print('Hello World!')
        raise UserError()

    @usable_as_dagster_type(loader=InputHydration)
    class CustomType(str):
        pass

    @op(ins={'custom_type': In(CustomType)})
    def input_hydration_op(context, custom_type):
        if False:
            for i in range(10):
                print('nop')
        context.log.info(custom_type)

    @job
    def input_hydration_job():
        if False:
            while True:
                i = 10
        input_hydration_op()
    result = input_hydration_job.execute_in_process({'ops': {'input_hydration_op': {'inputs': {'custom_type': 'hello'}}}}, raise_on_error=False)
    assert not result.success

def test_user_error_boundary_resource_init():
    if False:
        i = 10
        return i + 15

    @resource
    def resource_a(_):
        if False:
            while True:
                i = 10
        raise UserError()

    @op(required_resource_keys={'a'})
    def resource_op(_context):
        if False:
            while True:
                i = 10
        return 'hello'

    @job(resource_defs={'a': resource_a})
    def resource_job():
        if False:
            return 10
        resource_op()
    result = resource_job.execute_in_process(raise_on_error=False)
    assert not result.success