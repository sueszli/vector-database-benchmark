import string
import time
from dagster import DynamicPartitionsDefinition, In, Int, Out, ScheduleDefinition, asset, define_asset_job, job, op, repository, schedule, usable_as_dagster_type
from dagster._core.definitions.asset_graph import AssetGraph
from dagster._core.definitions.decorators.sensor_decorator import sensor
from dagster._core.definitions.partition import PartitionedConfig, StaticPartitionsDefinition
from dagster._core.definitions.sensor_definition import RunRequest
from dagster._core.errors import DagsterError

@op
def do_something():
    if False:
        return 10
    return 1

@op
def do_input(x):
    if False:
        i = 10
        return i + 15
    return x

@job(name='foo')
def foo_job():
    if False:
        for i in range(10):
            print('nop')
    do_input(do_something())

@op
def forever_op():
    if False:
        while True:
            i = 10
    while True:
        time.sleep(10)

@job(name='forever')
def forever_job():
    if False:
        for i in range(10):
            print('nop')
    forever_op()

@op
def do_fail():
    if False:
        while True:
            i = 10
    raise Exception('I have failed')

@job
def fail_job():
    if False:
        i = 10
        return i + 15
    do_fail()
baz_partitions = StaticPartitionsDefinition(list(string.ascii_lowercase))
baz_config = PartitionedConfig(partitions_def=baz_partitions, run_config_for_partition_key_fn=lambda key: {'ops': {'do_input': {'inputs': {'x': {'value': key}}}}}, tags_for_partition_key_fn=lambda _partition: {'foo': 'bar'})

@job(name='baz', description='Not much tbh', partitions_def=baz_partitions, config=baz_config)
def baz_job():
    if False:
        for i in range(10):
            print('nop')
    do_input()
dynamic_partitions_def = DynamicPartitionsDefinition(name='dynamic_partitions')

@asset(partitions_def=dynamic_partitions_def)
def dynamic_asset():
    if False:
        print('Hello World!')
    return 1

def throw_error(_):
    if False:
        while True:
            i = 10
    raise Exception('womp womp')

def define_foo_job():
    if False:
        for i in range(10):
            print('nop')
    return foo_job

@job(name='other_foo')
def other_foo_job():
    if False:
        return 10
    do_input(do_something())

def define_other_foo_job():
    if False:
        print('Hello World!')
    return other_foo_job

@job(name='bar')
def bar_job():
    if False:
        i = 10
        return i + 15

    @usable_as_dagster_type(name='InputTypeWithoutHydration')
    class InputTypeWithoutHydration(int):
        pass

    @op(out=Out(InputTypeWithoutHydration))
    def one(_):
        if False:
            print('Hello World!')
        return 1

    @op(ins={'some_input': In(InputTypeWithoutHydration)}, out=Out(Int))
    def fail_subset(_, some_input):
        if False:
            i = 10
            return i + 15
        return some_input
    fail_subset(one())

@schedule(job_name='baz', cron_schedule='* * * * *')
def partitioned_run_request_schedule():
    if False:
        i = 10
        return i + 15
    return RunRequest(partition_key='a')

def define_bar_schedules():
    if False:
        return 10
    return {'foo_schedule': ScheduleDefinition('foo_schedule', cron_schedule='* * * * *', job_name='foo', run_config={'fizz': 'buzz'}), 'foo_schedule_never_execute': ScheduleDefinition('foo_schedule_never_execute', cron_schedule='* * * * *', job_name='foo', run_config={'fizz': 'buzz'}, should_execute=lambda _context: False), 'foo_schedule_echo_time': ScheduleDefinition('foo_schedule_echo_time', cron_schedule='* * * * *', job_name='foo', run_config_fn=lambda context: {'passed_in_time': context.scheduled_execution_time.isoformat() if context.scheduled_execution_time else ''}), 'partitioned_run_request_schedule': partitioned_run_request_schedule}

@sensor(job_name='foo')
def sensor_foo(_):
    if False:
        i = 10
        return i + 15
    yield RunRequest(run_key=None, run_config={'foo': 'FOO'}, tags={'foo': 'foo_tag'})
    yield RunRequest(run_key=None, run_config={'foo': 'FOO'})

@sensor(job_name='foo')
def sensor_error(_):
    if False:
        while True:
            i = 10
    raise Exception('womp womp')

@sensor(job_name='foo')
def sensor_raises_dagster_error(_):
    if False:
        while True:
            i = 10
    raise DagsterError('Dagster error')

@repository(metadata={'string': 'foo', 'integer': 123})
def bar_repo():
    if False:
        print('Hello World!')
    return {'jobs': {'bar': lambda : bar_job, 'baz': lambda : baz_job, 'dynamic_job': define_asset_job('dynamic_job', [dynamic_asset], partitions_def=dynamic_partitions_def).resolve(asset_graph=AssetGraph.from_assets([dynamic_asset])), 'fail': fail_job, 'foo': foo_job, 'forever': forever_job}, 'schedules': define_bar_schedules(), 'sensors': {'sensor_foo': sensor_foo, 'sensor_error': lambda : sensor_error, 'sensor_raises_dagster_error': lambda : sensor_raises_dagster_error}}

@repository
def other_repo():
    if False:
        i = 10
        return i + 15
    return {'jobs': {'other_foo': define_other_foo_job}}