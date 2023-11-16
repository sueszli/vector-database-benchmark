import logging
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from typing import Any
from unittest import mock
import pendulum
import pytest
from dagster import AssetKey, AssetMaterialization, AssetObservation, AssetSelection, CodeLocationSelector, DagsterRunStatus, DailyPartitionsDefinition, DynamicPartitionsDefinition, Field, HourlyPartitionsDefinition, JobSelector, MultiPartitionKey, MultiPartitionsDefinition, Output, RepositorySelector, SourceAsset, StaticPartitionsDefinition, WeeklyPartitionsDefinition, asset, define_asset_job, load_assets_from_current_module, materialize, multi_asset_sensor, repository, run_failure_sensor
from dagster._core.definitions.asset_graph import AssetGraph
from dagster._core.definitions.decorators import op
from dagster._core.definitions.decorators.job_decorator import job
from dagster._core.definitions.decorators.sensor_decorator import asset_sensor, sensor
from dagster._core.definitions.instigation_logger import get_instigation_log_records
from dagster._core.definitions.run_request import InstigatorType, SensorResult
from dagster._core.definitions.run_status_sensor_definition import run_status_sensor
from dagster._core.definitions.sensor_definition import DefaultSensorStatus, RunRequest, SkipReason
from dagster._core.events import DagsterEventType
from dagster._core.host_representation import ExternalInstigatorOrigin, ExternalRepositoryOrigin
from dagster._core.host_representation.external import ExternalRepository
from dagster._core.host_representation.origin import ManagedGrpcPythonEnvCodeLocationOrigin
from dagster._core.instance import DagsterInstance
from dagster._core.log_manager import DAGSTER_META_KEY
from dagster._core.scheduler.instigation import DynamicPartitionsRequestResult, InstigatorState, InstigatorStatus, TickStatus
from dagster._core.storage.event_log.base import EventRecordsFilter
from dagster._core.test_utils import BlockingThreadPoolExecutor, create_test_daemon_workspace_context, instance_for_test, wait_for_futures
from dagster._core.workspace.context import WorkspaceProcessContext
from dagster._daemon import get_default_daemon_logger
from dagster._daemon.sensor import execute_sensor_iteration, execute_sensor_iteration_loop
from dagster._seven.compat.pendulum import create_pendulum_time, to_timezone
from .conftest import create_workspace_load_target

@asset
def a():
    if False:
        while True:
            i = 10
    return 1

@asset
def b(a):
    if False:
        for i in range(10):
            print('nop')
    return a + 1

@asset
def c(a):
    if False:
        print('Hello World!')
    return a + 2
asset_job = define_asset_job('abc', selection=AssetSelection.keys('c', 'b').upstream())

@op
def the_op(_):
    if False:
        print('Hello World!')
    return 1

@job
def the_job():
    if False:
        while True:
            i = 10
    the_op()

@job
def the_other_job():
    if False:
        return 10
    the_op()

@op(config_schema=Field(Any))
def config_op(_):
    if False:
        while True:
            i = 10
    return 1

@job
def config_job():
    if False:
        while True:
            i = 10
    config_op()

@op
def foo_op():
    if False:
        return 10
    yield AssetMaterialization(asset_key=AssetKey('foo'))
    yield Output(1)

@job
def foo_job():
    if False:
        i = 10
        return i + 15
    foo_op()

@op
def foo_observation_op():
    if False:
        while True:
            i = 10
    yield AssetObservation(asset_key=AssetKey('foo'), metadata={'text': 'FOO'})
    yield Output(5)

@job
def foo_observation_job():
    if False:
        for i in range(10):
            print('nop')
    foo_observation_op()

@op
def hanging_op():
    if False:
        return 10
    start_time = time.time()
    while True:
        if time.time() - start_time > 10:
            return
        time.sleep(0.5)

@job
def hanging_job():
    if False:
        while True:
            i = 10
    hanging_op()

@op
def failure_op():
    if False:
        for i in range(10):
            print('nop')
    raise Exception('womp womp')

@job
def failure_job():
    if False:
        i = 10
        return i + 15
    failure_op()

@job
def failure_job_2():
    if False:
        return 10
    failure_op()

@sensor(job_name='the_job')
def simple_sensor(context):
    if False:
        print('Hello World!')
    if not context.last_completion_time or not int(context.last_completion_time) % 2:
        return SkipReason()
    return RunRequest(run_key=None, run_config={}, tags={})

@sensor(job_name='the_job')
def always_on_sensor(_context):
    if False:
        while True:
            i = 10
    return RunRequest(run_key=None, run_config={}, tags={})

@sensor(job_name='the_job')
def run_key_sensor(_context):
    if False:
        for i in range(10):
            print('nop')
    return RunRequest(run_key='only_once', run_config={}, tags={})

@sensor(job_name='the_job')
def error_sensor(context):
    if False:
        return 10
    context.update_cursor('the exception below should keep this from being persisted')
    raise Exception('womp womp')

@sensor(job_name='the_job')
def wrong_config_sensor(_context):
    if False:
        for i in range(10):
            print('nop')
    return RunRequest(run_key='bad_config_key', run_config={'bad_key': 'bad_val'}, tags={})

@sensor(job_name='the_job', minimum_interval_seconds=60)
def custom_interval_sensor(_context):
    if False:
        for i in range(10):
            print('nop')
    return SkipReason()

@sensor(job_name='the_job')
def skip_cursor_sensor(context):
    if False:
        while True:
            i = 10
    if not context.cursor:
        cursor = 1
    else:
        cursor = int(context.cursor) + 1
    context.update_cursor(str(cursor))
    return SkipReason()

@sensor(job_name='the_job')
def run_cursor_sensor(context):
    if False:
        print('Hello World!')
    if not context.cursor:
        cursor = 1
    else:
        cursor = int(context.cursor) + 1
    context.update_cursor(str(cursor))
    return RunRequest(run_key=None, run_config={}, tags={})

@asset
def asset_a():
    if False:
        return 10
    return 1

@asset
def asset_b():
    if False:
        for i in range(10):
            print('nop')
    return 2

@asset
def asset_c(asset_b):
    if False:
        print('Hello World!')
    return 3

@multi_asset_sensor(monitored_assets=[AssetKey('asset_a'), AssetKey('asset_b')], job=the_job)
def asset_a_and_b_sensor(context):
    if False:
        return 10
    asset_events = context.latest_materialization_records_by_key()
    if all(asset_events.values()):
        context.advance_all_cursors()
        return RunRequest(run_key=f'{context.cursor}', run_config={})

@multi_asset_sensor(monitored_assets=[AssetKey('asset_a'), AssetKey('asset_b')], job=the_job)
def doesnt_update_cursor_sensor(context):
    if False:
        i = 10
        return i + 15
    asset_events = context.latest_materialization_records_by_key()
    if any(asset_events.values()):
        return RunRequest(run_key=f'{context.cursor}', run_config={})

@multi_asset_sensor(monitored_assets=[AssetKey('asset_a')], job=the_job)
def backlog_sensor(context):
    if False:
        print('Hello World!')
    asset_events = context.materialization_records_for_key(asset_key=AssetKey('asset_a'), limit=2)
    if len(asset_events) == 2:
        context.advance_cursor({AssetKey('asset_a'): asset_events[-1]})
        return RunRequest(run_key=f'{context.cursor}', run_config={})

@multi_asset_sensor(monitored_assets=AssetSelection.keys('asset_c').upstream(include_self=False))
def asset_selection_sensor(context):
    if False:
        print('Hello World!')
    assert context.asset_keys == [AssetKey('asset_b')]
    assert context.latest_materialization_records_by_key().keys() == {AssetKey('asset_b')}

@sensor(asset_selection=AssetSelection.keys('asset_a', 'asset_b'))
def targets_asset_selection_sensor():
    if False:
        i = 10
        return i + 15
    return [RunRequest(), RunRequest(asset_selection=[AssetKey('asset_b')])]

@multi_asset_sensor(monitored_assets=AssetSelection.keys('asset_b'), request_assets=AssetSelection.keys('asset_c'))
def multi_asset_sensor_targets_asset_selection(context):
    if False:
        for i in range(10):
            print('nop')
    asset_events = context.latest_materialization_records_by_key()
    if all(asset_events.values()):
        context.advance_all_cursors()
        return RunRequest()
hourly_partitions_def_2022 = HourlyPartitionsDefinition(start_date='2022-08-01-00:00')

@asset(partitions_def=hourly_partitions_def_2022)
def hourly_asset():
    if False:
        return 10
    return 1

@asset(partitions_def=hourly_partitions_def_2022)
def hourly_asset_2():
    if False:
        print('Hello World!')
    return 1

@asset(partitions_def=hourly_partitions_def_2022)
def hourly_asset_3():
    if False:
        return 10
    return 1
hourly_asset_job = define_asset_job('hourly_asset_job', AssetSelection.keys('hourly_asset_3'), partitions_def=hourly_partitions_def_2022)
weekly_partitions_def = WeeklyPartitionsDefinition(start_date='2020-01-01')

@asset(partitions_def=weekly_partitions_def)
def weekly_asset():
    if False:
        return 10
    return 1
weekly_asset_job = define_asset_job('weekly_asset_job', AssetSelection.keys('weekly_asset'), partitions_def=weekly_partitions_def)

@multi_asset_sensor(monitored_assets=[hourly_asset.key], job=weekly_asset_job)
def multi_asset_sensor_hourly_to_weekly(context):
    if False:
        i = 10
        return i + 15
    for (partition, materialization) in context.latest_materialization_records_by_partition(hourly_asset.key).items():
        mapped_partitions = context.get_downstream_partition_keys(partition, to_asset_key=weekly_asset.key, from_asset_key=hourly_asset.key)
        for mapped_partition in mapped_partitions:
            yield weekly_asset_job.run_request_for_partition(partition_key=mapped_partition, run_key=None)
        context.advance_cursor({hourly_asset.key: materialization})

@multi_asset_sensor(monitored_assets=[hourly_asset.key], job=hourly_asset_job)
def multi_asset_sensor_hourly_to_hourly(context):
    if False:
        for i in range(10):
            print('nop')
    materialization_by_partition = context.latest_materialization_records_by_partition(hourly_asset.key)
    latest_partition = None
    for (partition, materialization) in materialization_by_partition.items():
        if materialization:
            mapped_partitions = context.get_downstream_partition_keys(partition, to_asset_key=hourly_asset_3.key, from_asset_key=hourly_asset.key)
            for mapped_partition in mapped_partitions:
                yield hourly_asset_job.run_request_for_partition(partition_key=mapped_partition, run_key=None)
            latest_partition = partition if latest_partition is None else max(latest_partition, partition)
    if latest_partition:
        context.advance_cursor({hourly_asset.key: materialization_by_partition[latest_partition]})

@multi_asset_sensor(monitored_assets=[AssetKey('asset_a'), AssetKey('asset_b')], job=the_job)
def sensor_result_multi_asset_sensor(context):
    if False:
        while True:
            i = 10
    context.advance_all_cursors()
    return SensorResult([RunRequest('foo')])

@multi_asset_sensor(monitored_assets=[AssetKey('asset_a'), AssetKey('asset_b')], job=the_job)
def cursor_sensor_result_multi_asset_sensor(context):
    if False:
        while True:
            i = 10
    return SensorResult([RunRequest('foo')], cursor='foo')

def _random_string(length):
    if False:
        i = 10
        return i + 15
    return ''.join((random.choice(string.ascii_lowercase) for x in range(length)))

@sensor(job_name='config_job')
def large_sensor(_context):
    if False:
        print('Hello World!')
    REQUEST_COUNT = 25
    REQUEST_TAG_COUNT = 5000
    REQUEST_CONFIG_COUNT = 100
    for _ in range(REQUEST_COUNT):
        tags_garbage = {_random_string(10): _random_string(20) for i in range(REQUEST_TAG_COUNT)}
        config_garbage = {_random_string(10): _random_string(20) for i in range(REQUEST_CONFIG_COUNT)}
        config = {'ops': {'config_op': {'config': {'foo': config_garbage}}}}
        yield RunRequest(run_key=None, run_config=config, tags=tags_garbage)

@sensor(job_name='config_job')
def many_request_sensor(_context):
    if False:
        print('Hello World!')
    REQUEST_COUNT = 15
    for _ in range(REQUEST_COUNT):
        config = {'ops': {'config_op': {'config': {'foo': 'bar'}}}}
        yield RunRequest(run_key=None, run_config=config)

@sensor(job=asset_job)
def run_request_asset_selection_sensor(_context):
    if False:
        i = 10
        return i + 15
    yield RunRequest(run_key=None, asset_selection=[AssetKey('a'), AssetKey('b')])

@sensor(job=asset_job)
def run_request_stale_asset_sensor(_context):
    if False:
        i = 10
        return i + 15
    yield RunRequest(run_key=None, stale_assets_only=True)

@sensor(job=hourly_asset_job)
def partitioned_asset_selection_sensor(_context):
    if False:
        return 10
    return hourly_asset_job.run_request_for_partition(partition_key='2022-08-01-00:00', run_key=None, asset_selection=[AssetKey('hourly_asset_3')])

@asset_sensor(job_name='the_job', asset_key=AssetKey('foo'))
def asset_foo_sensor(context, _event):
    if False:
        for i in range(10):
            print('nop')
    return RunRequest(run_key=context.cursor, run_config={})

@asset_sensor(asset_key=AssetKey('foo'), job=the_job)
def asset_job_sensor(context, _event):
    if False:
        while True:
            i = 10
    return RunRequest(run_key=context.cursor, run_config={})

@run_failure_sensor
def my_run_failure_sensor(context):
    if False:
        return 10
    assert isinstance(context.instance, DagsterInstance)
    if 'failure_op' in context.failure_event.message:
        step_failure_events = context.get_step_failure_events()
        assert len(step_failure_events) == 1
        step_error_str = step_failure_events[0].event_specific_data.error.to_string()
        assert 'womp womp' in step_error_str, step_error_str

@run_failure_sensor(job_selection=[failure_job])
def my_run_failure_sensor_filtered(context):
    if False:
        i = 10
        return i + 15
    assert isinstance(context.instance, DagsterInstance)

@run_failure_sensor()
def my_run_failure_sensor_that_itself_fails(context):
    if False:
        while True:
            i = 10
    raise Exception('How meta')

@run_status_sensor(run_status=DagsterRunStatus.SUCCESS)
def my_job_success_sensor(context):
    if False:
        return 10
    assert isinstance(context.instance, DagsterInstance)

@run_status_sensor(run_status=DagsterRunStatus.STARTED)
def my_job_started_sensor(context):
    if False:
        return 10
    assert isinstance(context.instance, DagsterInstance)

@sensor(jobs=[the_job, config_job])
def two_job_sensor(context):
    if False:
        return 10
    counter = int(context.cursor) if context.cursor else 0
    if counter % 2 == 0:
        yield RunRequest(run_key=str(counter), job_name=the_job.name)
    else:
        yield RunRequest(run_key=str(counter), job_name=config_job.name, run_config={'ops': {'config_op': {'config': {'foo': 'blah'}}}})
    context.update_cursor(str(counter + 1))

@sensor()
def bad_request_untargeted(_ctx):
    if False:
        while True:
            i = 10
    yield RunRequest(run_key=None, job_name='should_fail')

@sensor(job=the_job)
def bad_request_mismatch(_ctx):
    if False:
        for i in range(10):
            print('nop')
    yield RunRequest(run_key=None, job_name='config_job')

@sensor(jobs=[the_job, config_job])
def bad_request_unspecified(_ctx):
    if False:
        while True:
            i = 10
    yield RunRequest(run_key=None)

@sensor(job=the_job)
def request_list_sensor(_ctx):
    if False:
        i = 10
        return i + 15
    return [RunRequest(run_key='1'), RunRequest(run_key='2')]

@run_status_sensor(monitored_jobs=[JobSelector(location_name='test_location', repository_name='the_other_repo', job_name='the_job')], run_status=DagsterRunStatus.SUCCESS, request_job=the_other_job)
def cross_repo_job_sensor():
    if False:
        print('Hello World!')
    from time import time
    return RunRequest(run_key=str(time()))

@run_status_sensor(monitored_jobs=[RepositorySelector(location_name='test_location', repository_name='the_other_repo')], run_status=DagsterRunStatus.SUCCESS)
def cross_repo_sensor(context):
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(context.instance, DagsterInstance)

@run_status_sensor(monitor_all_repositories=True, run_status=DagsterRunStatus.SUCCESS)
def instance_sensor():
    if False:
        for i in range(10):
            print('nop')
    pass

@sensor(job=the_job)
def logging_sensor(context):
    if False:
        print('Hello World!')

    class Handler(logging.Handler):

        def handle(self, record):
            if False:
                i = 10
                return i + 15
            try:
                self.message = record.getMessage()
            except TypeError:
                self.message = 'error'
    handler = Handler()
    context.log.addHandler(handler)
    context.log.info('hello %s', 'hello')
    context.log.info(handler.message)
    context.log.removeHandler(handler)
    return SkipReason()

@run_status_sensor(monitor_all_repositories=True, run_status=DagsterRunStatus.SUCCESS)
def logging_status_sensor(context):
    if False:
        print('Hello World!')
    context.log.info(f'run succeeded: {context.dagster_run.run_id}')
quux = DynamicPartitionsDefinition(name='quux')

@asset(partitions_def=quux)
def quux_asset(context):
    if False:
        i = 10
        return i + 15
    return 1
quux_asset_job = define_asset_job('quux_asset_job', [quux_asset], partitions_def=quux)

@sensor()
def add_dynamic_partitions_sensor(context):
    if False:
        while True:
            i = 10
    return SensorResult(dynamic_partitions_requests=[quux.build_add_request(['baz', 'foo'])])

@sensor(job=quux_asset_job)
def add_delete_dynamic_partitions_and_yield_run_requests_sensor(context):
    if False:
        return 10
    return SensorResult(dynamic_partitions_requests=[quux.build_add_request(['1']), quux.build_delete_request(['2', '3'])], run_requests=[RunRequest(partition_key='1')])

@sensor(job=quux_asset_job)
def error_on_deleted_dynamic_partitions_run_requests_sensor(context):
    if False:
        for i in range(10):
            print('nop')
    return SensorResult(dynamic_partitions_requests=[quux.build_delete_request(['2'])], run_requests=[RunRequest(partition_key='2')])
dynamic1 = DynamicPartitionsDefinition(name='dynamic1')
dynamic2 = DynamicPartitionsDefinition(name='dynamic2')

@asset(partitions_def=MultiPartitionsDefinition({'dynamic1': dynamic1, 'dynamic2': dynamic2}))
def multipartitioned_with_two_dynamic_dims():
    if False:
        for i in range(10):
            print('nop')
    pass

@sensor(asset_selection=AssetSelection.keys(multipartitioned_with_two_dynamic_dims.key))
def success_on_multipartition_run_request_with_two_dynamic_dimensions_sensor(context):
    if False:
        for i in range(10):
            print('nop')
    return SensorResult(dynamic_partitions_requests=[dynamic1.build_add_request(['1']), dynamic2.build_add_request(['2'])], run_requests=[RunRequest(partition_key=MultiPartitionKey({'dynamic1': '1', 'dynamic2': '2'}))])

@sensor(asset_selection=AssetSelection.keys(multipartitioned_with_two_dynamic_dims.key))
def error_on_multipartition_run_request_with_two_dynamic_dimensions_sensor(context):
    if False:
        print('Hello World!')
    return SensorResult(dynamic_partitions_requests=[dynamic1.build_add_request(['1']), dynamic2.build_add_request(['2'])], run_requests=[RunRequest(partition_key=MultiPartitionKey({'dynamic1': '2', 'dynamic2': '1'}))])

@asset(partitions_def=MultiPartitionsDefinition({'static': StaticPartitionsDefinition(['a', 'b', 'c']), 'time': DailyPartitionsDefinition('2023-01-01')}))
def multipartitioned_asset_with_static_time_dimensions():
    if False:
        for i in range(10):
            print('nop')
    pass

@sensor(asset_selection=AssetSelection.keys(multipartitioned_asset_with_static_time_dimensions.key))
def multipartitions_with_static_time_dimensions_run_requests_sensor(context):
    if False:
        for i in range(10):
            print('nop')
    return SensorResult(run_requests=[RunRequest(partition_key=MultiPartitionKey({'static': 'b', 'time': '2023-01-05'}))])
daily_partitions_def = DailyPartitionsDefinition(start_date='2022-08-01')

@asset(partitions_def=daily_partitions_def)
def partitioned_asset():
    if False:
        while True:
            i = 10
    return 1
daily_partitioned_job = define_asset_job('daily_partitioned_job', partitions_def=daily_partitions_def).resolve(asset_graph=AssetGraph.from_assets([partitioned_asset]))

@run_status_sensor(run_status=DagsterRunStatus.SUCCESS, monitored_jobs=[daily_partitioned_job])
def partitioned_pipeline_success_sensor(_context):
    if False:
        i = 10
        return i + 15
    assert _context.partition_key == '2022-08-01'

@repository
def the_repo():
    if False:
        while True:
            i = 10
    return [the_job, the_other_job, config_job, foo_job, large_sensor, many_request_sensor, simple_sensor, error_sensor, wrong_config_sensor, always_on_sensor, run_key_sensor, custom_interval_sensor, skip_cursor_sensor, run_cursor_sensor, asset_foo_sensor, asset_job_sensor, my_run_failure_sensor, my_run_failure_sensor_filtered, my_run_failure_sensor_that_itself_fails, my_job_success_sensor, my_job_started_sensor, failure_job, failure_job_2, hanging_job, two_job_sensor, bad_request_untargeted, bad_request_mismatch, bad_request_unspecified, request_list_sensor, asset_a_and_b_sensor, doesnt_update_cursor_sensor, backlog_sensor, cross_repo_sensor, cross_repo_job_sensor, instance_sensor, load_assets_from_current_module(), run_request_asset_selection_sensor, run_request_stale_asset_sensor, weekly_asset_job, multi_asset_sensor_hourly_to_weekly, multi_asset_sensor_hourly_to_hourly, sensor_result_multi_asset_sensor, cursor_sensor_result_multi_asset_sensor, partitioned_asset_selection_sensor, asset_selection_sensor, targets_asset_selection_sensor, multi_asset_sensor_targets_asset_selection, logging_sensor, logging_status_sensor, add_delete_dynamic_partitions_and_yield_run_requests_sensor, add_dynamic_partitions_sensor, quux_asset_job, error_on_deleted_dynamic_partitions_run_requests_sensor, partitioned_pipeline_success_sensor, daily_partitioned_job, success_on_multipartition_run_request_with_two_dynamic_dimensions_sensor, error_on_multipartition_run_request_with_two_dynamic_dimensions_sensor, multipartitions_with_static_time_dimensions_run_requests_sensor]

@repository
def the_other_repo():
    if False:
        for i in range(10):
            print('nop')
    return [the_job, run_key_sensor]

@sensor(job_name='the_job', default_status=DefaultSensorStatus.RUNNING)
def always_running_sensor(context):
    if False:
        print('Hello World!')
    if not context.last_completion_time or not int(context.last_completion_time) % 2:
        return SkipReason()
    return RunRequest(run_key=None, run_config={}, tags={})

@sensor(job_name='the_job', default_status=DefaultSensorStatus.STOPPED)
def never_running_sensor(context):
    if False:
        while True:
            i = 10
    if not context.last_completion_time or not int(context.last_completion_time) % 2:
        return SkipReason()
    return RunRequest(run_key=None, run_config={}, tags={})

@repository
def the_status_in_code_repo():
    if False:
        while True:
            i = 10
    return [the_job, always_running_sensor, never_running_sensor]

@asset
def x():
    if False:
        while True:
            i = 10
    return 1

@asset
def y(x):
    if False:
        i = 10
        return i + 15
    return x + 1

@asset
def z():
    if False:
        while True:
            i = 10
    return 2

@asset
def d(x, z):
    if False:
        print('Hello World!')
    return x + z

@asset
def e():
    if False:
        i = 10
        return i + 15
    return 3

@asset
def f(z, e):
    if False:
        for i in range(10):
            print('nop')
    return z + e

@asset
def g(d, f):
    if False:
        for i in range(10):
            print('nop')
    return d + f

@asset
def h():
    if False:
        i = 10
        return i + 15
    return 1

@asset
def i(h):
    if False:
        i = 10
        return i + 15
    return h + 1

@asset
def sleeper():
    if False:
        print('Hello World!')
    from time import sleep
    sleep(30)
    return 1

@asset
def waits_on_sleep(sleeper, x):
    if False:
        for i in range(10):
            print('nop')
    return sleeper + x

@asset
def a_source_asset():
    if False:
        return 10
    return 1
source_asset_source = SourceAsset(key=AssetKey('a_source_asset'))

@asset
def depends_on_source(a_source_asset):
    if False:
        while True:
            i = 10
    return a_source_asset + 1

@repository
def with_source_asset_repo():
    if False:
        print('Hello World!')
    return [a_source_asset]

@multi_asset_sensor(monitored_assets=[AssetKey('a_source_asset')], job=the_job)
def monitor_source_asset_sensor(context):
    if False:
        print('Hello World!')
    asset_events = context.latest_materialization_records_by_key()
    if all(asset_events.values()):
        context.advance_all_cursors()
        return RunRequest(run_key=f'{context.cursor}', run_config={})

@repository
def asset_sensor_repo():
    if False:
        i = 10
        return i + 15
    return [x, y, z, d, e, f, g, h, i, sleeper, waits_on_sleep, source_asset_source, depends_on_source, the_job, monitor_source_asset_sensor]
FUTURES_TIMEOUT = 75

def evaluate_sensors(workspace_context, executor, submit_executor=None, timeout=FUTURES_TIMEOUT):
    if False:
        i = 10
        return i + 15
    logger = get_default_daemon_logger('SensorDaemon')
    futures = {}
    list(execute_sensor_iteration(workspace_context, logger, threadpool_executor=executor, sensor_tick_futures=futures, submit_threadpool_executor=submit_executor))
    wait_for_futures(futures, timeout=timeout)

def validate_tick(tick, external_sensor, expected_datetime, expected_status, expected_run_ids=None, expected_error=None):
    if False:
        for i in range(10):
            print('nop')
    tick_data = tick.tick_data
    assert tick_data.instigator_origin_id == external_sensor.get_external_origin_id()
    assert tick_data.instigator_name == external_sensor.name
    assert tick_data.instigator_type == InstigatorType.SENSOR
    assert tick_data.status == expected_status, tick_data.error
    if expected_datetime:
        assert tick_data.timestamp == expected_datetime.timestamp()
    if expected_run_ids is not None:
        assert set(tick_data.run_ids) == set(expected_run_ids)
    if expected_error:
        assert expected_error in str(tick_data.error)

def validate_run_started(run, expected_success=True):
    if False:
        print('Hello World!')
    if expected_success:
        assert run.status == DagsterRunStatus.STARTED or run.status == DagsterRunStatus.SUCCESS or run.status == DagsterRunStatus.STARTING
    else:
        assert run.status == DagsterRunStatus.FAILURE

def wait_for_all_runs_to_start(instance, timeout=10):
    if False:
        print('Hello World!')
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            raise Exception('Timed out waiting for runs to start')
        time.sleep(0.5)
        not_started_runs = [run for run in instance.get_runs() if run.status == DagsterRunStatus.NOT_STARTED]
        if len(not_started_runs) == 0:
            break

def wait_for_all_runs_to_finish(instance, timeout=10):
    if False:
        return 10
    start_time = time.time()
    FINISHED_STATES = [DagsterRunStatus.SUCCESS, DagsterRunStatus.FAILURE, DagsterRunStatus.CANCELED]
    while True:
        if time.time() - start_time > timeout:
            raise Exception('Timed out waiting for runs to finish')
        time.sleep(0.5)
        not_finished_runs = [run for run in instance.get_runs() if run.status not in FINISHED_STATES]
        if len(not_finished_runs) == 0:
            break

def test_simple_sensor(instance, workspace_context, external_repo, executor):
    if False:
        i = 10
        return i + 15
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, hour=23, minute=59, second=59, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo.get_external_sensor('simple_sensor')
        instance.add_instigator_state(InstigatorState(external_sensor.get_external_origin(), InstigatorType.SENSOR, InstigatorStatus.RUNNING))
        assert instance.get_runs_count() == 0
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 0
        evaluate_sensors(workspace_context, executor)
        assert instance.get_runs_count() == 0
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], external_sensor, freeze_datetime, TickStatus.SKIPPED)
        freeze_datetime = freeze_datetime.add(seconds=30)
    with pendulum.test(freeze_datetime):
        evaluate_sensors(workspace_context, executor)
        wait_for_all_runs_to_start(instance)
        assert instance.get_runs_count() == 1
        run = instance.get_runs()[0]
        validate_run_started(run)
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 2
        expected_datetime = create_pendulum_time(year=2019, month=2, day=28, hour=0, minute=0, second=29)
        validate_tick(ticks[0], external_sensor, expected_datetime, TickStatus.SUCCESS, [run.run_id])

def test_sensors_keyed_on_selector_not_origin(instance: DagsterInstance, workspace_context: WorkspaceProcessContext, external_repo: ExternalRepository, executor: ThreadPoolExecutor):
    if False:
        while True:
            i = 10
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, hour=23, minute=59, second=59, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo.get_external_sensor('simple_sensor')
        existing_origin = external_sensor.get_external_origin()
        code_location_origin = existing_origin.external_repository_origin.code_location_origin
        assert isinstance(code_location_origin, ManagedGrpcPythonEnvCodeLocationOrigin)
        modified_loadable_target_origin = code_location_origin.loadable_target_origin._replace(executable_path='/different/executable_path')
        modified_origin = existing_origin._replace(external_repository_origin=existing_origin.external_repository_origin._replace(code_location_origin=code_location_origin._replace(loadable_target_origin=modified_loadable_target_origin)))
        instance.add_instigator_state(InstigatorState(modified_origin, InstigatorType.SENSOR, InstigatorStatus.RUNNING))
        evaluate_sensors(workspace_context, executor)
        assert instance.get_runs_count() == 0
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 1

def test_bad_load_sensor_repository(caplog: pytest.LogCaptureFixture, executor: ThreadPoolExecutor, instance: DagsterInstance, workspace_context: WorkspaceProcessContext, external_repo: ExternalRepository):
    if False:
        for i in range(10):
            print('nop')
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, hour=23, minute=59, second=59, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo.get_external_sensor('simple_sensor')
        valid_origin = external_sensor.get_external_origin()
        invalid_repo_origin = ExternalInstigatorOrigin(ExternalRepositoryOrigin(valid_origin.external_repository_origin.code_location_origin, 'invalid_repo_name'), valid_origin.instigator_name)
        invalid_state = instance.add_instigator_state(InstigatorState(invalid_repo_origin, InstigatorType.SENSOR, InstigatorStatus.RUNNING))
        assert instance.get_runs_count() == 0
        ticks = instance.get_ticks(invalid_state.instigator_origin_id, invalid_state.selector_id)
        assert len(ticks) == 0
        evaluate_sensors(workspace_context, executor)
        assert instance.get_runs_count() == 0
        ticks = instance.get_ticks(invalid_state.instigator_origin_id, invalid_state.selector_id)
        assert len(ticks) == 0
        assert 'Could not find repository invalid_repo_name in location test_location to run sensor simple_sensor' in caplog.text

def test_bad_load_sensor(caplog, executor, instance, workspace_context, external_repo):
    if False:
        while True:
            i = 10
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, hour=23, minute=59, second=59, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo.get_external_sensor('simple_sensor')
        valid_origin = external_sensor.get_external_origin()
        invalid_repo_origin = ExternalInstigatorOrigin(valid_origin.external_repository_origin, 'invalid_sensor')
        invalid_state = instance.add_instigator_state(InstigatorState(invalid_repo_origin, InstigatorType.SENSOR, InstigatorStatus.RUNNING))
        assert instance.get_runs_count() == 0
        ticks = instance.get_ticks(invalid_state.instigator_origin_id, invalid_state.selector_id)
        assert len(ticks) == 0
        evaluate_sensors(workspace_context, executor)
        assert instance.get_runs_count() == 0
        ticks = instance.get_ticks(invalid_state.instigator_origin_id, invalid_state.selector_id)
        assert len(ticks) == 0
        assert 'Could not find sensor invalid_sensor in repository the_repo.' in caplog.text

def test_error_sensor(caplog, executor, instance, workspace_context, external_repo):
    if False:
        return 10
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, hour=23, minute=59, second=59, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo.get_external_sensor('error_sensor')
        instance.add_instigator_state(InstigatorState(external_sensor.get_external_origin(), InstigatorType.SENSOR, InstigatorStatus.RUNNING))
        state = instance.get_instigator_state(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert state.instigator_data is None
        assert instance.get_runs_count() == 0
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 0
        evaluate_sensors(workspace_context, executor)
        assert instance.get_runs_count() == 0
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], external_sensor, freeze_datetime, TickStatus.FAILURE, [], 'Error occurred during the execution of evaluation_fn for sensor error_sensor')
        assert 'Error occurred during the execution of evaluation_fn for sensor error_sensor' in caplog.text
        state = instance.get_instigator_state(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert state.instigator_data.cursor is None
        assert state.instigator_data.last_tick_timestamp == freeze_datetime.timestamp()

def test_wrong_config_sensor(caplog, executor, instance, workspace_context, external_repo):
    if False:
        while True:
            i = 10
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, hour=23, minute=59, second=59), 'US/Central')
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo.get_external_sensor('wrong_config_sensor')
        instance.add_instigator_state(InstigatorState(external_sensor.get_external_origin(), InstigatorType.SENSOR, InstigatorStatus.RUNNING))
        assert instance.get_runs_count() == 0
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 0
        evaluate_sensors(workspace_context, executor)
        assert instance.get_runs_count() == 0
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], external_sensor, freeze_datetime, TickStatus.FAILURE, [], 'Error in config for job')
        assert 'Error in config for job' in caplog.text
    freeze_datetime = freeze_datetime.add(seconds=60)
    caplog.clear()
    with pendulum.test(freeze_datetime):
        evaluate_sensors(workspace_context, executor)
        assert instance.get_runs_count() == 0
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 2
        validate_tick(ticks[0], external_sensor, freeze_datetime, TickStatus.FAILURE, [], 'Error in config for job')
        assert 'Error in config for job' in caplog.text

def test_launch_failure(caplog, executor, workspace_context, external_repo):
    if False:
        print('Hello World!')
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, hour=23, minute=59, second=59, tz='UTC'), 'US/Central')
    with instance_for_test(overrides={'run_launcher': {'module': 'dagster._core.test_utils', 'class': 'ExplodingRunLauncher'}}) as instance:
        with pendulum.test(freeze_datetime):
            exploding_workspace_context = workspace_context.copy_for_test_instance(instance)
            external_sensor = external_repo.get_external_sensor('always_on_sensor')
            instance.add_instigator_state(InstigatorState(external_sensor.get_external_origin(), InstigatorType.SENSOR, InstigatorStatus.RUNNING))
            assert instance.get_runs_count() == 0
            ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
            assert len(ticks) == 0
            evaluate_sensors(exploding_workspace_context, executor)
            assert instance.get_runs_count() == 1
            run = instance.get_runs()[0]
            ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
            assert len(ticks) == 1
            validate_tick(ticks[0], external_sensor, freeze_datetime, TickStatus.SUCCESS, [run.run_id])
            assert f'Run {run.run_id} created successfully but failed to launch:' in caplog.text
            assert 'The entire purpose of this is to throw on launch' in caplog.text

def test_launch_once(caplog, executor, instance, workspace_context, external_repo):
    if False:
        return 10
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, hour=23, minute=59, second=59, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo.get_external_sensor('run_key_sensor')
        instance.add_instigator_state(InstigatorState(external_sensor.get_external_origin(), InstigatorType.SENSOR, InstigatorStatus.RUNNING))
        assert instance.get_runs_count() == 0
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 0
        evaluate_sensors(workspace_context, executor)
        wait_for_all_runs_to_start(instance)
        assert instance.get_runs_count() == 1
        run = instance.get_runs()[0]
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], external_sensor, freeze_datetime, TickStatus.SUCCESS, expected_run_ids=[run.run_id])
    freeze_datetime = freeze_datetime.add(seconds=30)
    with pendulum.test(freeze_datetime):
        evaluate_sensors(workspace_context, executor)
        assert instance.get_runs_count() == 1
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 2
        validate_tick(ticks[0], external_sensor, freeze_datetime, TickStatus.SKIPPED)
        assert ticks[0].run_keys
        assert len(ticks[0].run_keys) == 1
        assert not ticks[0].run_ids
        assert 'Skipping 1 run for sensor run_key_sensor already completed with run keys: ["only_once"]' in caplog.text
        launched_run = instance.get_runs()[0]
        the_job.execute_in_process(run_config=launched_run.run_config, tags=launched_run.tags, instance=instance)
    freeze_datetime = freeze_datetime.add(seconds=30)
    with pendulum.test(freeze_datetime):
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 3
        validate_tick(ticks[0], external_sensor, freeze_datetime, TickStatus.SKIPPED)

def test_custom_interval_sensor(executor, instance, workspace_context, external_repo):
    if False:
        print('Hello World!')
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=28, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo.get_external_sensor('custom_interval_sensor')
        instance.add_instigator_state(InstigatorState(external_sensor.get_external_origin(), InstigatorType.SENSOR, InstigatorStatus.RUNNING))
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 0
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], external_sensor, freeze_datetime, TickStatus.SKIPPED)
        freeze_datetime = freeze_datetime.add(seconds=30)
    with pendulum.test(freeze_datetime):
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 1
        freeze_datetime = freeze_datetime.add(seconds=30)
    with pendulum.test(freeze_datetime):
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 2
        expected_datetime = create_pendulum_time(year=2019, month=2, day=28, hour=0, minute=1)
        validate_tick(ticks[0], external_sensor, expected_datetime, TickStatus.SKIPPED)

def test_custom_interval_sensor_with_offset(monkeypatch, executor, instance, workspace_context, external_repo):
    if False:
        print('Hello World!')
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=28, tz='UTC'), 'US/Central')
    sleeps = []

    def fake_sleep(s):
        if False:
            return 10
        sleeps.append(s)
        pendulum.set_test_now(pendulum.now().add(seconds=s))
    monkeypatch.setattr(time, 'sleep', fake_sleep)
    shutdown_event = mock.MagicMock()
    shutdown_event.wait.side_effect = fake_sleep
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo.get_external_sensor('custom_interval_sensor')
        instance.add_instigator_state(InstigatorState(external_sensor.get_external_origin(), InstigatorType.SENSOR, InstigatorStatus.RUNNING))
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 1
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 1
        list(execute_sensor_iteration_loop(workspace_context, get_default_daemon_logger('dagster.daemon.SensorDaemon'), shutdown_event=shutdown_event, until=freeze_datetime.add(seconds=65).timestamp()))
        assert pendulum.now() == freeze_datetime.add(seconds=65)
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 2
        assert sum(sleeps) == 65

def test_sensor_start_stop(executor, instance, workspace_context, external_repo):
    if False:
        for i in range(10):
            print('nop')
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo.get_external_sensor('always_on_sensor')
        external_origin_id = external_sensor.get_external_origin_id()
        instance.start_sensor(external_sensor)
        assert instance.get_runs_count() == 0
        ticks = instance.get_ticks(external_origin_id, external_sensor.selector_id)
        assert len(ticks) == 0
        evaluate_sensors(workspace_context, executor)
        assert instance.get_runs_count() == 1
        run = instance.get_runs()[0]
        ticks = instance.get_ticks(external_origin_id, external_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], external_sensor, freeze_datetime, TickStatus.SUCCESS, [run.run_id])
        freeze_datetime = freeze_datetime.add(seconds=15)
    with pendulum.test(freeze_datetime):
        evaluate_sensors(workspace_context, executor)
        assert instance.get_runs_count() == 1
        ticks = instance.get_ticks(external_origin_id, external_sensor.selector_id)
        assert len(ticks) == 1
        instance.stop_sensor(external_origin_id, external_sensor.selector_id, external_sensor)
        instance.start_sensor(external_sensor)
        evaluate_sensors(workspace_context, executor)
        assert instance.get_runs_count() == 1
        ticks = instance.get_ticks(external_origin_id, external_sensor.selector_id)
        assert len(ticks) == 1
        freeze_datetime = freeze_datetime.add(seconds=16)
    with pendulum.test(freeze_datetime):
        evaluate_sensors(workspace_context, executor)
        assert instance.get_runs_count() == 2
        ticks = instance.get_ticks(external_origin_id, external_sensor.selector_id)
        assert len(ticks) == 2

def test_large_sensor(executor, instance, workspace_context, external_repo):
    if False:
        while True:
            i = 10
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo.get_external_sensor('large_sensor')
        instance.start_sensor(external_sensor)
        evaluate_sensors(workspace_context, executor, timeout=300)
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], external_sensor, freeze_datetime, TickStatus.SUCCESS)

def test_many_request_sensor(executor, submit_executor, instance, workspace_context, external_repo):
    if False:
        print('Hello World!')
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo.get_external_sensor('many_request_sensor')
        instance.start_sensor(external_sensor)
        evaluate_sensors(workspace_context, executor, submit_executor=submit_executor)
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], external_sensor, freeze_datetime, TickStatus.SUCCESS)

def test_cursor_sensor(executor, instance, workspace_context, external_repo):
    if False:
        print('Hello World!')
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        skip_sensor = external_repo.get_external_sensor('skip_cursor_sensor')
        run_sensor = external_repo.get_external_sensor('run_cursor_sensor')
        instance.start_sensor(skip_sensor)
        instance.start_sensor(run_sensor)
        evaluate_sensors(workspace_context, executor)
        skip_ticks = instance.get_ticks(skip_sensor.get_external_origin_id(), skip_sensor.selector_id)
        assert len(skip_ticks) == 1
        validate_tick(skip_ticks[0], skip_sensor, freeze_datetime, TickStatus.SKIPPED)
        assert skip_ticks[0].cursor == '1'
        run_ticks = instance.get_ticks(run_sensor.get_external_origin_id(), run_sensor.selector_id)
        assert len(run_ticks) == 1
        validate_tick(run_ticks[0], run_sensor, freeze_datetime, TickStatus.SUCCESS)
        assert run_ticks[0].cursor == '1'
    freeze_datetime = freeze_datetime.add(seconds=60)
    with pendulum.test(freeze_datetime):
        evaluate_sensors(workspace_context, executor)
        skip_ticks = instance.get_ticks(skip_sensor.get_external_origin_id(), skip_sensor.selector_id)
        assert len(skip_ticks) == 2
        validate_tick(skip_ticks[0], skip_sensor, freeze_datetime, TickStatus.SKIPPED)
        assert skip_ticks[0].cursor == '2'
        run_ticks = instance.get_ticks(run_sensor.get_external_origin_id(), run_sensor.selector_id)
        assert len(run_ticks) == 2
        validate_tick(run_ticks[0], run_sensor, freeze_datetime, TickStatus.SUCCESS)
        assert run_ticks[0].cursor == '2'

def test_run_request_asset_selection_sensor(executor, instance, workspace_context, external_repo):
    if False:
        return 10
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo.get_external_sensor('run_request_asset_selection_sensor')
        external_origin_id = external_sensor.get_external_origin_id()
        instance.start_sensor(external_sensor)
        assert instance.get_runs_count() == 0
        ticks = instance.get_ticks(external_origin_id, external_sensor.selector_id)
        assert len(ticks) == 0
        evaluate_sensors(workspace_context, executor)
        assert instance.get_runs_count() == 1
        run = instance.get_runs()[0]
        assert run.asset_selection == {AssetKey('a'), AssetKey('b')}
        ticks = instance.get_ticks(external_origin_id, external_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], external_sensor, freeze_datetime, TickStatus.SUCCESS, [run.run_id])
        planned_asset_keys = {record.event_log_entry.dagster_event.event_specific_data.asset_key for record in instance.get_event_records(EventRecordsFilter(DagsterEventType.ASSET_MATERIALIZATION_PLANNED))}
        assert planned_asset_keys == {AssetKey('a'), AssetKey('b')}

def test_run_request_stale_asset_selection_sensor_never_materialized(executor, instance, workspace_context, external_repo):
    if False:
        return 10
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo.get_external_sensor('run_request_stale_asset_sensor')
        instance.start_sensor(external_sensor)
        evaluate_sensors(workspace_context, executor)
        sensor_run = next((r for r in instance.get_runs() if r.job_name == 'abc'), None)
        assert sensor_run is not None
        assert sensor_run.asset_selection == {AssetKey('a'), AssetKey('b'), AssetKey('c')}

def test_run_request_stale_asset_selection_sensor_empty(executor, instance, workspace_context, external_repo):
    if False:
        return 10
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    materialize([a, b, c], instance=instance)
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo.get_external_sensor('run_request_stale_asset_sensor')
        instance.start_sensor(external_sensor)
        evaluate_sensors(workspace_context, executor)
        sensor_run = next((r for r in instance.get_runs() if r.job_name == 'abc'), None)
        assert sensor_run is None

def test_run_request_stale_asset_selection_sensor_subset(executor, instance, workspace_context, external_repo):
    if False:
        for i in range(10):
            print('nop')
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    materialize([a], instance=instance)
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo.get_external_sensor('run_request_stale_asset_sensor')
        instance.start_sensor(external_sensor)
        evaluate_sensors(workspace_context, executor)
        sensor_run = next((r for r in instance.get_runs() if r.job_name == 'abc'), None)
        assert sensor_run is not None
        assert sensor_run.asset_selection == {AssetKey('b'), AssetKey('c')}

def test_targets_asset_selection_sensor(executor, instance, workspace_context, external_repo):
    if False:
        while True:
            i = 10
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo.get_external_sensor('targets_asset_selection_sensor')
        external_origin_id = external_sensor.get_external_origin_id()
        instance.start_sensor(external_sensor)
        assert instance.get_runs_count() == 0
        ticks = instance.get_ticks(external_origin_id, external_sensor.selector_id)
        assert len(ticks) == 0
        evaluate_sensors(workspace_context, executor)
        assert instance.get_runs_count() == 2
        runs = instance.get_runs()
        assert len([run for run in runs if run.asset_selection == {AssetKey('asset_a'), AssetKey('asset_b')}]) == 1
        assert len([run for run in runs if run.asset_selection == {AssetKey('asset_b')}]) == 1
        ticks = instance.get_ticks(external_origin_id, external_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], external_sensor, freeze_datetime, TickStatus.SUCCESS, [run.run_id for run in runs])
        planned_asset_keys = [record.event_log_entry.dagster_event.event_specific_data.asset_key for record in instance.get_event_records(EventRecordsFilter(DagsterEventType.ASSET_MATERIALIZATION_PLANNED))]
        assert len(planned_asset_keys) == 3
        assert set(planned_asset_keys) == {AssetKey('asset_a'), AssetKey('asset_b')}

def test_partitioned_asset_selection_sensor(executor, instance, workspace_context, external_repo):
    if False:
        while True:
            i = 10
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo.get_external_sensor('partitioned_asset_selection_sensor')
        external_origin_id = external_sensor.get_external_origin_id()
        instance.start_sensor(external_sensor)
        assert instance.get_runs_count() == 0
        ticks = instance.get_ticks(external_origin_id, external_sensor.selector_id)
        assert len(ticks) == 0
        evaluate_sensors(workspace_context, executor)
        assert instance.get_runs_count() == 1
        run = instance.get_runs()[0]
        assert run.asset_selection == {AssetKey('hourly_asset_3')}
        assert run.tags['dagster/partition'] == '2022-08-01-00:00'
        ticks = instance.get_ticks(external_origin_id, external_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], external_sensor, freeze_datetime, TickStatus.SUCCESS, [run.run_id])
        planned_asset_keys = {record.event_log_entry.dagster_event.event_specific_data.asset_key for record in instance.get_event_records(EventRecordsFilter(DagsterEventType.ASSET_MATERIALIZATION_PLANNED))}
        assert planned_asset_keys == {AssetKey('hourly_asset_3')}

def test_asset_sensor(executor, instance, workspace_context, external_repo):
    if False:
        return 10
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        foo_sensor = external_repo.get_external_sensor('asset_foo_sensor')
        instance.start_sensor(foo_sensor)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(foo_sensor.get_external_origin_id(), foo_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], foo_sensor, freeze_datetime, TickStatus.SKIPPED)
        freeze_datetime = freeze_datetime.add(seconds=60)
    with pendulum.test(freeze_datetime):
        foo_job.execute_in_process(instance=instance)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(foo_sensor.get_external_origin_id(), foo_sensor.selector_id)
        assert len(ticks) == 2
        validate_tick(ticks[0], foo_sensor, freeze_datetime, TickStatus.SUCCESS)
        run = instance.get_runs()[0]
        assert run.run_config == {}
        assert run.tags
        assert run.tags.get('dagster/sensor_name') == 'asset_foo_sensor'

def test_asset_job_sensor(executor, instance, workspace_context, external_repo):
    if False:
        while True:
            i = 10
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        job_sensor = external_repo.get_external_sensor('asset_job_sensor')
        instance.start_sensor(job_sensor)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(job_sensor.get_external_origin_id(), job_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], job_sensor, freeze_datetime, TickStatus.SKIPPED)
        assert 'No new materialization events' in ticks[0].tick_data.skip_reason
        freeze_datetime = freeze_datetime.add(seconds=60)
    with pendulum.test(freeze_datetime):
        foo_job.execute_in_process(instance=instance)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(job_sensor.get_external_origin_id(), job_sensor.selector_id)
        assert len(ticks) == 2
        validate_tick(ticks[0], job_sensor, freeze_datetime, TickStatus.SUCCESS)
        run = instance.get_runs()[0]
        assert run.run_config == {}
        assert run.tags
        assert run.tags.get('dagster/sensor_name') == 'asset_job_sensor'

def test_asset_sensor_not_triggered_on_observation(executor, instance, workspace_context, external_repo):
    if False:
        for i in range(10):
            print('nop')
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        foo_sensor = external_repo.get_external_sensor('asset_foo_sensor')
        instance.start_sensor(foo_sensor)
        foo_observation_job.execute_in_process(instance=instance)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(foo_sensor.get_external_origin_id(), foo_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], foo_sensor, freeze_datetime, TickStatus.SKIPPED)
        freeze_datetime = freeze_datetime.add(seconds=60)
    with pendulum.test(freeze_datetime):
        foo_job.execute_in_process(instance=instance)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(foo_sensor.get_external_origin_id(), foo_sensor.selector_id)
        assert len(ticks) == 2
        validate_tick(ticks[0], foo_sensor, freeze_datetime, TickStatus.SUCCESS)
        run = instance.get_runs()[0]
        assert run.run_config == {}
        assert run.tags
        assert run.tags.get('dagster/sensor_name') == 'asset_foo_sensor'

def test_multi_asset_sensor(executor, instance, workspace_context, external_repo):
    if False:
        i = 10
        return i + 15
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        a_and_b_sensor = external_repo.get_external_sensor('asset_a_and_b_sensor')
        instance.start_sensor(a_and_b_sensor)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(a_and_b_sensor.get_external_origin_id(), a_and_b_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], a_and_b_sensor, freeze_datetime, TickStatus.SKIPPED)
        freeze_datetime = freeze_datetime.add(seconds=60)
    with pendulum.test(freeze_datetime):
        materialize([asset_a], instance=instance)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(a_and_b_sensor.get_external_origin_id(), a_and_b_sensor.selector_id)
        assert len(ticks) == 2
        validate_tick(ticks[0], a_and_b_sensor, freeze_datetime, TickStatus.SKIPPED)
        freeze_datetime = freeze_datetime.add(seconds=60)
    with pendulum.test(freeze_datetime):
        materialize([asset_b], instance=instance)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(a_and_b_sensor.get_external_origin_id(), a_and_b_sensor.selector_id)
        assert len(ticks) == 3
        validate_tick(ticks[0], a_and_b_sensor, freeze_datetime, TickStatus.SUCCESS)
        run = instance.get_runs()[0]
        assert run.run_config == {}
        assert run.tags
        assert run.tags.get('dagster/sensor_name') == 'asset_a_and_b_sensor'

def test_asset_selection_sensor(executor, instance, workspace_context, external_repo):
    if False:
        for i in range(10):
            print('nop')
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        asset_selection_sensor = external_repo.get_external_sensor('asset_selection_sensor')
        instance.start_sensor(asset_selection_sensor)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(asset_selection_sensor.get_external_origin_id(), asset_selection_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], asset_selection_sensor, freeze_datetime, TickStatus.SKIPPED)

def test_multi_asset_sensor_targets_asset_selection(executor, instance, workspace_context, external_repo):
    if False:
        return 10
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        multi_asset_sensor_targets_asset_selection = external_repo.get_external_sensor('multi_asset_sensor_targets_asset_selection')
        instance.start_sensor(multi_asset_sensor_targets_asset_selection)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(multi_asset_sensor_targets_asset_selection.get_external_origin_id(), multi_asset_sensor_targets_asset_selection.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], multi_asset_sensor_targets_asset_selection, freeze_datetime, TickStatus.SKIPPED)
        freeze_datetime = freeze_datetime.add(seconds=60)
    with pendulum.test(freeze_datetime):
        materialize([asset_a], instance=instance)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(multi_asset_sensor_targets_asset_selection.get_external_origin_id(), multi_asset_sensor_targets_asset_selection.selector_id)
        assert len(ticks) == 2
        validate_tick(ticks[0], multi_asset_sensor_targets_asset_selection, freeze_datetime, TickStatus.SKIPPED)
        freeze_datetime = freeze_datetime.add(seconds=60)
    with pendulum.test(freeze_datetime):
        materialize([asset_b], instance=instance)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(multi_asset_sensor_targets_asset_selection.get_external_origin_id(), multi_asset_sensor_targets_asset_selection.selector_id)
        assert len(ticks) == 3
        validate_tick(ticks[0], multi_asset_sensor_targets_asset_selection, freeze_datetime, TickStatus.SUCCESS)
        run = instance.get_runs()[0]
        assert run.run_config == {}
        assert run.tags
        assert run.tags.get('dagster/sensor_name') == 'multi_asset_sensor_targets_asset_selection'
        assert run.asset_selection == {AssetKey(['asset_c'])}

def test_multi_asset_sensor_w_many_events(executor, instance, workspace_context, external_repo):
    if False:
        print('Hello World!')
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        backlog_sensor = external_repo.get_external_sensor('backlog_sensor')
        instance.start_sensor(backlog_sensor)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(backlog_sensor.get_external_origin_id(), backlog_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], backlog_sensor, freeze_datetime, TickStatus.SKIPPED)
        freeze_datetime = freeze_datetime.add(seconds=60)
    with pendulum.test(freeze_datetime):
        materialize([asset_a], instance=instance)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(backlog_sensor.get_external_origin_id(), backlog_sensor.selector_id)
        assert len(ticks) == 2
        validate_tick(ticks[0], backlog_sensor, freeze_datetime, TickStatus.SKIPPED)
        freeze_datetime = freeze_datetime.add(seconds=60)
    with pendulum.test(freeze_datetime):
        materialize([asset_a], instance=instance)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(backlog_sensor.get_external_origin_id(), backlog_sensor.selector_id)
        assert len(ticks) == 3
        validate_tick(ticks[0], backlog_sensor, freeze_datetime, TickStatus.SUCCESS)
        run = instance.get_runs()[0]
        assert run.run_config == {}
        assert run.tags
        assert run.tags.get('dagster/sensor_name') == 'backlog_sensor'

def test_multi_asset_sensor_w_no_cursor_update(executor, instance, workspace_context, external_repo):
    if False:
        while True:
            i = 10
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        cursor_sensor = external_repo.get_external_sensor('doesnt_update_cursor_sensor')
        instance.start_sensor(cursor_sensor)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(cursor_sensor.get_external_origin_id(), cursor_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], cursor_sensor, freeze_datetime, TickStatus.SKIPPED)
        freeze_datetime = freeze_datetime.add(seconds=60)
    with pendulum.test(freeze_datetime):
        materialize([asset_a], instance=instance)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(cursor_sensor.get_external_origin_id(), cursor_sensor.selector_id)
        assert len(ticks) == 2
        validate_tick(ticks[0], cursor_sensor, freeze_datetime, TickStatus.FAILURE)

def test_multi_asset_sensor_hourly_to_weekly(executor, instance, workspace_context, external_repo):
    if False:
        i = 10
        return i + 15
    freeze_datetime = to_timezone(create_pendulum_time(year=2022, month=8, day=2, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        materialize([hourly_asset], instance=instance, partition_key='2022-08-01-00:00')
        cursor_sensor = external_repo.get_external_sensor('multi_asset_sensor_hourly_to_weekly')
        instance.start_sensor(cursor_sensor)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(cursor_sensor.get_external_origin_id(), cursor_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], cursor_sensor, freeze_datetime, TickStatus.SUCCESS)
        run = instance.get_runs()[0]
        assert run.run_config == {}
        assert run.tags
        assert run.tags.get('dagster/sensor_name') == 'multi_asset_sensor_hourly_to_weekly'
        assert run.tags.get('dagster/partition') == '2022-07-31'

def test_multi_asset_sensor_hourly_to_hourly(executor, instance, workspace_context, external_repo):
    if False:
        return 10
    freeze_datetime = to_timezone(create_pendulum_time(year=2022, month=8, day=3, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        materialize([hourly_asset], instance=instance, partition_key='2022-08-02-00:00')
        cursor_sensor = external_repo.get_external_sensor('multi_asset_sensor_hourly_to_hourly')
        instance.start_sensor(cursor_sensor)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(cursor_sensor.get_external_origin_id(), cursor_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], cursor_sensor, freeze_datetime, TickStatus.SUCCESS)
        run = instance.get_runs()[0]
        assert run.run_config == {}
        assert run.tags
        assert run.tags.get('dagster/sensor_name') == 'multi_asset_sensor_hourly_to_hourly'
        assert run.tags.get('dagster/partition') == '2022-08-02-00:00'
        freeze_datetime = freeze_datetime.add(seconds=30)
    with pendulum.test(freeze_datetime):
        cursor_sensor = external_repo.get_external_sensor('multi_asset_sensor_hourly_to_hourly')
        instance.start_sensor(cursor_sensor)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(cursor_sensor.get_external_origin_id(), cursor_sensor.selector_id)
        assert len(ticks) == 2
        validate_tick(ticks[0], cursor_sensor, freeze_datetime, TickStatus.SKIPPED)

def test_sensor_result_multi_asset_sensor(executor, instance, workspace_context, external_repo):
    if False:
        while True:
            i = 10
    freeze_datetime = to_timezone(create_pendulum_time(year=2022, month=8, day=3, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        cursor_sensor = external_repo.get_external_sensor('sensor_result_multi_asset_sensor')
        instance.start_sensor(cursor_sensor)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(cursor_sensor.get_external_origin_id(), cursor_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], cursor_sensor, freeze_datetime, TickStatus.SUCCESS)

def test_cursor_update_sensor_result_multi_asset_sensor(executor, instance, workspace_context, external_repo):
    if False:
        i = 10
        return i + 15
    freeze_datetime = to_timezone(create_pendulum_time(year=2022, month=8, day=3, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        cursor_sensor = external_repo.get_external_sensor('cursor_sensor_result_multi_asset_sensor')
        instance.start_sensor(cursor_sensor)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(cursor_sensor.get_external_origin_id(), cursor_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], cursor_sensor, freeze_datetime, TickStatus.FAILURE)
        assert 'Cannot set cursor in a multi_asset_sensor' in ticks[0].error.message

def test_multi_job_sensor(executor, instance, workspace_context, external_repo):
    if False:
        return 10
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        job_sensor = external_repo.get_external_sensor('two_job_sensor')
        instance.start_sensor(job_sensor)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(job_sensor.get_external_origin_id(), job_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], job_sensor, freeze_datetime, TickStatus.SUCCESS)
        run = instance.get_runs()[0]
        assert run.run_config == {}
        assert run.tags.get('dagster/sensor_name') == 'two_job_sensor'
        assert run.job_name == 'the_job'
        freeze_datetime = freeze_datetime.add(seconds=60)
    with pendulum.test(freeze_datetime):
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(job_sensor.get_external_origin_id(), job_sensor.selector_id)
        assert len(ticks) == 2
        validate_tick(ticks[0], job_sensor, freeze_datetime, TickStatus.SUCCESS)
        run = instance.get_runs()[0]
        assert run.run_config == {'ops': {'config_op': {'config': {'foo': 'blah'}}}}
        assert run.tags
        assert run.tags.get('dagster/sensor_name') == 'two_job_sensor'
        assert run.job_name == 'config_job'

def test_bad_run_request_untargeted(executor, instance, workspace_context, external_repo):
    if False:
        i = 10
        return i + 15
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        job_sensor = external_repo.get_external_sensor('bad_request_untargeted')
        instance.start_sensor(job_sensor)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(job_sensor.get_external_origin_id(), job_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], job_sensor, freeze_datetime, TickStatus.FAILURE, None, 'Error in sensor bad_request_untargeted: Sensor evaluation function returned a RunRequest for a sensor lacking a specified target (job_name, job, or jobs).')

def test_bad_run_request_mismatch(executor, instance, workspace_context, external_repo):
    if False:
        return 10
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        job_sensor = external_repo.get_external_sensor('bad_request_mismatch')
        instance.start_sensor(job_sensor)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(job_sensor.get_external_origin_id(), job_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], job_sensor, freeze_datetime, TickStatus.FAILURE, None, "Error in sensor bad_request_mismatch: Sensor returned a RunRequest with job_name config_job. Expected one of: ['the_job']")

def test_bad_run_request_unspecified(executor, instance, workspace_context, external_repo):
    if False:
        for i in range(10):
            print('nop')
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        job_sensor = external_repo.get_external_sensor('bad_request_unspecified')
        instance.start_sensor(job_sensor)
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(job_sensor.get_external_origin_id(), job_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], job_sensor, freeze_datetime, TickStatus.FAILURE, None, "Error in sensor bad_request_unspecified: Sensor returned a RunRequest that did not specify job_name for the requested run. Expected one of: ['the_job', 'config_job']")

def test_status_in_code_sensor(executor, instance):
    if False:
        print('Hello World!')
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, hour=23, minute=59, second=59, tz='UTC'), 'US/Central')
    with create_test_daemon_workspace_context(create_workspace_load_target(attribute='the_status_in_code_repo'), instance=instance) as workspace_context:
        external_repo = next(iter(workspace_context.create_request_context().get_workspace_snapshot().values())).code_location.get_repository('the_status_in_code_repo')
        with pendulum.test(freeze_datetime):
            running_sensor = external_repo.get_external_sensor('always_running_sensor')
            not_running_sensor = external_repo.get_external_sensor('never_running_sensor')
            always_running_origin = running_sensor.get_external_origin()
            never_running_origin = not_running_sensor.get_external_origin()
            assert instance.get_runs_count() == 0
            assert len(instance.get_ticks(always_running_origin.get_id(), running_sensor.selector_id)) == 0
            assert len(instance.get_ticks(never_running_origin.get_id(), not_running_sensor.selector_id)) == 0
            assert len(instance.all_instigator_state()) == 0
            evaluate_sensors(workspace_context, executor)
            assert instance.get_runs_count() == 0
            assert len(instance.all_instigator_state()) == 1
            instigator_state = instance.get_instigator_state(always_running_origin.get_id(), running_sensor.selector_id)
            assert instigator_state.status == InstigatorStatus.AUTOMATICALLY_RUNNING
            ticks = instance.get_ticks(running_sensor.get_external_origin_id(), running_sensor.selector_id)
            assert len(ticks) == 1
            validate_tick(ticks[0], running_sensor, freeze_datetime, TickStatus.SKIPPED)
            assert len(instance.get_ticks(never_running_origin.get_id(), not_running_sensor.selector_id)) == 0
        freeze_datetime = freeze_datetime.add(seconds=30)
        with pendulum.test(freeze_datetime):
            evaluate_sensors(workspace_context, executor)
            wait_for_all_runs_to_start(instance)
            assert instance.get_runs_count() == 1
            run = instance.get_runs()[0]
            validate_run_started(run)
            ticks = instance.get_ticks(running_sensor.get_external_origin_id(), running_sensor.selector_id)
            assert len(ticks) == 2
            expected_datetime = create_pendulum_time(year=2019, month=2, day=28, hour=0, minute=0, second=29)
            validate_tick(ticks[0], running_sensor, expected_datetime, TickStatus.SUCCESS, [run.run_id])
            assert len(instance.get_ticks(never_running_origin.get_id(), not_running_sensor.selector_id)) == 0

def test_run_request_list_sensor(executor, instance, workspace_context, external_repo):
    if False:
        print('Hello World!')
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, hour=23, minute=59, second=59, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo.get_external_sensor('request_list_sensor')
        instance.add_instigator_state(InstigatorState(external_sensor.get_external_origin(), InstigatorType.SENSOR, InstigatorStatus.RUNNING))
        assert instance.get_runs_count() == 0
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 0
        evaluate_sensors(workspace_context, executor)
        assert instance.get_runs_count() == 2
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 1

def test_sensor_purge(executor, instance, workspace_context, external_repo):
    if False:
        i = 10
        return i + 15
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, hour=23, minute=59, second=59, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo.get_external_sensor('simple_sensor')
        instance.add_instigator_state(InstigatorState(external_sensor.get_external_origin(), InstigatorType.SENSOR, InstigatorStatus.RUNNING))
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 0
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 1
        freeze_datetime = freeze_datetime.add(days=6)
    with pendulum.test(freeze_datetime):
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 2
        freeze_datetime = freeze_datetime.add(days=2)
    with pendulum.test(freeze_datetime):
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 2

def test_sensor_custom_purge(executor, workspace_context, external_repo):
    if False:
        print('Hello World!')
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, hour=23, minute=59, second=59, tz='UTC'), 'US/Central')
    with instance_for_test(overrides={'retention': {'sensor': {'purge_after_days': {'skipped': 14}}}, 'run_launcher': {'module': 'dagster._core.test_utils', 'class': 'MockedRunLauncher'}}) as instance:
        purge_ws_ctx = workspace_context.copy_for_test_instance(instance)
        with pendulum.test(freeze_datetime):
            external_sensor = external_repo.get_external_sensor('simple_sensor')
            instance.add_instigator_state(InstigatorState(external_sensor.get_external_origin(), InstigatorType.SENSOR, InstigatorStatus.RUNNING))
            ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
            assert len(ticks) == 0
            evaluate_sensors(purge_ws_ctx, executor)
            ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
            assert len(ticks) == 1
            freeze_datetime = freeze_datetime.add(days=8)
        with pendulum.test(freeze_datetime):
            evaluate_sensors(purge_ws_ctx, executor)
            ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
            assert len(ticks) == 2
            freeze_datetime = freeze_datetime.add(days=7)
        with pendulum.test(freeze_datetime):
            evaluate_sensors(purge_ws_ctx, executor)
            ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
            assert len(ticks) == 2

def test_repository_namespacing(executor):
    if False:
        i = 10
        return i + 15
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, hour=23, minute=59, second=59, tz='UTC'), 'US/Central')
    with ExitStack() as exit_stack:
        instance = exit_stack.enter_context(instance_for_test())
        full_workspace_context = exit_stack.enter_context(create_test_daemon_workspace_context(create_workspace_load_target(attribute=None), instance=instance))
        full_location = next(iter(full_workspace_context.create_request_context().get_workspace_snapshot().values())).code_location
        external_repo = full_location.get_repository('the_repo')
        other_repo = full_location.get_repository('the_other_repo')
        status_in_code_repo = full_location.get_repository('the_status_in_code_repo')
        running_sensor = status_in_code_repo.get_external_sensor('always_running_sensor')
        instance.stop_sensor(running_sensor.get_external_origin_id(), running_sensor.selector_id, running_sensor)
        external_sensor = external_repo.get_external_sensor('run_key_sensor')
        other_sensor = other_repo.get_external_sensor('run_key_sensor')
        with pendulum.test(freeze_datetime):
            instance.start_sensor(external_sensor)
            assert instance.get_runs_count() == 0
            ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
            assert len(ticks) == 0
            instance.start_sensor(other_sensor)
            assert instance.get_runs_count() == 0
            ticks = instance.get_ticks(other_sensor.get_external_origin_id(), other_sensor.selector_id)
            assert len(ticks) == 0
            evaluate_sensors(full_workspace_context, executor)
            wait_for_all_runs_to_start(instance)
            assert instance.get_runs_count() == 2
            ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
            assert len(ticks) == 1
            assert ticks[0].status == TickStatus.SUCCESS
            ticks = instance.get_ticks(other_sensor.get_external_origin_id(), other_sensor.selector_id)
            assert len(ticks) == 1
        freeze_datetime = freeze_datetime.add(seconds=30)
        with pendulum.test(freeze_datetime):
            evaluate_sensors(full_workspace_context, executor)
            assert instance.get_runs_count() == 2
            ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
            assert len(ticks) == 2

def test_settings():
    if False:
        for i in range(10):
            print('nop')
    settings = {'use_threads': True, 'num_workers': 4}
    with instance_for_test(overrides={'sensors': settings}) as thread_inst:
        assert thread_inst.get_settings('sensors') == settings

def test_sensor_logging(executor, instance, workspace_context, external_repo):
    if False:
        while True:
            i = 10
    external_sensor = external_repo.get_external_sensor('logging_sensor')
    instance.add_instigator_state(InstigatorState(external_sensor.get_external_origin(), InstigatorType.SENSOR, InstigatorStatus.RUNNING))
    assert instance.get_runs_count() == 0
    ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
    assert len(ticks) == 0
    evaluate_sensors(workspace_context, executor)
    ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
    assert len(ticks) == 1
    tick = ticks[0]
    assert tick.log_key
    records = get_instigation_log_records(instance, tick.log_key)
    assert len(records) == 2
    assert records[0][DAGSTER_META_KEY]['orig_message'] == 'hello hello'
    assert records[1][DAGSTER_META_KEY]['orig_message'].endswith('hello hello')
    instance.compute_log_manager.delete_logs(log_key=tick.log_key)

def test_add_dynamic_partitions_sensor(caplog, executor, instance, workspace_context, external_repo):
    if False:
        return 10
    foo_job.execute_in_process(instance=instance)
    instance.add_dynamic_partitions('quux', ['foo'])
    assert set(instance.get_dynamic_partitions('quux')) == set(['foo'])
    external_sensor = external_repo.get_external_sensor('add_dynamic_partitions_sensor')
    instance.add_instigator_state(InstigatorState(external_sensor.get_external_origin(), InstigatorType.SENSOR, InstigatorStatus.RUNNING))
    ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
    assert len(ticks) == 0
    evaluate_sensors(workspace_context, executor)
    assert set(instance.get_dynamic_partitions('quux')) == set(['baz', 'foo'])
    ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
    assert "Added partition keys to dynamic partitions definition 'quux': ['baz']" in caplog.text
    assert "Skipping addition of partition keys for dynamic partitions definition 'quux' that already exist: ['foo']" in caplog.text
    assert ticks[0].tick_data.dynamic_partitions_request_results == [DynamicPartitionsRequestResult('quux', added_partitions=['baz'], deleted_partitions=None, skipped_partitions=['foo'])]

def test_add_delete_skip_dynamic_partitions(caplog, executor, instance, workspace_context, external_repo):
    if False:
        return 10
    foo_job.execute_in_process(instance=instance)
    instance.add_dynamic_partitions('quux', ['2'])
    assert set(instance.get_dynamic_partitions('quux')) == set(['2'])
    external_sensor = external_repo.get_external_sensor('add_delete_dynamic_partitions_and_yield_run_requests_sensor')
    instance.add_instigator_state(InstigatorState(external_sensor.get_external_origin(), InstigatorType.SENSOR, InstigatorStatus.RUNNING))
    ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
    assert len(ticks) == 0
    freeze_datetime = to_timezone(create_pendulum_time(year=2023, month=2, day=27, hour=23, minute=59, second=59, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 1
        assert set(instance.get_dynamic_partitions('quux')) == set(['1'])
        assert instance.get_runs_count() == 2
        assert "Added partition keys to dynamic partitions definition 'quux': ['1']" in caplog.text
        assert "Deleted partition keys from dynamic partitions definition 'quux': ['2']" in caplog.text
        assert "Skipping deletion of partition keys for dynamic partitions definition 'quux' that do not exist: ['3']" in caplog.text
        assert ticks[0].tick_data.dynamic_partitions_request_results == [DynamicPartitionsRequestResult('quux', added_partitions=['1'], deleted_partitions=None, skipped_partitions=[]), DynamicPartitionsRequestResult('quux', added_partitions=None, deleted_partitions=['2'], skipped_partitions=['3'])]
        run = instance.get_runs()[0]
        assert run.run_config == {}
        assert run.tags
        assert run.tags.get('dagster/partition') == '1'
    freeze_datetime = freeze_datetime.add(seconds=60)
    with pendulum.test(freeze_datetime):
        evaluate_sensors(workspace_context, executor)
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 2
        assert ticks[0].tick_data.dynamic_partitions_request_results == [DynamicPartitionsRequestResult('quux', added_partitions=[], deleted_partitions=None, skipped_partitions=['1']), DynamicPartitionsRequestResult('quux', added_partitions=None, deleted_partitions=[], skipped_partitions=['2', '3'])]
        assert "Skipping addition of partition keys for dynamic partitions definition 'quux' that already exist: ['1']" in caplog.text
        assert "Skipping deletion of partition keys for dynamic partitions definition 'quux' that do not exist: ['2', '3']" in caplog.text

def test_error_on_deleted_dynamic_partitions_run_request(executor, instance, workspace_context, external_repo):
    if False:
        print('Hello World!')
    foo_job.execute_in_process(instance=instance)
    instance.add_dynamic_partitions('quux', ['2'])
    assert set(instance.get_dynamic_partitions('quux')) == set(['2'])
    external_sensor = external_repo.get_external_sensor('error_on_deleted_dynamic_partitions_run_requests_sensor')
    instance.add_instigator_state(InstigatorState(external_sensor.get_external_origin(), InstigatorType.SENSOR, InstigatorStatus.RUNNING))
    ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
    assert len(ticks) == 0
    evaluate_sensors(workspace_context, executor)
    ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
    assert len(ticks) == 1
    validate_tick(ticks[0], external_sensor, expected_datetime=None, expected_status=TickStatus.FAILURE, expected_run_ids=None, expected_error="Dynamic partition key 2 for partitions def 'quux' is invalid")
    assert set(instance.get_dynamic_partitions('quux')) == set(['2'])

@pytest.mark.parametrize('sensor_name, is_expected_success', [('success_on_multipartition_run_request_with_two_dynamic_dimensions_sensor', True), ('error_on_multipartition_run_request_with_two_dynamic_dimensions_sensor', False)])
def test_multipartitions_with_dynamic_dims_run_request_sensor(sensor_name, is_expected_success, executor, instance, workspace_context, external_repo):
    if False:
        i = 10
        return i + 15
    external_sensor = external_repo.get_external_sensor(sensor_name)
    instance.add_instigator_state(InstigatorState(external_sensor.get_external_origin(), InstigatorType.SENSOR, InstigatorStatus.RUNNING))
    ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
    assert len(ticks) == 0
    evaluate_sensors(workspace_context, executor)
    ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
    assert len(ticks) == 1
    if is_expected_success:
        validate_tick(ticks[0], external_sensor, expected_datetime=None, expected_status=TickStatus.SUCCESS, expected_run_ids=None)
    else:
        validate_tick(ticks[0], external_sensor, expected_datetime=None, expected_status=TickStatus.FAILURE, expected_run_ids=None, expected_error='does not exist in the set of valid partition keys')

def test_multipartition_asset_with_static_time_dimensions_run_requests_sensor(executor, instance, workspace_context, external_repo):
    if False:
        return 10
    external_sensor = external_repo.get_external_sensor('multipartitions_with_static_time_dimensions_run_requests_sensor')
    instance.add_instigator_state(InstigatorState(external_sensor.get_external_origin(), InstigatorType.SENSOR, InstigatorStatus.RUNNING))
    ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
    assert len(ticks) == 0
    evaluate_sensors(workspace_context, executor)
    ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
    assert len(ticks) == 1
    validate_tick(ticks[0], external_sensor, expected_datetime=None, expected_status=TickStatus.SUCCESS, expected_run_ids=None)

def test_code_location_construction():
    if False:
        for i in range(10):
            print('nop')

    @run_status_sensor(monitored_jobs=[CodeLocationSelector(location_name='test_location')], run_status=DagsterRunStatus.SUCCESS)
    def cross_code_location_sensor(context):
        if False:
            i = 10
            return i + 15
        raise Exception('never executed')
    assert cross_code_location_sensor

def test_stale_request_context(instance, workspace_context, external_repo):
    if False:
        i = 10
        return i + 15
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, hour=23, minute=59, second=59, tz='UTC'), 'US/Central')
    executor = ThreadPoolExecutor()
    blocking_executor = BlockingThreadPoolExecutor()
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo.get_external_sensor('simple_sensor')
        instance.add_instigator_state(InstigatorState(external_sensor.get_external_origin(), InstigatorType.SENSOR, InstigatorStatus.RUNNING))
        assert instance.get_runs_count() == 0
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 0
        futures = {}
        list(execute_sensor_iteration(workspace_context, get_default_daemon_logger('SensorDaemon'), threadpool_executor=executor, sensor_tick_futures=futures))
        blocking_executor.allow()
        wait_for_futures(futures, timeout=FUTURES_TIMEOUT)
        assert instance.get_runs_count() == 0
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 1
        validate_tick(ticks[0], external_sensor, freeze_datetime, TickStatus.SKIPPED)
    blocking_executor.block()
    freeze_datetime = freeze_datetime.add(seconds=30)
    with pendulum.test(freeze_datetime):
        futures = {}
        list(execute_sensor_iteration(workspace_context, get_default_daemon_logger('SensorDaemon'), threadpool_executor=executor, sensor_tick_futures=futures, submit_threadpool_executor=blocking_executor))
        p = workspace_context._grpc_server_registry._all_processes[0]
        workspace_context.reload_workspace()
        p.server_process.kill()
        p.wait()
        blocking_executor.allow()
        wait_for_futures(futures, timeout=FUTURES_TIMEOUT)
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 2
        wait_for_all_runs_to_start(instance)
        assert instance.get_runs_count() == 1, ticks[0].error
        run = instance.get_runs()[0]
        validate_run_started(run)
        expected_datetime = create_pendulum_time(year=2019, month=2, day=28, hour=0, minute=0, second=29)
        validate_tick(ticks[0], external_sensor, expected_datetime, TickStatus.SUCCESS, [run.run_id])