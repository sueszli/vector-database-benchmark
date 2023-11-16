from contextlib import contextmanager
from typing import Iterator, List, cast
from unittest import mock
import pytest
from dagster import AssetKey, AssetOut, AssetSelection, Config, DagsterEventType, DagsterInstance, DagsterInvariantViolationError, DagsterRunStatus, DagsterUnknownPartitionError, DailyPartitionsDefinition, Definitions, EventRecordsFilter, FreshnessPolicy, Output, RunConfig, RunRequest, SkipReason, SourceAsset, StaticPartitionsDefinition, asset, asset_sensor, build_freshness_policy_sensor_context, build_multi_asset_sensor_context, build_run_status_sensor_context, build_sensor_context, define_asset_job, freshness_policy_sensor, job, materialize, multi_asset, multi_asset_sensor, op, repository, resource, run_failure_sensor, run_status_sensor, sensor, static_partitioned_config
from dagster._config.pythonic_config import ConfigurableResource
from dagster._core.definitions.metadata import MetadataValue
from dagster._core.definitions.partition import DynamicPartitionsDefinition
from dagster._core.definitions.resource_annotation import ResourceParam
from dagster._core.errors import DagsterInvalidDefinitionError, DagsterInvalidInvocationError
from dagster._core.execution.build_resources import build_resources
from dagster._core.storage.tags import PARTITION_NAME_TAG
from dagster._core.test_utils import instance_for_test

def test_sensor_invocation_args():
    if False:
        while True:
            i = 10

    @sensor(job_name='foo_job')
    def basic_sensor_no_arg():
        if False:
            while True:
                i = 10
        return RunRequest(run_key=None, run_config={}, tags={})
    assert basic_sensor_no_arg().run_config == {}

    @sensor(job_name='foo_job')
    def basic_sensor(_):
        if False:
            while True:
                i = 10
        return RunRequest(run_key=None, run_config={}, tags={})
    assert basic_sensor(build_sensor_context()).run_config == {}
    assert basic_sensor(None).run_config == {}

    @sensor(job_name='foo_job')
    def basic_sensor_with_context(_arbitrary_context):
        if False:
            for i in range(10):
                print('nop')
        return RunRequest(run_key=None, run_config={}, tags={})
    context = build_sensor_context()
    assert basic_sensor_with_context(context).run_config == {}
    assert basic_sensor_with_context(_arbitrary_context=context).run_config == {}
    with pytest.raises(DagsterInvalidInvocationError, match="Sensor invocation expected argument '_arbitrary_context'."):
        basic_sensor_with_context(bad_context=context)
    with pytest.raises(DagsterInvalidInvocationError, match='Sensor evaluation function expected context argument, but no context argument was provided when invoking.'):
        basic_sensor_with_context()
    with pytest.raises(DagsterInvalidInvocationError, match='Sensor invocation received multiple non-resource arguments. Only a first positional context parameter should be provided when invoking.'):
        basic_sensor_with_context(context, _arbitrary_context=None)

def test_sensor_invocation_resources() -> None:
    if False:
        print('Hello World!')

    class MyResource(ConfigurableResource):
        a_str: str

    @sensor(job_name='foo_job')
    def basic_sensor_resource_req(my_resource: MyResource):
        if False:
            for i in range(10):
                print('nop')
        return RunRequest(run_key=None, run_config={'foo': my_resource.a_str}, tags={})
    with pytest.raises(DagsterInvalidDefinitionError, match="Resource with key 'my_resource' required by sensor 'basic_sensor_resource_req' was not provided."):
        basic_sensor_resource_req()
    assert cast(RunRequest, basic_sensor_resource_req(build_sensor_context(resources={'my_resource': MyResource(a_str='foo')}))).run_config == {'foo': 'foo'}

def test_sensor_invocation_resources_direct() -> None:
    if False:
        print('Hello World!')

    class MyResource(ConfigurableResource):
        a_str: str

    @sensor(job_name='foo_job')
    def basic_sensor_resource_req(my_resource: MyResource):
        if False:
            i = 10
            return i + 15
        return RunRequest(run_key=None, run_config={'foo': my_resource.a_str}, tags={})
    with pytest.raises(DagsterInvalidDefinitionError, match="Resource with key 'my_resource' required by sensor 'basic_sensor_resource_req' was not provided."):
        basic_sensor_resource_req()
    assert cast(RunRequest, basic_sensor_resource_req(context=build_sensor_context(resources={'my_resource': MyResource(a_str='foo')}))).run_config == {'foo': 'foo'}
    assert cast(RunRequest, basic_sensor_resource_req(my_resource=MyResource(a_str='foo'))).run_config == {'foo': 'foo'}
    with pytest.raises(DagsterInvalidInvocationError, match='If directly invoking a sensor, you may not provide resources as positional arguments, only as keyword arguments.'):
        assert cast(RunRequest, basic_sensor_resource_req(MyResource(a_str='foo'))).run_config == {'foo': 'foo'}
    assert cast(RunRequest, basic_sensor_resource_req(build_sensor_context(), my_resource=MyResource(a_str='foo'))).run_config == {'foo': 'foo'}

    @sensor(job_name='foo_job')
    def basic_sensor_with_context_resource_req(my_resource: MyResource, context):
        if False:
            return 10
        return RunRequest(run_key=None, run_config={'foo': my_resource.a_str}, tags={})
    assert cast(RunRequest, basic_sensor_with_context_resource_req(build_sensor_context(), my_resource=MyResource(a_str='foo'))).run_config == {'foo': 'foo'}

def test_recreating_sensor_with_resource_arg() -> None:
    if False:
        while True:
            i = 10

    class MyResource(ConfigurableResource):
        a_str: str

    @sensor(job_name='foo_job')
    def basic_sensor_with_context_resource_req(my_resource: MyResource, context):
        if False:
            while True:
                i = 10
        return RunRequest(run_key=None, run_config={'foo': my_resource.a_str}, tags={})

    @job
    def junk_job():
        if False:
            return 10
        pass
    updated_sensor = basic_sensor_with_context_resource_req.with_updated_job(junk_job)
    assert cast(RunRequest, updated_sensor(build_sensor_context(), my_resource=MyResource(a_str='foo'))).run_config == {'foo': 'foo'}

def test_sensor_invocation_resources_direct_many() -> None:
    if False:
        print('Hello World!')

    class MyResource(ConfigurableResource):
        a_str: str

    @sensor(job_name='foo_job')
    def basic_sensor_resource_req(my_resource: MyResource, my_other_resource: MyResource):
        if False:
            for i in range(10):
                print('nop')
        return RunRequest(run_key=None, run_config={'foo': my_resource.a_str, 'bar': my_other_resource.a_str}, tags={})
    assert cast(RunRequest, basic_sensor_resource_req(my_other_resource=MyResource(a_str='bar'), my_resource=MyResource(a_str='foo'))).run_config == {'foo': 'foo', 'bar': 'bar'}
    assert cast(RunRequest, basic_sensor_resource_req(context=build_sensor_context(resources={'my_other_resource': MyResource(a_str='bar')}), my_resource=MyResource(a_str='foo'))).run_config == {'foo': 'foo', 'bar': 'bar'}

def test_sensor_invocation_resources_context_manager() -> None:
    if False:
        i = 10
        return i + 15

    @sensor(job_name='foo_job')
    def basic_sensor_str_resource_req(my_resource: ResourceParam[str]):
        if False:
            for i in range(10):
                print('nop')
        return RunRequest(run_key=None, run_config={'foo': my_resource}, tags={})

    @resource
    @contextmanager
    def my_cm_resource(_) -> Iterator[str]:
        if False:
            i = 10
            return i + 15
        yield 'foo'
    with pytest.raises(DagsterInvariantViolationError, match='At least one provided resource is a generator'):
        basic_sensor_str_resource_req(build_sensor_context(resources={'my_resource': my_cm_resource}))
    with build_sensor_context(resources={'my_resource': my_cm_resource}) as context:
        assert cast(RunRequest, basic_sensor_str_resource_req(context)).run_config == {'foo': 'foo'}

def test_sensor_invocation_resources_deferred() -> None:
    if False:
        i = 10
        return i + 15

    class MyResource(ConfigurableResource):

        def create_resource(self, context) -> None:
            if False:
                for i in range(10):
                    print('nop')
            raise Exception()

    @sensor(job_name='foo_job', required_resource_keys={'my_resource'})
    def basic_sensor_resource_req() -> RunRequest:
        if False:
            while True:
                i = 10
        return RunRequest(run_key=None, run_config={}, tags={})
    context = build_sensor_context(resources={'my_resource': MyResource()})
    with pytest.raises(Exception):
        basic_sensor_resource_req(context)
    with context as open_context:
        with pytest.raises(Exception):
            basic_sensor_resource_req(open_context)

def test_multi_asset_sensor_invocation_resources() -> None:
    if False:
        while True:
            i = 10

    class MyResource(ConfigurableResource):
        a_str: str

    @op
    def an_op():
        if False:
            return 10
        return 1

    @job
    def the_job():
        if False:
            for i in range(10):
                print('nop')
        an_op()

    @asset
    def asset_a():
        if False:
            for i in range(10):
                print('nop')
        return 1

    @asset
    def asset_b():
        if False:
            for i in range(10):
                print('nop')
        return 1

    @multi_asset_sensor(monitored_assets=[AssetKey('asset_a'), AssetKey('asset_b')], job=the_job)
    def a_and_b_sensor(context, my_resource: MyResource):
        if False:
            i = 10
            return i + 15
        asset_events = context.latest_materialization_records_by_key()
        if all(asset_events.values()):
            context.advance_all_cursors()
            return RunRequest(run_key=context.cursor, run_config={'foo': my_resource.a_str}, tags={})

    @repository
    def my_repo():
        if False:
            for i in range(10):
                print('nop')
        return [asset_a, asset_b, a_and_b_sensor]
    with instance_for_test() as instance:
        materialize([asset_a, asset_b], instance=instance)
        ctx = build_multi_asset_sensor_context(monitored_assets=[AssetKey('asset_a'), AssetKey('asset_b')], instance=instance, repository_def=my_repo, resources={'my_resource': MyResource(a_str='bar')})
        assert cast(RunRequest, a_and_b_sensor(ctx)).run_config == {'foo': 'bar'}

def test_multi_asset_sensor_with_source_assets() -> None:
    if False:
        while True:
            i = 10

    @asset(partitions_def=DailyPartitionsDefinition(start_date='2023-03-01'))
    def upstream_asset1():
        if False:
            return 10
        ...
    upstream_asset1_source = SourceAsset(key=upstream_asset1.key, partitions_def=DailyPartitionsDefinition(start_date='2023-03-01'))

    @asset()
    def downstream_asset(upstream_asset1):
        if False:
            for i in range(10):
                print('nop')
        ...

    @multi_asset_sensor(monitored_assets=[upstream_asset1.key], job=define_asset_job('foo', selection=[downstream_asset]))
    def my_sensor(context):
        if False:
            while True:
                i = 10
        run_requests = []
        for (partition, record) in context.latest_materialization_records_by_partition(AssetKey('upstream_asset1')).items():
            context.advance_cursor({upstream_asset1.key: record})
            run_requests.append(RunRequest(partition_key=partition))
        return run_requests

    @repository
    def my_repo():
        if False:
            while True:
                i = 10
        return [upstream_asset1_source, downstream_asset, my_sensor]
    with instance_for_test() as instance:
        materialize([upstream_asset1], instance=instance, partition_key='2023-03-01')
        ctx = build_multi_asset_sensor_context(monitored_assets=[AssetKey('upstream_asset1')], instance=instance, repository_def=my_repo)
        run_requests = cast(List[RunRequest], my_sensor(ctx))
        assert len(run_requests) == 1
        assert run_requests[0].partition_key == '2023-03-01'

def test_freshness_policy_sensor_invocation_resources() -> None:
    if False:
        i = 10
        return i + 15

    class MyResource(ConfigurableResource):
        a_str: str

    @freshness_policy_sensor(asset_selection=AssetSelection.all())
    def freshness_sensor(context, my_resource: MyResource) -> None:
        if False:
            while True:
                i = 10
        assert context.minutes_overdue == 10
        assert context.previous_minutes_overdue is None
        assert my_resource.a_str == 'bar'
    with build_resources({'my_resource': MyResource(a_str='bar')}) as resources:
        context = build_freshness_policy_sensor_context(sensor_name='status_sensor', asset_key=AssetKey('a'), freshness_policy=FreshnessPolicy(maximum_lag_minutes=30), minutes_overdue=10, resources=resources)
        freshness_sensor(context)

def test_run_status_sensor_invocation_resources() -> None:
    if False:
        while True:
            i = 10

    class MyResource(ConfigurableResource):
        a_str: str

    @run_status_sensor(run_status=DagsterRunStatus.SUCCESS)
    def status_sensor(context, my_resource: MyResource):
        if False:
            return 10
        assert context.dagster_event.event_type_value == 'PIPELINE_SUCCESS'
        assert my_resource.a_str == 'bar'

    @run_status_sensor(run_status=DagsterRunStatus.SUCCESS)
    def status_sensor_no_context(my_resource: MyResource):
        if False:
            print('Hello World!')
        assert my_resource.a_str == 'bar'

    @op
    def succeeds():
        if False:
            while True:
                i = 10
        return 1

    @job
    def my_job_2():
        if False:
            for i in range(10):
                print('nop')
        succeeds()
    instance = DagsterInstance.ephemeral()
    result = my_job_2.execute_in_process(instance=instance, raise_on_error=False)
    dagster_run = result.dagster_run
    dagster_event = result.get_job_success_event()
    context = build_run_status_sensor_context(sensor_name='status_sensor', dagster_instance=instance, dagster_run=dagster_run, dagster_event=dagster_event, resources={'my_resource': MyResource(a_str='bar')})
    status_sensor(context)
    status_sensor_no_context(context)

def test_run_status_sensor_invocation_resources_direct() -> None:
    if False:
        while True:
            i = 10

    class MyResource(ConfigurableResource):
        a_str: str

    @run_status_sensor(run_status=DagsterRunStatus.SUCCESS)
    def status_sensor(context, my_resource: MyResource):
        if False:
            print('Hello World!')
        assert context.dagster_event.event_type_value == 'PIPELINE_SUCCESS'
        assert my_resource.a_str == 'bar'

    @run_status_sensor(run_status=DagsterRunStatus.SUCCESS)
    def status_sensor_no_context(my_resource: MyResource):
        if False:
            i = 10
            return i + 15
        assert my_resource.a_str == 'bar'

    @op
    def succeeds():
        if False:
            while True:
                i = 10
        return 1

    @job
    def my_job_2():
        if False:
            print('Hello World!')
        succeeds()
    instance = DagsterInstance.ephemeral()
    result = my_job_2.execute_in_process(instance=instance, raise_on_error=False)
    dagster_run = result.dagster_run
    dagster_event = result.get_job_success_event()
    context = build_run_status_sensor_context(sensor_name='status_sensor', dagster_instance=instance, dagster_run=dagster_run, dagster_event=dagster_event)
    status_sensor(context, my_resource=MyResource(a_str='bar'))
    status_sensor_no_context(context, my_resource=MyResource(a_str='bar'))

def test_run_failure_sensor_invocation_resources() -> None:
    if False:
        return 10

    class MyResource(ConfigurableResource):
        a_str: str

    @run_failure_sensor
    def failure_sensor(context, my_resource: MyResource):
        if False:
            while True:
                i = 10
        assert context.dagster_event.event_type_value == 'PIPELINE_SUCCESS'
        assert my_resource.a_str == 'bar'

    @run_failure_sensor
    def failure_sensor_no_context(my_resource: MyResource):
        if False:
            i = 10
            return i + 15
        assert my_resource.a_str == 'bar'

    @op
    def succeeds():
        if False:
            return 10
        return 1

    @job
    def my_job_2():
        if False:
            return 10
        succeeds()
    instance = DagsterInstance.ephemeral()
    result = my_job_2.execute_in_process(instance=instance, raise_on_error=False)
    dagster_run = result.dagster_run
    dagster_event = result.get_job_success_event()
    context = build_run_status_sensor_context(sensor_name='failure_sensor', dagster_instance=instance, dagster_run=dagster_run, dagster_event=dagster_event, resources={'my_resource': MyResource(a_str='bar')}).for_run_failure()
    failure_sensor(context)
    failure_sensor_no_context(context)

def test_instance_access_built_sensor():
    if False:
        while True:
            i = 10
    with pytest.raises(DagsterInvariantViolationError, match='Attempted to initialize dagster instance, but no instance reference was provided.'):
        build_sensor_context().instance
    with instance_for_test() as instance:
        assert isinstance(build_sensor_context(instance).instance, DagsterInstance)

def test_instance_access_with_mock():
    if False:
        return 10
    mock_instance = mock.MagicMock(spec=DagsterInstance)
    assert build_sensor_context(instance=mock_instance).instance == mock_instance

def test_sensor_w_no_job():
    if False:
        i = 10
        return i + 15

    @sensor()
    def no_job_sensor():
        if False:
            return 10
        return RunRequest(run_key=None, run_config=None, tags=None)
    with pytest.raises(Exception, match='.* Sensor evaluation function returned a RunRequest for a sensor lacking a specified target .*'):
        with build_sensor_context() as context:
            no_job_sensor.evaluate_tick(context)

def test_validated_partitions():
    if False:
        print('Hello World!')

    @job(partitions_def=StaticPartitionsDefinition(['foo', 'bar']))
    def my_job():
        if False:
            return 10
        pass

    @sensor(job=my_job)
    def invalid_req_sensor():
        if False:
            i = 10
            return i + 15
        return RunRequest(partition_key='nonexistent')

    @sensor(job=my_job)
    def valid_req_sensor():
        if False:
            print('Hello World!')
        return RunRequest(partition_key='foo', tags={'yay': 'yay!'})

    @repository
    def my_repo():
        if False:
            print('Hello World!')
        return [my_job, invalid_req_sensor, valid_req_sensor]
    with pytest.raises(DagsterInvariantViolationError, match='Must provide repository def'):
        with build_sensor_context() as context:
            invalid_req_sensor.evaluate_tick(context)
    with pytest.raises(DagsterUnknownPartitionError, match='Could not find a partition'):
        with build_sensor_context(repository_def=my_repo) as context:
            invalid_req_sensor.evaluate_tick(context)
    with build_sensor_context(repository_def=my_repo) as context:
        run_requests = valid_req_sensor.evaluate_tick(context).run_requests
        assert len(run_requests) == 1
        run_request = run_requests[0]
        assert run_request.partition_key == 'foo'
        assert run_request.run_config == {}
        assert run_request.tags.get(PARTITION_NAME_TAG) == 'foo'
        assert run_request.tags.get('yay') == 'yay!'

def test_partitioned_config_run_request():
    if False:
        while True:
            i = 10

    def partition_fn(partition_key: str):
        if False:
            for i in range(10):
                print('nop')
        return {'ops': {'my_op': {'config': {'partition': partition_key}}}}

    @static_partitioned_config(partition_keys=['a', 'b', 'c', 'd'])
    def my_partitioned_config(partition_key: str):
        if False:
            for i in range(10):
                print('nop')
        return partition_fn(partition_key)

    @op
    def my_op():
        if False:
            print('Hello World!')
        pass

    @job(config=my_partitioned_config)
    def my_job():
        if False:
            while True:
                i = 10
        my_op()

    @sensor(job=my_job)
    def valid_req_sensor():
        if False:
            for i in range(10):
                print('nop')
        return RunRequest(partition_key='a', tags={'yay': 'yay!'})

    @sensor(job=my_job)
    def invalid_req_sensor():
        if False:
            print('Hello World!')
        return RunRequest(partition_key='nonexistent')

    @sensor(job_name='my_job')
    def job_str_target_sensor():
        if False:
            for i in range(10):
                print('nop')
        return RunRequest(partition_key='a', tags={'yay': 'yay!'})

    @sensor(job_name='my_job')
    def invalid_job_str_target_sensor():
        if False:
            while True:
                i = 10
        return RunRequest(partition_key='invalid')

    @repository
    def my_repo():
        if False:
            while True:
                i = 10
        return [my_job, valid_req_sensor, invalid_req_sensor, job_str_target_sensor, invalid_job_str_target_sensor]
    with build_sensor_context(repository_def=my_repo) as context:
        for valid_sensor in [valid_req_sensor, job_str_target_sensor]:
            run_requests = valid_sensor.evaluate_tick(context).run_requests
            assert len(run_requests) == 1
            run_request = run_requests[0]
            assert run_request.run_config == partition_fn('a')
            assert run_request.tags.get(PARTITION_NAME_TAG) == 'a'
            assert run_request.tags.get('yay') == 'yay!'
        for invalid_sensor in [invalid_req_sensor, invalid_job_str_target_sensor]:
            with pytest.raises(DagsterUnknownPartitionError, match='Could not find a partition'):
                run_requests = invalid_sensor.evaluate_tick(context).run_requests

def test_asset_selection_run_request_partition_key():
    if False:
        i = 10
        return i + 15

    @sensor(asset_selection=AssetSelection.keys('a_asset'))
    def valid_req_sensor():
        if False:
            print('Hello World!')
        return RunRequest(partition_key='a')

    @sensor(asset_selection=AssetSelection.keys('a_asset'))
    def invalid_req_sensor():
        if False:
            return 10
        return RunRequest(partition_key='b')

    @asset(partitions_def=StaticPartitionsDefinition(['a']))
    def a_asset():
        if False:
            print('Hello World!')
        return 1
    daily_partitions_def = DailyPartitionsDefinition('2023-01-01')

    @asset(partitions_def=daily_partitions_def)
    def b_asset():
        if False:
            print('Hello World!')
        return 1

    @asset(partitions_def=daily_partitions_def)
    def c_asset():
        if False:
            return 10
        return 1

    @repository
    def my_repo():
        if False:
            for i in range(10):
                print('nop')
        return [a_asset, b_asset, c_asset, valid_req_sensor, invalid_req_sensor, define_asset_job('a_job', [a_asset]), define_asset_job('b_job', [b_asset])]
    with build_sensor_context(repository_def=my_repo) as context:
        run_requests = valid_req_sensor.evaluate_tick(context).run_requests
        assert len(run_requests) == 1
        assert run_requests[0].partition_key == 'a'
        assert run_requests[0].tags.get(PARTITION_NAME_TAG) == 'a'
        assert run_requests[0].asset_selection == [a_asset.key]
        with pytest.raises(DagsterUnknownPartitionError, match='Could not find a partition'):
            invalid_req_sensor.evaluate_tick(context)

def test_run_status_sensor():
    if False:
        for i in range(10):
            print('nop')

    @run_status_sensor(run_status=DagsterRunStatus.SUCCESS)
    def status_sensor(context):
        if False:
            return 10
        assert context.dagster_event.event_type_value == 'PIPELINE_SUCCESS'

    @op
    def succeeds():
        if False:
            i = 10
            return i + 15
        return 1

    @job
    def my_job_2():
        if False:
            return 10
        succeeds()
    instance = DagsterInstance.ephemeral()
    result = my_job_2.execute_in_process(instance=instance, raise_on_error=False)
    dagster_run = result.dagster_run
    dagster_event = result.get_job_success_event()
    context = build_run_status_sensor_context(sensor_name='status_sensor', dagster_instance=instance, dagster_run=dagster_run, dagster_event=dagster_event)
    status_sensor(context)

def test_run_failure_sensor():
    if False:
        print('Hello World!')

    @run_failure_sensor
    def failure_sensor(context):
        if False:
            while True:
                i = 10
        assert context.dagster_event.event_type_value == 'PIPELINE_FAILURE'

    @op
    def will_fail():
        if False:
            while True:
                i = 10
        raise Exception('failure')

    @job
    def my_job():
        if False:
            for i in range(10):
                print('nop')
        will_fail()
    instance = DagsterInstance.ephemeral()
    result = my_job.execute_in_process(instance=instance, raise_on_error=False)
    dagster_run = result.dagster_run
    dagster_event = result.get_job_failure_event()
    context = build_run_status_sensor_context(sensor_name='failure_sensor', dagster_instance=instance, dagster_run=dagster_run, dagster_event=dagster_event).for_run_failure()
    failure_sensor(context)

def test_run_status_sensor_run_request():
    if False:
        while True:
            i = 10

    @op
    def succeeds():
        if False:
            print('Hello World!')
        return 1

    @job
    def my_job_2():
        if False:
            while True:
                i = 10
        succeeds()
    instance = DagsterInstance.ephemeral()
    result = my_job_2.execute_in_process(instance=instance, raise_on_error=False)
    dagster_run = result.dagster_run
    dagster_event = result.get_job_success_event()
    context = build_run_status_sensor_context(sensor_name='status_sensor', dagster_instance=instance, dagster_run=dagster_run, dagster_event=dagster_event)

    @run_status_sensor(run_status=DagsterRunStatus.SUCCESS)
    def basic_sensor(_):
        if False:
            print('Hello World!')
        return RunRequest(run_key=None, run_config={}, tags={})
    assert basic_sensor(context).run_config == {}

    @run_status_sensor(run_status=DagsterRunStatus.SUCCESS)
    def basic_sensor_w_arg(context):
        if False:
            while True:
                i = 10
        assert context.dagster_event.event_type_value == 'PIPELINE_SUCCESS'
        return RunRequest(run_key=None, run_config={}, tags={})
    assert basic_sensor_w_arg(context).run_config == {}

def test_run_failure_w_run_request():
    if False:
        print('Hello World!')

    @op
    def will_fail():
        if False:
            while True:
                i = 10
        raise Exception('failure')

    @job
    def my_job():
        if False:
            for i in range(10):
                print('nop')
        will_fail()
    instance = DagsterInstance.ephemeral()
    result = my_job.execute_in_process(instance=instance, raise_on_error=False)
    dagster_run = result.dagster_run
    dagster_event = result.get_job_failure_event()
    context = build_run_status_sensor_context(sensor_name='failure_sensor', dagster_instance=instance, dagster_run=dagster_run, dagster_event=dagster_event).for_run_failure()

    @run_failure_sensor
    def basic_sensor(_):
        if False:
            while True:
                i = 10
        return RunRequest(run_key=None, run_config={}, tags={})
    assert basic_sensor(context).run_config == {}

    @run_failure_sensor
    def basic_sensor_w_arg(context):
        if False:
            return 10
        assert context.dagster_event.event_type_value == 'PIPELINE_FAILURE'
        return RunRequest(run_key=None, run_config={}, tags={})
    assert basic_sensor_w_arg(context).run_config == {}

def test_freshness_policy_sensor():
    if False:
        return 10

    @freshness_policy_sensor(asset_selection=AssetSelection.all())
    def freshness_sensor(context):
        if False:
            print('Hello World!')
        assert context.minutes_overdue == 10
        assert context.previous_minutes_overdue is None
    context = build_freshness_policy_sensor_context(sensor_name='status_sensor', asset_key=AssetKey('a'), freshness_policy=FreshnessPolicy(maximum_lag_minutes=30), minutes_overdue=10)
    freshness_sensor(context)

def test_freshness_policy_sensor_params_out_of_order():
    if False:
        for i in range(10):
            print('nop')

    @freshness_policy_sensor(name='some_name', asset_selection=AssetSelection.all(), minimum_interval_seconds=10, description='foo')
    def freshness_sensor(context):
        if False:
            i = 10
            return i + 15
        assert context.minutes_overdue == 10
        assert context.previous_minutes_overdue is None
    context = build_freshness_policy_sensor_context(sensor_name='some_name', asset_key=AssetKey('a'), freshness_policy=FreshnessPolicy(maximum_lag_minutes=30), minutes_overdue=10)
    freshness_sensor(context)

def test_multi_asset_sensor():
    if False:
        for i in range(10):
            print('nop')

    @op
    def an_op():
        if False:
            for i in range(10):
                print('nop')
        return 1

    @job
    def the_job():
        if False:
            for i in range(10):
                print('nop')
        an_op()

    @asset
    def asset_a():
        if False:
            for i in range(10):
                print('nop')
        return 1

    @asset
    def asset_b():
        if False:
            for i in range(10):
                print('nop')
        return 1

    @multi_asset_sensor(monitored_assets=[AssetKey('asset_a'), AssetKey('asset_b')], job=the_job)
    def a_and_b_sensor(context):
        if False:
            print('Hello World!')
        asset_events = context.latest_materialization_records_by_key()
        if all(asset_events.values()):
            context.advance_all_cursors()
            return RunRequest(run_key=context.cursor, run_config={})
    defs = Definitions(assets=[asset_a, asset_b], sensors=[a_and_b_sensor])
    my_repo = defs.get_repository_def()
    for (definitions, repository_def) in [(defs, None), (None, my_repo)]:
        with instance_for_test() as instance:
            materialize([asset_a, asset_b], instance=instance)
            ctx = build_multi_asset_sensor_context(monitored_assets=[AssetKey('asset_a'), AssetKey('asset_b')], instance=instance, repository_def=repository_def, definitions=definitions)
            assert a_and_b_sensor(ctx).run_config == {}

def test_multi_asset_sensor_selection():
    if False:
        print('Hello World!')

    @multi_asset(outs={'a': AssetOut(key='asset_a'), 'b': AssetOut(key='asset_b')})
    def two_assets():
        if False:
            while True:
                i = 10
        return (1, 2)

    @multi_asset_sensor(monitored_assets=[AssetKey('asset_a')])
    def passing_sensor(context):
        if False:
            return 10
        pass

    @repository
    def my_repo():
        if False:
            print('Hello World!')
        return [two_assets, passing_sensor]

def test_multi_asset_sensor_has_assets():
    if False:
        i = 10
        return i + 15

    @multi_asset(outs={'a': AssetOut(key='asset_a'), 'b': AssetOut(key='asset_b')})
    def two_assets():
        if False:
            print('Hello World!')
        return (1, 2)

    @multi_asset_sensor(monitored_assets=[AssetKey('asset_a'), AssetKey('asset_b')])
    def passing_sensor(context):
        if False:
            print('Hello World!')
        assert context.assets_defs_by_key[AssetKey('asset_a')] == two_assets
        assert context.assets_defs_by_key[AssetKey('asset_b')] == two_assets
        assert len(context.assets_defs_by_key) == 2

    @repository
    def my_repo():
        if False:
            for i in range(10):
                print('nop')
        return [two_assets, passing_sensor]
    with instance_for_test() as instance:
        ctx = build_multi_asset_sensor_context(monitored_assets=[AssetKey('asset_a'), AssetKey('asset_b')], instance=instance, repository_def=my_repo)
        passing_sensor(ctx)

def test_partitions_multi_asset_sensor_context():
    if False:
        i = 10
        return i + 15
    daily_partitions_def = DailyPartitionsDefinition('2020-01-01')

    @asset(partitions_def=daily_partitions_def)
    def daily_partitions_asset():
        if False:
            print('Hello World!')
        return 1

    @asset(partitions_def=daily_partitions_def)
    def daily_partitions_asset_2():
        if False:
            return 10
        return 1

    @repository
    def my_repo():
        if False:
            i = 10
            return i + 15
        return [daily_partitions_asset, daily_partitions_asset_2]
    asset_job = define_asset_job('yay', selection='daily_partitions_asset', partitions_def=daily_partitions_def)

    @multi_asset_sensor(monitored_assets=[daily_partitions_asset.key, daily_partitions_asset_2.key], job=asset_job)
    def two_asset_sensor(context):
        if False:
            print('Hello World!')
        partition_1 = next(iter(context.latest_materialization_records_by_partition(daily_partitions_asset.key).keys()))
        partition_2 = next(iter(context.latest_materialization_records_by_partition(daily_partitions_asset_2.key).keys()))
        if partition_1 == partition_2:
            context.advance_all_cursors()
            return asset_job.run_request_for_partition(partition_1, run_key=None)
    with instance_for_test() as instance:
        materialize([daily_partitions_asset, daily_partitions_asset_2], partition_key='2022-08-01', instance=instance)
        ctx = build_multi_asset_sensor_context(monitored_assets=[daily_partitions_asset.key, daily_partitions_asset_2.key], instance=instance, repository_def=my_repo)
        sensor_data = two_asset_sensor.evaluate_tick(ctx)
        assert len(sensor_data.run_requests) == 1
        assert sensor_data.run_requests[0].partition_key == '2022-08-01'
        assert sensor_data.run_requests[0].tags['dagster/partition'] == '2022-08-01'
        assert ctx.cursor == '{"AssetKey([\'daily_partitions_asset\'])": ["2022-08-01", 4, {}], "AssetKey([\'daily_partitions_asset_2\'])": ["2022-08-01", 5, {}]}'

@asset(partitions_def=DailyPartitionsDefinition('2022-07-01'))
def july_asset():
    if False:
        return 10
    return 1

@asset(partitions_def=DailyPartitionsDefinition('2022-07-01'))
def july_asset_2():
    if False:
        i = 10
        return i + 15
    return 1

@asset(partitions_def=DailyPartitionsDefinition('2022-08-01'))
def august_asset():
    if False:
        while True:
            i = 10
    return 1

@repository
def my_repo():
    if False:
        print('Hello World!')
    return [july_asset, july_asset_2, august_asset]

def test_multi_asset_sensor_after_cursor_partition_flag():
    if False:
        return 10

    @multi_asset_sensor(monitored_assets=[july_asset.key])
    def after_cursor_partitions_asset_sensor(context):
        if False:
            print('Hello World!')
        events = context.latest_materialization_records_by_key([july_asset.key])
        if events[july_asset.key] and events[july_asset.key].event_log_entry.dagster_event.partition == '2022-07-10':
            context.advance_all_cursors()
        else:
            assert context.get_cursor_partition(july_asset.key) == '2022-07-10'
            materializations_by_key = context.latest_materialization_records_by_key()
            later_materialization = materializations_by_key.get(july_asset.key)
            assert later_materialization
            assert later_materialization.event_log_entry.dagster_event.partition == '2022-07-05'
            materializations_by_partition = context.latest_materialization_records_by_partition(july_asset.key)
            assert list(materializations_by_partition.keys()) == ['2022-07-05']
            materializations_by_partition = context.latest_materialization_records_by_partition(july_asset.key, after_cursor_partition=True)
            assert set(materializations_by_partition.keys()) == set()
    with instance_for_test() as instance:
        materialize([july_asset], partition_key='2022-07-10', instance=instance)
        ctx = build_multi_asset_sensor_context(monitored_assets=[july_asset.key], instance=instance, repository_def=my_repo)
        after_cursor_partitions_asset_sensor(ctx)
        materialize([july_asset], partition_key='2022-07-05', instance=instance)
        after_cursor_partitions_asset_sensor(ctx)

def test_multi_asset_sensor_can_start_from_asset_sensor_cursor():
    if False:
        for i in range(10):
            print('nop')

    @asset
    def my_asset():
        if False:
            return 10
        return Output(99)

    @job
    def my_job():
        if False:
            while True:
                i = 10
        pass

    @asset_sensor(asset_key=my_asset.key, job=my_job)
    def my_asset_sensor(context):
        if False:
            while True:
                i = 10
        return RunRequest(run_key=context.cursor, run_config={})

    @multi_asset_sensor(monitored_assets=[my_asset.key])
    def my_multi_asset_sensor(context):
        if False:
            return 10
        ctx.advance_all_cursors()
    with instance_for_test() as instance:
        ctx = build_sensor_context(instance=instance)
        materialize([my_asset], instance=instance)
        my_asset_sensor.evaluate_tick(ctx)
        assert ctx.cursor == '3'
        ctx = build_multi_asset_sensor_context(monitored_assets=[my_asset.key], instance=instance, repository_def=my_repo, cursor=ctx.cursor)
        my_multi_asset_sensor(ctx)
        assert ctx.cursor == '{"AssetKey([\'my_asset\'])": [null, 3, {}]}'

def test_multi_asset_sensor_all_partitions_materialized():
    if False:
        print('Hello World!')

    @multi_asset_sensor(monitored_assets=[july_asset.key])
    def asset_sensor(context):
        if False:
            while True:
                i = 10
        assert context.all_partitions_materialized(july_asset.key) is False
        assert context.all_partitions_materialized(july_asset.key, ['2022-07-10', '2022-07-11']) is True
    with instance_for_test() as instance:
        materialize([july_asset], partition_key='2022-07-10', instance=instance)
        materialize([july_asset], partition_key='2022-07-11', instance=instance)
        ctx = build_multi_asset_sensor_context(monitored_assets=[july_asset.key], instance=instance, repository_def=my_repo)
        asset_sensor(ctx)

def test_multi_asset_sensor_partition_mapping():
    if False:
        print('Hello World!')

    @asset(partitions_def=DailyPartitionsDefinition('2022-07-01'))
    def july_daily_partitions():
        if False:
            return 10
        return 1

    @asset(partitions_def=DailyPartitionsDefinition('2022-08-01'))
    def downstream_daily_partitions(july_daily_partitions):
        if False:
            i = 10
            return i + 15
        return 1

    @repository
    def my_repo():
        if False:
            i = 10
            return i + 15
        return [july_daily_partitions, downstream_daily_partitions]

    @multi_asset_sensor(monitored_assets=[july_daily_partitions.key])
    def asset_sensor(context):
        if False:
            print('Hello World!')
        for partition_key in context.latest_materialization_records_by_partition(july_daily_partitions.key).keys():
            for downstream_partition in context.get_downstream_partition_keys(partition_key, to_asset_key=downstream_daily_partitions.key, from_asset_key=july_daily_partitions.key):
                assert downstream_partition == '2022-08-10'
    with instance_for_test() as instance:
        materialize([july_daily_partitions], partition_key='2022-08-10', instance=instance)
        ctx = build_multi_asset_sensor_context(monitored_assets=[july_daily_partitions.key], instance=instance, repository_def=my_repo)
        asset_sensor(ctx)

def test_multi_asset_sensor_retains_ordering_and_fetches_latest_per_partition():
    if False:
        i = 10
        return i + 15
    partition_ordering = ['2022-07-15', '2022-07-14', '2022-07-13', '2022-07-12', '2022-07-15']

    @multi_asset_sensor(monitored_assets=[july_asset.key])
    def asset_sensor(context):
        if False:
            return 10
        assert list(context.latest_materialization_records_by_partition(july_asset.key).keys()) == partition_ordering[1:]
    with instance_for_test() as instance:
        for partition in partition_ordering:
            materialize([july_asset], partition_key=partition, instance=instance)
        ctx = build_multi_asset_sensor_context(monitored_assets=[july_asset.key], instance=instance, repository_def=my_repo)
        asset_sensor(ctx)

def test_multi_asset_sensor_update_cursor_no_overwrite():
    if False:
        print('Hello World!')

    @multi_asset_sensor(monitored_assets=[july_asset.key, august_asset.key])
    def after_cursor_partitions_asset_sensor(context):
        if False:
            print('Hello World!')
        events = context.latest_materialization_records_by_key()
        if events[july_asset.key] and events[july_asset.key].event_log_entry.dagster_event.partition == '2022-07-10':
            context.advance_cursor({july_asset.key: events[july_asset.key]})
        else:
            materialization = events[august_asset.key]
            assert materialization
            context.advance_cursor({august_asset.key: materialization})
            assert context._get_cursor(july_asset.key).latest_consumed_event_partition == '2022-07-10'
    with instance_for_test() as instance:
        materialize([july_asset], partition_key='2022-07-10', instance=instance)
        ctx = build_multi_asset_sensor_context(monitored_assets=[july_asset.key, august_asset.key], instance=instance, repository_def=my_repo)
        after_cursor_partitions_asset_sensor(ctx)
        materialize([august_asset], partition_key='2022-08-05', instance=instance)
        after_cursor_partitions_asset_sensor(ctx)

def test_multi_asset_sensor_no_unconsumed_events():
    if False:
        i = 10
        return i + 15

    @multi_asset_sensor(monitored_assets=[july_asset.key, july_asset_2.key])
    def my_sensor(context):
        if False:
            i = 10
            return i + 15
        context.latest_materialization_records_by_partition_and_asset()
        assert context._initial_unconsumed_events_by_id == {}
    with instance_for_test() as instance:
        materialize([july_asset], partition_key='2022-08-04', instance=instance)
        ctx = build_multi_asset_sensor_context(monitored_assets=[july_asset.key, july_asset_2.key], instance=instance, repository_def=my_repo)
        my_sensor(ctx)

def test_multi_asset_sensor_latest_materialization_records_by_partition_and_asset():
    if False:
        i = 10
        return i + 15

    @multi_asset_sensor(monitored_assets=[july_asset.key, july_asset_2.key])
    def my_sensor(context):
        if False:
            while True:
                i = 10
        events = context.latest_materialization_records_by_partition_and_asset()
        for (partition_key, materialization_by_asset) in events.items():
            assert partition_key == '2022-08-04'
            assert len(materialization_by_asset) == 2
            assert july_asset.key in materialization_by_asset
            assert july_asset_2.key in materialization_by_asset
    with instance_for_test() as instance:
        materialize([july_asset_2, july_asset], partition_key='2022-08-04', instance=instance)
        materialize([july_asset], partition_key='2022-08-04', instance=instance)
        ctx = build_multi_asset_sensor_context(monitored_assets=[july_asset.key, july_asset_2.key], instance=instance, repository_def=my_repo)
        my_sensor(ctx)

def test_build_multi_asset_sensor_context_asset_selection():
    if False:
        while True:
            i = 10
    from dagster_tests.asset_defs_tests.test_asset_selection import alice, bob, candace, danny, earth, edgar, fiona, george

    @multi_asset_sensor(monitored_assets=AssetSelection.groups('ladies').upstream(depth=1, include_self=False))
    def asset_selection_sensor(context):
        if False:
            while True:
                i = 10
        assert context.asset_keys == [danny.key]

    @repository
    def my_repo():
        if False:
            return 10
        return [earth, alice, bob, candace, danny, edgar, fiona, george, asset_selection_sensor]
    with instance_for_test() as instance:
        ctx = build_multi_asset_sensor_context(monitored_assets=AssetSelection.groups('ladies').upstream(depth=1, include_self=False), instance=instance, repository_def=my_repo)
        asset_selection_sensor(ctx)

def test_multi_asset_sensor_unconsumed_events():
    if False:
        i = 10
        return i + 15
    invocation_num = 0

    @multi_asset_sensor(monitored_assets=[july_asset.key])
    def test_unconsumed_events_sensor(context):
        if False:
            return 10
        if invocation_num == 0:
            events = context.latest_materialization_records_by_partition(july_asset.key)
            assert len(events) == 2
            context.advance_cursor({july_asset.key: events['2022-07-10']})
        if invocation_num == 1:
            events = context.latest_materialization_records_by_partition(july_asset.key)
            assert len(events) == 1
            context.advance_cursor({july_asset.key: events['2022-07-05']})
    with instance_for_test() as instance:
        materialize([july_asset], partition_key='2022-07-05', instance=instance)
        materialize([july_asset], partition_key='2022-07-10', instance=instance)
        materialize([july_asset], partition_key='2022-07-10', instance=instance)
        event_records = list(instance.get_event_records(EventRecordsFilter(DagsterEventType.ASSET_MATERIALIZATION), ascending=True))
        assert len(event_records) == 3
        first_2022_07_10_mat = event_records[1].storage_id
        unconsumed_storage_id = event_records[0].storage_id
        assert first_2022_07_10_mat > unconsumed_storage_id
        assert first_2022_07_10_mat < event_records[2].storage_id
        ctx = build_multi_asset_sensor_context(monitored_assets=[july_asset.key], instance=instance, repository_def=my_repo)
        test_unconsumed_events_sensor(ctx)
        july_asset_cursor = ctx._get_cursor(july_asset.key)
        assert first_2022_07_10_mat < july_asset_cursor.latest_consumed_event_id
        assert july_asset_cursor.latest_consumed_event_partition == '2022-07-10'
        assert july_asset_cursor.trailing_unconsumed_partitioned_event_ids == {'2022-07-05': unconsumed_storage_id}
        invocation_num += 1
        test_unconsumed_events_sensor(ctx)
        second_july_cursor = ctx._get_cursor(july_asset.key)
        assert second_july_cursor.latest_consumed_event_partition == '2022-07-10'
        assert second_july_cursor.latest_consumed_event_id == july_asset_cursor.latest_consumed_event_id
        assert second_july_cursor.trailing_unconsumed_partitioned_event_ids == {}

def test_advance_all_cursors_clears_unconsumed_events():
    if False:
        return 10
    invocation_num = 0

    @multi_asset_sensor(monitored_assets=[july_asset.key])
    def test_unconsumed_events_sensor(context):
        if False:
            while True:
                i = 10
        if invocation_num == 0:
            events = context.latest_materialization_records_by_partition(july_asset.key)
            assert len(events) == 2
            context.advance_cursor({july_asset.key: events['2022-07-10']})
        if invocation_num == 1:
            events = context.latest_materialization_records_by_partition(july_asset.key)
            assert len(events) == 2
            unconsumed_events = context.get_trailing_unconsumed_events(july_asset.key)
            assert len(unconsumed_events) == 1
            assert events['2022-07-05'] == unconsumed_events[0]
            context.advance_all_cursors()
    with instance_for_test() as instance:
        materialize([july_asset], partition_key='2022-07-05', instance=instance)
        materialize([july_asset], partition_key='2022-07-10', instance=instance)
        ctx = build_multi_asset_sensor_context(monitored_assets=[july_asset.key], instance=instance, repository_def=my_repo)
        test_unconsumed_events_sensor(ctx)
        july_asset_cursor = ctx._get_cursor(july_asset.key)
        first_storage_id = july_asset_cursor.latest_consumed_event_id
        assert first_storage_id
        assert july_asset_cursor.latest_consumed_event_partition == '2022-07-10'
        assert len(july_asset_cursor.trailing_unconsumed_partitioned_event_ids) == 1
        invocation_num += 1
        materialize([july_asset], partition_key='2022-07-06', instance=instance)
        test_unconsumed_events_sensor(ctx)
        july_asset_cursor = ctx._get_cursor(july_asset.key)
        assert july_asset_cursor.latest_consumed_event_partition == '2022-07-06'
        assert july_asset_cursor.trailing_unconsumed_partitioned_event_ids == {}
        assert july_asset_cursor.latest_consumed_event_id > first_storage_id

def test_error_when_max_num_unconsumed_events():
    if False:
        i = 10
        return i + 15

    @multi_asset_sensor(monitored_assets=[july_asset.key])
    def test_unconsumed_events_sensor(context):
        if False:
            i = 10
            return i + 15
        latest_record = context.materialization_records_for_key(july_asset.key, limit=25)
        context.advance_cursor({july_asset.key: latest_record[-1]})
    with instance_for_test() as instance:
        for num in range(1, 26):
            str_num = '0' + str(num) if num < 10 else str(num)
            materialize([july_asset], partition_key=f'2022-07-{str_num}', instance=instance)
        ctx = build_multi_asset_sensor_context(monitored_assets=[july_asset.key], instance=instance, repository_def=my_repo)
        test_unconsumed_events_sensor(ctx)
        july_asset_cursor = ctx._get_cursor(july_asset.key)
        assert july_asset_cursor.latest_consumed_event_id
        assert july_asset_cursor.latest_consumed_event_partition == '2022-07-25'
        assert len(july_asset_cursor.trailing_unconsumed_partitioned_event_ids) == 24
        for date in ['26', '27', '28']:
            materialize([july_asset], partition_key=f'2022-07-{date}', instance=instance)
        with pytest.raises(DagsterInvariantViolationError, match='maximum number of trailing unconsumed events'):
            test_unconsumed_events_sensor(ctx)

def test_latest_materialization_records_by_partition_fetches_unconsumed_events():
    if False:
        print('Hello World!')
    invocation_num = 0

    @multi_asset_sensor(monitored_assets=[july_asset.key])
    def test_unconsumed_events_sensor(context):
        if False:
            for i in range(10):
                print('nop')
        if invocation_num == 0:
            context.advance_cursor({july_asset.key: context.latest_materialization_records_by_partition(july_asset.key)['2022-07-03']})
        if invocation_num == 1:
            records_dict = context.latest_materialization_records_by_partition(july_asset.key)
            ordered_records = list(enumerate(records_dict))
            get_partition_key_from_ordered_record = lambda record: record[1]
            assert [get_partition_key_from_ordered_record(record) for record in ordered_records] == ['2022-07-01', '2022-07-04', '2022-07-02']
            for event_log_entry in records_dict.values():
                context.advance_cursor({july_asset.key: event_log_entry})
    with instance_for_test() as instance:
        for date in ['01', '02', '03']:
            materialize([july_asset], partition_key=f'2022-07-{date}', instance=instance)
        ctx = build_multi_asset_sensor_context(monitored_assets=[july_asset.key], instance=instance, repository_def=my_repo)
        test_unconsumed_events_sensor(ctx)
        first_july_cursor = ctx._get_cursor(july_asset.key)
        assert first_july_cursor.latest_consumed_event_id
        assert first_july_cursor.latest_consumed_event_partition == '2022-07-03'
        assert len(first_july_cursor.trailing_unconsumed_partitioned_event_ids) == 2
        invocation_num += 1
        for date in ['04', '02']:
            materialize([july_asset], partition_key=f'2022-07-{date}', instance=instance)
        test_unconsumed_events_sensor(ctx)
        second_july_cursor = ctx._get_cursor(july_asset.key)
        assert second_july_cursor.latest_consumed_event_partition == '2022-07-02'
        assert second_july_cursor.latest_consumed_event_id > first_july_cursor.latest_consumed_event_id
        assert len(second_july_cursor.trailing_unconsumed_partitioned_event_ids.keys()) == 0

def test_unfetched_partitioned_events_are_unconsumed():
    if False:
        return 10

    @multi_asset_sensor(monitored_assets=[july_asset.key])
    def test_unconsumed_events_sensor(context):
        if False:
            for i in range(10):
                print('nop')
        context.advance_cursor({july_asset.key: context.latest_materialization_records_by_partition(july_asset.key)['2022-07-05']})
    with instance_for_test() as instance:
        for _ in range(5):
            materialize([july_asset], partition_key='2022-07-04', instance=instance)
            materialize([july_asset], partition_key='2022-07-05', instance=instance)
        ctx = build_multi_asset_sensor_context(monitored_assets=[july_asset.key], instance=instance, repository_def=my_repo)
        test_unconsumed_events_sensor(ctx)
        first_july_cursor = ctx._get_cursor(july_asset.key)
        assert first_july_cursor.latest_consumed_event_id
        assert first_july_cursor.latest_consumed_event_partition == '2022-07-05'
        mats_2022_07_04 = list(instance.get_event_records(EventRecordsFilter(DagsterEventType.ASSET_MATERIALIZATION, asset_partitions=['2022-07-04'])))
        assert first_july_cursor.trailing_unconsumed_partitioned_event_ids['2022-07-04'] == mats_2022_07_04[0].storage_id
        materialize([july_asset], partition_key='2022-07-04', instance=instance)
        materialize([july_asset], partition_key='2022-07-05', instance=instance)
        ctx = build_multi_asset_sensor_context(monitored_assets=[july_asset.key], instance=instance, repository_def=my_repo)
        test_unconsumed_events_sensor(ctx)
        second_july_cursor = ctx._get_cursor(july_asset.key)
        assert second_july_cursor.latest_consumed_event_id > first_july_cursor.latest_consumed_event_id
        assert second_july_cursor.latest_consumed_event_partition == '2022-07-05'
        assert second_july_cursor.trailing_unconsumed_partitioned_event_ids['2022-07-04'] > first_july_cursor.trailing_unconsumed_partitioned_event_ids['2022-07-04']

def test_build_multi_asset_sensor_context_asset_selection_set_to_latest_materializations():
    if False:
        return 10

    @asset
    def my_asset():
        if False:
            while True:
                i = 10
        pass

    @multi_asset_sensor(monitored_assets=[my_asset.key])
    def my_sensor(context):
        if False:
            print('Hello World!')
        my_asset_cursor = context._get_cursor(my_asset.key)
        assert my_asset_cursor.latest_consumed_event_id is not None

    @repository
    def my_repo():
        if False:
            return 10
        return [my_asset, my_sensor]
    with instance_for_test() as instance:
        result = materialize([my_asset], instance=instance)
        records = next(iter(instance.get_event_records(EventRecordsFilter(DagsterEventType.ASSET_MATERIALIZATION))))
        assert records.event_log_entry.run_id == result.run_id
        ctx = build_multi_asset_sensor_context(monitored_assets=AssetSelection.groups('default'), instance=instance, cursor_from_latest_materializations=True, repository_def=my_repo)
        assert ctx._get_cursor(my_asset.key).latest_consumed_event_id == records.storage_id
        my_sensor(ctx)

def test_build_multi_asset_sensor_context_set_to_latest_materializations():
    if False:
        return 10
    evaluated = False

    @asset
    def my_asset():
        if False:
            while True:
                i = 10
        return Output(1, metadata={'evaluated': evaluated})

    @multi_asset_sensor(monitored_assets=[my_asset.key])
    def my_sensor(context):
        if False:
            print('Hello World!')
        if not evaluated:
            assert context.latest_materialization_records_by_key()[my_asset.key] is None
        else:
            assert context.latest_materialization_records_by_key()[my_asset.key].event_log_entry.dagster_event.materialization.metadata['evaluated'] == MetadataValue.bool(True)

    @repository
    def my_repo():
        if False:
            while True:
                i = 10
        return [my_asset, my_sensor]
    with instance_for_test() as instance:
        result = materialize([my_asset], instance=instance)
        records = next(iter(instance.get_event_records(EventRecordsFilter(DagsterEventType.ASSET_MATERIALIZATION))))
        assert records.event_log_entry.run_id == result.run_id
        ctx = build_multi_asset_sensor_context(monitored_assets=[my_asset.key], instance=instance, cursor_from_latest_materializations=True, repository_def=my_repo)
        assert ctx._get_cursor(my_asset.key).latest_consumed_event_id == records.storage_id
        my_sensor(ctx)
        evaluated = True
        materialize([my_asset], instance=instance)
        my_sensor(ctx)

def test_build_multi_asset_context_set_after_multiple_materializations():
    if False:
        for i in range(10):
            print('nop')

    @asset
    def my_asset():
        if False:
            print('Hello World!')
        return 1

    @asset
    def my_asset_2():
        if False:
            i = 10
            return i + 15
        return 1

    @repository
    def my_repo():
        if False:
            for i in range(10):
                print('nop')
        return [my_asset, my_asset_2]
    with instance_for_test() as instance:
        materialize([my_asset], instance=instance)
        materialize([my_asset_2], instance=instance)
        records = sorted(list(instance.get_event_records(EventRecordsFilter(DagsterEventType.ASSET_MATERIALIZATION))), key=lambda x: x.storage_id)
        assert len(records) == 2
        my_asset_cursor = records[0].storage_id
        my_asset_2_cursor = records[1].storage_id
        ctx = build_multi_asset_sensor_context(monitored_assets=[my_asset.key, my_asset_2.key], instance=instance, cursor_from_latest_materializations=True, repository_def=my_repo)
        assert ctx._get_cursor(my_asset.key).latest_consumed_event_id == my_asset_cursor
        assert ctx._get_cursor(my_asset_2.key).latest_consumed_event_id == my_asset_2_cursor

def test_error_exec_in_process_to_build_multi_asset_sensor_context():
    if False:
        return 10

    @asset
    def my_asset():
        if False:
            i = 10
            return i + 15
        return 1

    @repository
    def my_repo():
        if False:
            for i in range(10):
                print('nop')
        return [my_asset]
    with pytest.raises(DagsterInvalidInvocationError, match='Dagster instance'):
        with instance_for_test() as instance:
            materialize([my_asset], instance=instance)
            build_multi_asset_sensor_context(monitored_assets=[my_asset.key], repository_def=my_repo, cursor_from_latest_materializations=True)
    with pytest.raises(DagsterInvalidInvocationError, match='Cannot provide both cursor and cursor_from_latest_materializations'):
        with instance_for_test() as instance:
            materialize([my_asset], instance=instance)
            build_multi_asset_sensor_context(monitored_assets=[my_asset.key], repository_def=my_repo, cursor_from_latest_materializations=True, cursor='alskdjalsjk')

def test_error_not_thrown_for_skip_reason():
    if False:
        i = 10
        return i + 15

    @multi_asset_sensor(monitored_assets=[july_asset.key])
    def test_unconsumed_events_sensor(_):
        if False:
            for i in range(10):
                print('nop')
        return SkipReason('I am skipping')
    with instance_for_test() as instance:
        ctx = build_multi_asset_sensor_context(monitored_assets=[july_asset.key], repository_def=my_repo, instance=instance)
        test_unconsumed_events_sensor(ctx)

def test_dynamic_partitions_sensor():
    if False:
        for i in range(10):
            print('nop')
    dynamic_partitions_def = DynamicPartitionsDefinition(name='fruits')

    @asset(partitions_def=dynamic_partitions_def)
    def fruits_asset():
        if False:
            while True:
                i = 10
        return 1
    my_job = define_asset_job('fruits_job', [fruits_asset], partitions_def=dynamic_partitions_def)

    @repository
    def my_repo():
        if False:
            while True:
                i = 10
        return [fruits_asset]

    @sensor(job=my_job)
    def test_sensor(context):
        if False:
            for i in range(10):
                print('nop')
        context.instance.add_dynamic_partitions(dynamic_partitions_def.name, ['apple'])
        return RunRequest(partition_key='apple')
    with instance_for_test() as instance:
        ctx = build_sensor_context(repository_def=my_repo, instance=instance)
        run_request = test_sensor(ctx)
        assert run_request.partition_key == 'apple'

def test_sensor_invocation_runconfig() -> None:
    if False:
        while True:
            i = 10

    class MyConfig(Config):
        a_str: str
        an_int: int

    @sensor(job_name='foo_job')
    def basic_sensor():
        if False:
            while True:
                i = 10
        return RunRequest(run_key=None, run_config=RunConfig(ops={'foo': MyConfig(a_str='foo', an_int=55)}), tags={})
    assert cast(RunRequest, basic_sensor()).run_config.get('ops', {}) == {'foo': {'config': {'a_str': 'foo', 'an_int': 55}}}