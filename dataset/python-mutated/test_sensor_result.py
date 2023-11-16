import pytest
from dagster import AssetKey, AssetMaterialization, DagsterInvariantViolationError, Output, RunRequest, SkipReason, asset_sensor, build_sensor_context, job, op, sensor
from dagster._annotations import get_experimental_params
from dagster._check import CheckError
from dagster._core.definitions.asset_check_evaluation import AssetCheckEvaluation
from dagster._core.definitions.events import AssetObservation
from dagster._core.definitions.run_request import SensorResult
from dagster._core.instance import DagsterInstance
from dagster._core.test_utils import instance_for_test

@op
def do_something():
    if False:
        print('Hello World!')
    pass

@job
def do_something_job():
    if False:
        for i in range(10):
            print('nop')
    do_something()

def test_sensor_result_one_run_request():
    if False:
        print('Hello World!')

    @sensor(job=do_something_job)
    def test_sensor(_):
        if False:
            i = 10
            return i + 15
        return SensorResult(run_requests=[RunRequest(run_key='foo')])
    with instance_for_test() as instance:
        ctx = build_sensor_context(instance=instance)
        sensor_data = test_sensor.evaluate_tick(ctx)
        assert len(sensor_data.run_requests) == 1
        assert sensor_data.run_requests[0].run_key == 'foo'
        assert not sensor_data.skip_message
        assert not sensor_data.dagster_run_reactions
        assert not sensor_data.cursor

def test_sensor_result_skip_reason():
    if False:
        while True:
            i = 10
    skip_reason = SkipReason("I'm skipping")

    @sensor(job=do_something_job)
    def test_sensor(_):
        if False:
            i = 10
            return i + 15
        return [SensorResult(skip_reason=skip_reason)]
    with instance_for_test() as instance:
        ctx = build_sensor_context(instance=instance)
        sensor_data = test_sensor.evaluate_tick(ctx)
        assert not sensor_data.run_requests
        assert sensor_data.skip_message == skip_reason.skip_message
        assert not sensor_data.dagster_run_reactions
        assert not sensor_data.cursor

def test_sensor_result_string_skip_reason():
    if False:
        return 10
    skip_reason = "I'm skipping"

    @sensor(job=do_something_job)
    def test_sensor(_):
        if False:
            print('Hello World!')
        return [SensorResult(skip_reason=skip_reason)]
    with instance_for_test() as instance:
        ctx = build_sensor_context(instance=instance)
        sensor_data = test_sensor.evaluate_tick(ctx)
        assert not sensor_data.run_requests
        assert sensor_data.skip_message == skip_reason
        assert not sensor_data.dagster_run_reactions
        assert not sensor_data.cursor

def test_invalid_skip_reason_invocations():
    if False:
        while True:
            i = 10

    @sensor(job=do_something_job)
    def multiple_sensor_results(_):
        if False:
            i = 10
            return i + 15
        return [SensorResult(skip_reason=SkipReason("I'm skipping")), SensorResult(skip_reason=SkipReason("I'm skipping"))]

    @sensor(job=do_something_job)
    def sensor_result_w_other_objects(_):
        if False:
            print('Hello World!')
        return [SensorResult(run_requests=[RunRequest(run_key='foo')]), RunRequest(run_key='foo')]

    @sensor(job=do_something_job)
    def invalid_sensor_result(_):
        if False:
            return 10
        return [SensorResult(run_requests=[RunRequest(run_key='foo')], skip_reason=SkipReason('aklsdj'))]
    with instance_for_test() as instance:
        ctx = build_sensor_context(instance=instance)
        with pytest.raises(CheckError, match='When a SensorResult is returned from a sensor, it must be the only object returned.'):
            multiple_sensor_results.evaluate_tick(ctx)
        with pytest.raises(CheckError, match='When a SensorResult is returned from a sensor, it must be the only object returned.'):
            sensor_result_w_other_objects.evaluate_tick(ctx)
        with pytest.raises(CheckError, match='Expected a single skip reason or one or more run requests'):
            invalid_sensor_result.evaluate_tick(ctx)

def test_update_cursor():
    if False:
        while True:
            i = 10

    @sensor(job=do_something_job)
    def test_sensor(_):
        if False:
            i = 10
            return i + 15
        return [SensorResult([RunRequest('foo')], cursor='foo')]
    with instance_for_test() as instance:
        ctx = build_sensor_context(instance=instance)
        result = test_sensor.evaluate_tick(ctx)
        assert result.cursor == 'foo'

def test_update_cursor_and_sensor_result_cursor():
    if False:
        while True:
            i = 10

    @sensor(job=do_something_job)
    def test_sensor(context):
        if False:
            i = 10
            return i + 15
        context.update_cursor('bar')
        return [SensorResult([RunRequest('foo')], cursor='foo')]
    with instance_for_test() as instance:
        ctx = build_sensor_context(instance=instance)
        with pytest.raises(DagsterInvariantViolationError, match='cannot be set if context.update_cursor()'):
            test_sensor.evaluate_tick(ctx)

def test_sensor_result_asset_sensor():
    if False:
        print('Hello World!')
    observed = {}

    @op
    def my_table_materialization():
        if False:
            i = 10
            return i + 15
        yield AssetMaterialization('my_table')
        yield Output(1)

    @job
    def my_table_job():
        if False:
            print('Hello World!')
        my_table_materialization()

    @asset_sensor(asset_key=AssetKey('my_table'), job=do_something_job)
    def my_asset_sensor(context, asset_event):
        if False:
            i = 10
            return i + 15
        observed['cursor'] = context.cursor
        return SensorResult([RunRequest('foo')])

    @asset_sensor(asset_key=AssetKey('my_table'), job=do_something_job)
    def asset_sensor_set_cursor(context, asset_event):
        if False:
            while True:
                i = 10
        return SensorResult([RunRequest('foo')], cursor='foo')
    with instance_for_test() as instance:
        my_table_job.execute_in_process(instance=instance)
        with build_sensor_context(instance=instance) as ctx:
            result = my_asset_sensor.evaluate_tick(ctx)
            assert len(result.run_requests) == 1
            assert result.run_requests[0].run_key == 'foo'
            assert result.cursor != observed['cursor']
        with build_sensor_context(instance=instance) as ctx:
            with pytest.raises(DagsterInvariantViolationError, match='The cursor is managed by the asset sensor'):
                asset_sensor_set_cursor.evaluate_tick(ctx)

def test_yield_and_return():
    if False:
        return 10

    @job
    def job1():
        if False:
            for i in range(10):
                print('nop')
        pass

    @sensor(job=job1)
    def sensor_with_yield_run_request_and_return_skip_reason(context):
        if False:
            return 10
        if context.cursor == 'skip':
            return SkipReason('This is a skip reason')
        else:
            yield RunRequest()
    result_with_skip = sensor_with_yield_run_request_and_return_skip_reason.evaluate_tick(build_sensor_context(cursor='skip'))
    assert result_with_skip.skip_message == 'This is a skip reason'
    assert result_with_skip.run_requests == []
    result_without_skip = sensor_with_yield_run_request_and_return_skip_reason.evaluate_tick(build_sensor_context(cursor='go'))
    assert result_without_skip.skip_message is None
    assert len(result_without_skip.run_requests) == 1

    @sensor(job=job1)
    def sensor_with_yield_and_return_run_request(context):
        if False:
            while True:
                i = 10
        yield RunRequest()
        return RunRequest()
    result_yield_and_return_run_request = sensor_with_yield_and_return_run_request.evaluate_tick(build_sensor_context())
    assert len(result_yield_and_return_run_request.run_requests) == 2

def test_asset_events_experimental_param_on_sensor_result() -> None:
    if False:
        i = 10
        return i + 15
    assert 'asset_events' in get_experimental_params(SensorResult)

def test_asset_materialization_in_sensor() -> None:
    if False:
        print('Hello World!')

    @sensor()
    def a_sensor() -> SensorResult:
        if False:
            print('Hello World!')
        return SensorResult(asset_events=[AssetMaterialization('asset_one')])
    instance = DagsterInstance.ephemeral()
    sensor_execution_data = a_sensor.evaluate_tick(build_sensor_context(instance=instance))
    assert len(sensor_execution_data.asset_events) == 1
    output_mat = sensor_execution_data.asset_events[0]
    assert isinstance(output_mat, AssetMaterialization)
    assert output_mat.asset_key == AssetKey('asset_one')

def test_asset_observation_in_sensor() -> None:
    if False:
        print('Hello World!')

    @sensor()
    def a_sensor() -> SensorResult:
        if False:
            i = 10
            return i + 15
        return SensorResult(asset_events=[AssetObservation('asset_one')])
    instance = DagsterInstance.ephemeral()
    sensor_execution_data = a_sensor.evaluate_tick(build_sensor_context(instance=instance))
    assert len(sensor_execution_data.asset_events) == 1
    output_mat = sensor_execution_data.asset_events[0]
    assert isinstance(output_mat, AssetObservation)
    assert output_mat.asset_key == AssetKey('asset_one')

def test_asset_check_evaluation() -> None:
    if False:
        i = 10
        return i + 15

    @sensor()
    def a_sensor() -> SensorResult:
        if False:
            for i in range(10):
                print('nop')
        return SensorResult(asset_events=[AssetCheckEvaluation(asset_key=AssetKey('asset_one'), check_name='check_one', passed=True, metadata={})])
    instance = DagsterInstance.ephemeral()
    sensor_execution_data = a_sensor.evaluate_tick(build_sensor_context(instance=instance))
    assert len(sensor_execution_data.asset_events) == 1
    output_ace = sensor_execution_data.asset_events[0]
    assert isinstance(output_ace, AssetCheckEvaluation)
    assert output_ace.asset_key == AssetKey('asset_one')

def test_asset_materialization_in_sensor_direct_invocation() -> None:
    if False:
        for i in range(10):
            print('nop')

    @sensor()
    def a_sensor() -> SensorResult:
        if False:
            while True:
                i = 10
        return SensorResult(asset_events=[AssetMaterialization('asset_one')])
    instance = DagsterInstance.ephemeral()
    a_sensor(build_sensor_context(instance=instance))