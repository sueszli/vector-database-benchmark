import os
import sys
from contextlib import contextmanager
from typing import Iterator, Optional
import pendulum
import pytest
from dagster import AssetKey, SensorEvaluationContext, job, multi_asset_sensor, op, resource, sensor
from dagster._check import ParameterCheckError
from dagster._config.pythonic_config import ConfigurableResource
from dagster._core.definitions.asset_selection import AssetSelection
from dagster._core.definitions.decorators.sensor_decorator import asset_sensor
from dagster._core.definitions.definitions_class import Definitions
from dagster._core.definitions.events import AssetMaterialization
from dagster._core.definitions.freshness_policy_sensor_definition import FreshnessPolicySensorContext, freshness_policy_sensor
from dagster._core.definitions.multi_asset_sensor_definition import MultiAssetSensorEvaluationContext
from dagster._core.definitions.repository_definition.valid_definitions import SINGLETON_REPOSITORY_NAME
from dagster._core.definitions.resource_annotation import ResourceParam
from dagster._core.definitions.run_request import InstigatorType
from dagster._core.definitions.run_status_sensor_definition import RunFailureSensorContext, RunStatusSensorContext, run_failure_sensor, run_status_sensor
from dagster._core.definitions.sensor_definition import RunRequest
from dagster._core.events.log import EventLogEntry
from dagster._core.execution.context.compute import OpExecutionContext
from dagster._core.instance import DagsterInstance
from dagster._core.scheduler.instigation import InstigatorState, InstigatorStatus, TickStatus
from dagster._core.storage.dagster_run import DagsterRunStatus
from dagster._core.test_utils import create_test_daemon_workspace_context
from dagster._core.types.loadable_target_origin import LoadableTargetOrigin
from dagster._core.workspace.context import WorkspaceProcessContext
from dagster._core.workspace.load_target import ModuleTarget
from dagster._seven.compat.pendulum import create_pendulum_time, to_timezone
from .test_sensor_run import evaluate_sensors, validate_tick, wait_for_all_runs_to_start

@op(out={})
def the_op(context: OpExecutionContext):
    if False:
        i = 10
        return i + 15
    yield AssetMaterialization(asset_key=AssetKey('my_asset'), description='my_asset')

@job
def the_job() -> None:
    if False:
        return 10
    the_op()

@op(out={})
def the_failure_op(context: OpExecutionContext):
    if False:
        print('Hello World!')
    raise Exception()

@job
def the_failure_job() -> None:
    if False:
        return 10
    the_failure_op()

class MyResource(ConfigurableResource):
    a_str: str

@sensor(job_name='the_job', required_resource_keys={'my_resource'})
def sensor_from_context(context: SensorEvaluationContext):
    if False:
        i = 10
        return i + 15
    return RunRequest(context.resources.my_resource.a_str, run_config={}, tags={})

@sensor(job_name='the_job')
def sensor_from_fn_arg(context: SensorEvaluationContext, my_resource: MyResource):
    if False:
        i = 10
        return i + 15
    return RunRequest(my_resource.a_str, run_config={}, tags={})

@op(out={})
def the_op_but_with_a_resource_dep(my_resource: MyResource):
    if False:
        for i in range(10):
            print('nop')
    assert my_resource.a_str == 'foo'

@job
def the_job_but_with_a_resource_dep() -> None:
    if False:
        for i in range(10):
            print('nop')
    the_op_but_with_a_resource_dep()

@sensor(job_name='the_job_but_with_a_resource_dep')
def sensor_with_job_with_resource_dep(context: SensorEvaluationContext, my_resource: MyResource):
    if False:
        for i in range(10):
            print('nop')
    return RunRequest(my_resource.a_str, run_config={}, tags={})
is_in_cm = False

@resource
@contextmanager
def my_cm_resource(_) -> Iterator[str]:
    if False:
        return 10
    global is_in_cm
    is_in_cm = True
    yield 'foo'
    is_in_cm = False

@sensor(job_name='the_job')
def sensor_with_cm(context: SensorEvaluationContext, my_cm_resource: ResourceParam[str]):
    if False:
        return 10
    assert is_in_cm
    return RunRequest(my_cm_resource, run_config={}, tags={})

@sensor(job_name='the_job', required_resource_keys={'my_resource'})
def sensor_from_context_weird_name(not_called_context: SensorEvaluationContext):
    if False:
        while True:
            i = 10
    return RunRequest(not_called_context.resources.my_resource.a_str, run_config={}, tags={})

@sensor(job_name='the_job')
def sensor_from_fn_arg_no_context(my_resource: MyResource):
    if False:
        while True:
            i = 10
    return RunRequest(my_resource.a_str, run_config={}, tags={})

@sensor(job_name='the_job')
def sensor_context_arg_not_first_and_weird_name(my_resource: MyResource, not_called_context: SensorEvaluationContext):
    if False:
        for i in range(10):
            print('nop')
    assert not_called_context.resources.my_resource.a_str == my_resource.a_str
    return RunRequest(not_called_context.resources.my_resource.a_str, run_config={}, tags={})

@resource
def the_inner() -> str:
    if False:
        for i in range(10):
            print('nop')
    return 'oo'

@resource(required_resource_keys={'the_inner'})
def the_outer(init_context) -> str:
    if False:
        return 10
    return 'f' + init_context.resources.the_inner

@sensor(job=the_job, required_resource_keys={'the_outer'})
def sensor_resource_deps(context):
    if False:
        print('Hello World!')
    return RunRequest(context.resources.the_outer, run_config={}, tags={})

@asset_sensor(asset_key=AssetKey('my_asset'), job_name='the_job')
def sensor_asset(my_resource: MyResource, not_called_context: SensorEvaluationContext):
    if False:
        print('Hello World!')
    assert not_called_context.resources.my_resource.a_str == my_resource.a_str
    return RunRequest(my_resource.a_str, run_config={}, tags={})

@asset_sensor(asset_key=AssetKey('my_asset'), job_name='the_job')
def sensor_asset_with_cm(my_cm_resource: ResourceParam[str], not_called_context: SensorEvaluationContext):
    if False:
        i = 10
        return i + 15
    assert not_called_context.resources.my_cm_resource == my_cm_resource
    assert is_in_cm
    return RunRequest(my_cm_resource, run_config={}, tags={})

@asset_sensor(asset_key=AssetKey('my_asset'), job_name='the_job')
def sensor_asset_with_event(my_resource: MyResource, not_called_context: SensorEvaluationContext, my_asset_event: EventLogEntry):
    if False:
        for i in range(10):
            print('nop')
    assert not_called_context.resources.my_resource.a_str == my_resource.a_str
    assert my_asset_event.dagster_event
    assert my_asset_event.dagster_event.asset_key == AssetKey('my_asset')
    return RunRequest(my_resource.a_str, run_config={}, tags={})

@asset_sensor(asset_key=AssetKey('my_asset'), job_name='the_job')
def sensor_asset_no_context(my_resource: MyResource):
    if False:
        print('Hello World!')
    return RunRequest(my_resource.a_str, run_config={}, tags={})

@multi_asset_sensor(monitored_assets=[AssetKey('my_asset')], job_name='the_job')
def sensor_multi_asset(my_resource: MyResource, not_called_context: MultiAssetSensorEvaluationContext) -> RunRequest:
    if False:
        while True:
            i = 10
    assert not_called_context.resources.my_resource.a_str == my_resource.a_str
    asset_events = list(not_called_context.materialization_records_for_key(asset_key=AssetKey('my_asset'), limit=1))
    if asset_events:
        not_called_context.advance_all_cursors()
    return RunRequest(my_resource.a_str, run_config={}, tags={})

@multi_asset_sensor(monitored_assets=[AssetKey('my_asset')], job_name='the_job')
def sensor_multi_asset_with_cm(my_cm_resource: ResourceParam[str], not_called_context: MultiAssetSensorEvaluationContext) -> RunRequest:
    if False:
        i = 10
        return i + 15
    assert not_called_context.resources.my_cm_resource == my_cm_resource
    assert is_in_cm
    asset_events = list(not_called_context.materialization_records_for_key(asset_key=AssetKey('my_asset'), limit=1))
    if asset_events:
        not_called_context.advance_all_cursors()
    return RunRequest(my_cm_resource, run_config={}, tags={})

@freshness_policy_sensor(asset_selection=AssetSelection.all())
def sensor_freshness_policy(my_resource: MyResource, not_called_context: FreshnessPolicySensorContext):
    if False:
        for i in range(10):
            print('nop')
    assert not_called_context.resources.my_resource.a_str == my_resource.a_str
    return RunRequest(my_resource.a_str, run_config={}, tags={})

@freshness_policy_sensor(asset_selection=AssetSelection.all())
def sensor_freshness_policy_with_cm(my_cm_resource: ResourceParam[str], not_called_context: FreshnessPolicySensorContext):
    if False:
        i = 10
        return i + 15
    assert is_in_cm
    assert not_called_context.resources.my_cm_resource == my_cm_resource
    return RunRequest(my_cm_resource, run_config={}, tags={})

@run_status_sensor(monitor_all_repositories=True, run_status=DagsterRunStatus.SUCCESS, request_job=the_job)
def sensor_run_status(my_resource: MyResource, not_called_context: RunStatusSensorContext):
    if False:
        print('Hello World!')
    assert not_called_context.resources.my_resource.a_str == my_resource.a_str
    return RunRequest(my_resource.a_str, run_config={}, tags={})

@run_status_sensor(monitor_all_repositories=True, run_status=DagsterRunStatus.SUCCESS, request_job=the_job)
def sensor_run_status_with_cm(my_cm_resource: ResourceParam[str], not_called_context: RunStatusSensorContext):
    if False:
        i = 10
        return i + 15
    assert not_called_context.resources.my_cm_resource == my_cm_resource
    assert is_in_cm
    return RunRequest(my_cm_resource, run_config={}, tags={})

@run_failure_sensor(monitor_all_repositories=True, request_job=the_job)
def sensor_run_failure(my_resource: MyResource, not_called_context: RunFailureSensorContext):
    if False:
        print('Hello World!')
    assert not_called_context.failure_event
    assert not_called_context.resources.my_resource.a_str == my_resource.a_str
    return RunRequest(my_resource.a_str, run_config={}, tags={})

@run_failure_sensor(monitor_all_repositories=True, request_job=the_job)
def sensor_run_failure_with_cm(my_cm_resource: ResourceParam[str], not_called_context: RunFailureSensorContext):
    if False:
        while True:
            i = 10
    assert not_called_context.failure_event
    assert not_called_context.resources.my_cm_resource == my_cm_resource
    assert is_in_cm
    return RunRequest(my_cm_resource, run_config={}, tags={})
the_repo = Definitions(jobs=[the_job, the_failure_job, the_job_but_with_a_resource_dep], sensors=[sensor_from_context, sensor_from_fn_arg, sensor_with_job_with_resource_dep, sensor_with_cm, sensor_from_context_weird_name, sensor_from_fn_arg_no_context, sensor_context_arg_not_first_and_weird_name, sensor_resource_deps, sensor_asset, sensor_asset_with_cm, sensor_asset_with_event, sensor_asset_no_context, sensor_multi_asset, sensor_multi_asset_with_cm, sensor_freshness_policy, sensor_freshness_policy_with_cm, sensor_run_status, sensor_run_status_with_cm, sensor_run_failure, sensor_run_failure_with_cm], resources={'my_resource': MyResource(a_str='foo'), 'my_cm_resource': my_cm_resource, 'the_inner': the_inner, 'the_outer': the_outer})

def create_workspace_load_target(attribute: Optional[str]=SINGLETON_REPOSITORY_NAME):
    if False:
        print('Hello World!')
    return ModuleTarget(module_name='dagster_tests.daemon_sensor_tests.test_pythonic_resources', attribute=None, working_directory=os.path.dirname(__file__), location_name='test_location')

@pytest.fixture(name='workspace_context_struct_resources', scope='module')
def workspace_fixture(instance_module_scoped):
    if False:
        print('Hello World!')
    with create_test_daemon_workspace_context(workspace_load_target=create_workspace_load_target(), instance=instance_module_scoped) as workspace:
        yield workspace

@pytest.fixture(name='external_repo_struct_resources', scope='module')
def external_repo_fixture(workspace_context_struct_resources: WorkspaceProcessContext):
    if False:
        for i in range(10):
            print('nop')
    repo_loc = next(iter(workspace_context_struct_resources.create_request_context().get_workspace_snapshot().values())).code_location
    assert repo_loc
    return repo_loc.get_repository(SINGLETON_REPOSITORY_NAME)

def loadable_target_origin() -> LoadableTargetOrigin:
    if False:
        print('Hello World!')
    return LoadableTargetOrigin(executable_path=sys.executable, module_name='dagster_tests.daemon_sensor_tests.test_pythonic_resources', working_directory=os.getcwd(), attribute=None)

def test_cant_use_required_resource_keys_and_params_both() -> None:
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ParameterCheckError):

        @sensor(job_name='the_job', required_resource_keys={'my_other_resource'})
        def sensor_from_context_and_params(context: SensorEvaluationContext, my_resource: MyResource):
            if False:
                i = 10
                return i + 15
            return RunRequest(my_resource.a_str, run_config={}, tags={})

@pytest.mark.parametrize('sensor_name', ['sensor_from_context', 'sensor_from_fn_arg', 'sensor_with_job_with_resource_dep', 'sensor_with_cm', 'sensor_from_context_weird_name', 'sensor_from_fn_arg_no_context', 'sensor_context_arg_not_first_and_weird_name', 'sensor_resource_deps', 'sensor_asset', 'sensor_asset_with_cm', 'sensor_asset_with_event', 'sensor_asset_no_context', 'sensor_multi_asset', 'sensor_multi_asset_with_cm'])
def test_resources(caplog, instance: DagsterInstance, workspace_context_struct_resources, external_repo_struct_resources, sensor_name) -> None:
    if False:
        print('Hello World!')
    assert not is_in_cm
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, hour=23, minute=59, second=59, tz='UTC'), 'US/Central')
    with pendulum.test(freeze_datetime):
        base_run_count = 0
        if 'asset' in sensor_name:
            the_job.execute_in_process(instance=instance)
            base_run_count = 1
        external_sensor = external_repo_struct_resources.get_external_sensor(sensor_name)
        instance.add_instigator_state(InstigatorState(external_sensor.get_external_origin(), InstigatorType.SENSOR, InstigatorStatus.RUNNING))
        assert instance.get_runs_count() == base_run_count
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 0
        evaluate_sensors(workspace_context_struct_resources, None)
        wait_for_all_runs_to_start(instance)
        assert instance.get_runs_count() == base_run_count + 1
        run = instance.get_runs()[0]
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 1
        assert ticks[0].run_keys == ['foo']
        validate_tick(ticks[0], external_sensor, freeze_datetime, TickStatus.SUCCESS, expected_run_ids=[run.run_id])
    assert not is_in_cm

@pytest.mark.parametrize('sensor_name', ['sensor_freshness_policy', 'sensor_freshness_policy_with_cm'])
def test_resources_freshness_policy_sensor(caplog, instance, workspace_context_struct_resources, external_repo_struct_resources, sensor_name) -> None:
    if False:
        for i in range(10):
            print('nop')
    assert not is_in_cm
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, hour=23, minute=59, second=59, tz='UTC'), 'US/Central')
    original_time = freeze_datetime
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo_struct_resources.get_external_sensor(sensor_name)
        instance.add_instigator_state(InstigatorState(external_sensor.get_external_origin(), InstigatorType.SENSOR, InstigatorStatus.RUNNING))
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 0
    with pendulum.test(freeze_datetime):
        evaluate_sensors(workspace_context_struct_resources, None)
        wait_for_all_runs_to_start(instance)
    freeze_datetime = freeze_datetime.add(seconds=60)
    with pendulum.test(freeze_datetime):
        evaluate_sensors(workspace_context_struct_resources, None)
        wait_for_all_runs_to_start(instance)
    with pendulum.test(freeze_datetime):
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 2
        validate_tick(ticks[0], external_sensor, freeze_datetime, TickStatus.SKIPPED, expected_run_ids=[])
        validate_tick(ticks[1], external_sensor, original_time, TickStatus.SKIPPED, expected_run_ids=[])
    assert not is_in_cm

@pytest.mark.parametrize('sensor_name', ['sensor_run_status', 'sensor_run_status_with_cm'])
def test_resources_run_status_sensor(caplog, instance: DagsterInstance, workspace_context_struct_resources, external_repo_struct_resources, sensor_name) -> None:
    if False:
        while True:
            i = 10
    assert not is_in_cm
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, hour=23, minute=59, second=59, tz='UTC'), 'US/Central')
    original_time = freeze_datetime
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo_struct_resources.get_external_sensor(sensor_name)
        instance.add_instigator_state(InstigatorState(external_sensor.get_external_origin(), InstigatorType.SENSOR, InstigatorStatus.RUNNING))
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 0
    with pendulum.test(freeze_datetime):
        evaluate_sensors(workspace_context_struct_resources, None)
        wait_for_all_runs_to_start(instance)
    the_job.execute_in_process(instance=instance)
    freeze_datetime = freeze_datetime.add(seconds=60)
    with pendulum.test(freeze_datetime):
        evaluate_sensors(workspace_context_struct_resources, None)
        wait_for_all_runs_to_start(instance)
    with pendulum.test(freeze_datetime):
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 2
        assert instance.get_runs_count() == 2
        run = instance.get_runs()[0]
        assert ticks[0].run_keys == ['foo']
        validate_tick(ticks[0], external_sensor, freeze_datetime, TickStatus.SUCCESS, expected_run_ids=[run.run_id])
        validate_tick(ticks[1], external_sensor, original_time, TickStatus.SKIPPED, expected_run_ids=[])
    assert not is_in_cm

@pytest.mark.parametrize('sensor_name', ['sensor_run_failure', 'sensor_run_failure_with_cm'])
def test_resources_run_failure_sensor(caplog, instance: DagsterInstance, workspace_context_struct_resources, external_repo_struct_resources, sensor_name) -> None:
    if False:
        while True:
            i = 10
    assert not is_in_cm
    freeze_datetime = to_timezone(create_pendulum_time(year=2019, month=2, day=27, hour=23, minute=59, second=59, tz='UTC'), 'US/Central')
    original_time = freeze_datetime
    with pendulum.test(freeze_datetime):
        external_sensor = external_repo_struct_resources.get_external_sensor(sensor_name)
        instance.add_instigator_state(InstigatorState(external_sensor.get_external_origin(), InstigatorType.SENSOR, InstigatorStatus.RUNNING))
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 0
    with pendulum.test(freeze_datetime):
        evaluate_sensors(workspace_context_struct_resources, None)
        wait_for_all_runs_to_start(instance)
    the_failure_job.execute_in_process(instance=instance, raise_on_error=False)
    freeze_datetime = freeze_datetime.add(seconds=60)
    with pendulum.test(freeze_datetime):
        evaluate_sensors(workspace_context_struct_resources, None)
        wait_for_all_runs_to_start(instance)
    with pendulum.test(freeze_datetime):
        ticks = instance.get_ticks(external_sensor.get_external_origin_id(), external_sensor.selector_id)
        assert len(ticks) == 2
        assert instance.get_runs_count() == 2
        run = instance.get_runs()[0]
        assert ticks[0].run_keys == ['foo']
        validate_tick(ticks[0], external_sensor, freeze_datetime, TickStatus.SUCCESS, expected_run_ids=[run.run_id])
        validate_tick(ticks[1], external_sensor, original_time, TickStatus.SKIPPED, expected_run_ids=[])
    assert not is_in_cm