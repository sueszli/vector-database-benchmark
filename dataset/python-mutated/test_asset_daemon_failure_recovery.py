import multiprocessing
from typing import TYPE_CHECKING
import pendulum
import pytest
from dagster import AssetKey
from dagster._core.errors import DagsterUserCodeUnreachableError
from dagster._core.instance import DagsterInstance
from dagster._core.instance.ref import InstanceRef
from dagster._core.instance_for_test import instance_for_test
from dagster._core.scheduler.instigation import TickStatus
from dagster._core.storage.tags import PARTITION_NAME_TAG
from dagster._core.test_utils import cleanup_test_instance
from dagster._daemon.asset_daemon import FIXED_AUTO_MATERIALIZATION_ORIGIN_ID, FIXED_AUTO_MATERIALIZATION_SELECTOR_ID, MAX_TIME_TO_RESUME_TICK_SECONDS, _get_raw_cursor, set_auto_materialize_paused
from dagster._utils import SingleInstigatorDebugCrashFlags, get_terminate_signal
from .scenarios.auto_materialize_policy_scenarios import auto_materialize_policy_scenarios
from .scenarios.auto_observe_scenarios import auto_observe_scenarios
from .scenarios.multi_code_location_scenarios import multi_code_location_scenarios
if TYPE_CHECKING:
    from pendulum.datetime import DateTime
daemon_scenarios = {**auto_materialize_policy_scenarios, **multi_code_location_scenarios, **auto_observe_scenarios}

def _assert_run_requests_match(expected_run_requests, run_requests):
    if False:
        print('Hello World!')

    def sort_run_request_key_fn(run_request):
        if False:
            while True:
                i = 10
        return (min(run_request.asset_selection), run_request.partition_key)
    sorted_run_requests = sorted(run_requests, key=sort_run_request_key_fn)
    sorted_expected_run_requests = sorted(expected_run_requests, key=sort_run_request_key_fn)
    for (run_request, expected_run_request) in zip(sorted_run_requests, sorted_expected_run_requests):
        assert set(run_request.asset_selection) == set(expected_run_request.asset_selection)
        assert run_request.partition_key == expected_run_request.partition_key

@pytest.fixture
def instance():
    if False:
        for i in range(10):
            print('nop')
    with instance_for_test() as the_instance:
        yield the_instance

@pytest.fixture
def daemon_paused_instance():
    if False:
        for i in range(10):
            print('nop')
    with instance_for_test(overrides={'run_launcher': {'module': 'dagster._core.launcher.sync_in_memory_run_launcher', 'class': 'SyncInMemoryRunLauncher'}, 'auto_materialize': {'max_tick_retries': 2}}) as the_instance:
        yield the_instance

@pytest.fixture
def daemon_not_paused_instance(daemon_paused_instance):
    if False:
        while True:
            i = 10
    set_auto_materialize_paused(daemon_paused_instance, False)
    return daemon_paused_instance

def test_old_tick_not_resumed(daemon_not_paused_instance):
    if False:
        return 10
    instance = daemon_not_paused_instance
    error_asset_scenario = daemon_scenarios['auto_materialize_policy_max_materializations_not_exceeded']
    execution_time = error_asset_scenario.current_time
    error_asset_scenario = error_asset_scenario._replace(current_time=None)
    debug_crash_flags = {'RUN_CREATED': Exception('OOPS')}
    with pendulum.test(execution_time):
        with pytest.raises(Exception, match='OOPS'):
            error_asset_scenario.do_daemon_scenario(instance, scenario_name='auto_materialize_policy_max_materializations_not_exceeded', debug_crash_flags=debug_crash_flags)
        ticks = instance.get_ticks(origin_id=FIXED_AUTO_MATERIALIZATION_ORIGIN_ID, selector_id=FIXED_AUTO_MATERIALIZATION_SELECTOR_ID)
        assert len(ticks) == 1
        assert ticks[0].tick_data.auto_materialize_evaluation_id == 1
        assert ticks[0].timestamp == execution_time.timestamp()
    execution_time = execution_time.add(seconds=MAX_TIME_TO_RESUME_TICK_SECONDS + 1)
    with pendulum.test(execution_time):
        with pytest.raises(Exception, match='OOPS'):
            error_asset_scenario.do_daemon_scenario(instance, scenario_name='auto_materialize_policy_max_materializations_not_exceeded', debug_crash_flags=debug_crash_flags)
        ticks = instance.get_ticks(origin_id=FIXED_AUTO_MATERIALIZATION_ORIGIN_ID, selector_id=FIXED_AUTO_MATERIALIZATION_SELECTOR_ID)
        assert len(ticks) == 2
        assert ticks[0].tick_data.auto_materialize_evaluation_id == 2
    execution_time = execution_time.add(seconds=MAX_TIME_TO_RESUME_TICK_SECONDS - 1)
    with pendulum.test(execution_time):
        with pytest.raises(Exception, match='OOPS'):
            error_asset_scenario.do_daemon_scenario(instance, scenario_name='auto_materialize_policy_max_materializations_not_exceeded', debug_crash_flags=debug_crash_flags)
        ticks = instance.get_ticks(origin_id=FIXED_AUTO_MATERIALIZATION_ORIGIN_ID, selector_id=FIXED_AUTO_MATERIALIZATION_SELECTOR_ID)
        assert len(ticks) == 3
        assert ticks[0].tick_data.auto_materialize_evaluation_id == 2

@pytest.mark.parametrize('crash_location', ['EVALUATIONS_FINISHED', 'RUN_REQUESTS_CREATED'])
def test_error_loop_before_cursor_written(daemon_not_paused_instance, crash_location):
    if False:
        for i in range(10):
            print('nop')
    instance = daemon_not_paused_instance
    error_asset_scenario = daemon_scenarios['auto_materialize_policy_max_materializations_not_exceeded']
    execution_time = error_asset_scenario.current_time
    error_asset_scenario = error_asset_scenario._replace(current_time=None)
    for trial_num in range(3):
        test_time = execution_time.add(seconds=15 * trial_num)
        with pendulum.test(test_time):
            debug_crash_flags = {crash_location: Exception(f'Oops {trial_num}')}
            with pytest.raises(Exception, match=f'Oops {trial_num}'):
                error_asset_scenario.do_daemon_scenario(instance, scenario_name='auto_materialize_policy_max_materializations_not_exceeded', debug_crash_flags=debug_crash_flags)
            ticks = instance.get_ticks(origin_id=FIXED_AUTO_MATERIALIZATION_ORIGIN_ID, selector_id=FIXED_AUTO_MATERIALIZATION_SELECTOR_ID)
            assert len(ticks) == trial_num + 1
            assert ticks[0].status == TickStatus.FAILURE
            assert ticks[0].timestamp == test_time.timestamp()
            assert ticks[0].tick_data.end_timestamp == test_time.timestamp()
            assert ticks[0].tick_data.auto_materialize_evaluation_id == 1
            assert ticks[0].tick_data.failure_count == 1
            assert f'Oops {trial_num}' in str(ticks[0].tick_data.error)
            _assert_run_requests_match(error_asset_scenario.expected_run_requests, ticks[0].tick_data.run_requests)
            cursor = _get_raw_cursor(instance)
            assert not cursor
    test_time = test_time.add(seconds=45)
    with pendulum.test(test_time):
        error_asset_scenario.do_daemon_scenario(instance, scenario_name='auto_materialize_policy_max_materializations_not_exceeded', debug_crash_flags={})
    ticks = instance.get_ticks(origin_id=FIXED_AUTO_MATERIALIZATION_ORIGIN_ID, selector_id=FIXED_AUTO_MATERIALIZATION_SELECTOR_ID)
    assert len(ticks) == 4
    assert ticks[0].status == TickStatus.SUCCESS
    assert ticks[0].timestamp == test_time.timestamp()
    assert ticks[0].tick_data.end_timestamp == test_time.timestamp()
    assert ticks[0].tick_data.auto_materialize_evaluation_id == 1
    runs = instance.get_runs()
    assert len(runs) == 5

@pytest.mark.parametrize('crash_location', ['RUN_CREATED', 'RUN_SUBMITTED', 'EXECUTION_PLAN_CREATED_1', 'RUN_CREATED_1', 'RUN_SUBMITTED_1', 'RUN_IDS_ADDED_TO_EVALUATIONS'])
def test_error_loop_after_cursor_written(daemon_not_paused_instance, crash_location):
    if False:
        i = 10
        return i + 15
    instance = daemon_not_paused_instance
    error_asset_scenario = daemon_scenarios['auto_materialize_policy_max_materializations_not_exceeded']
    execution_time = error_asset_scenario.current_time
    error_asset_scenario = error_asset_scenario._replace(current_time=None)
    last_cursor = None
    test_time = execution_time.add(seconds=15)
    with pendulum.test(test_time):
        debug_crash_flags = {crash_location: DagsterUserCodeUnreachableError('WHERE IS THE CODE')}
        with pytest.raises(Exception, match='WHERE IS THE CODE'):
            error_asset_scenario.do_daemon_scenario(instance, scenario_name='auto_materialize_policy_max_materializations_not_exceeded', debug_crash_flags=debug_crash_flags)
        ticks = instance.get_ticks(origin_id=FIXED_AUTO_MATERIALIZATION_ORIGIN_ID, selector_id=FIXED_AUTO_MATERIALIZATION_SELECTOR_ID)
        assert len(ticks) == 1
        assert ticks[0].status == TickStatus.FAILURE
        assert ticks[0].timestamp == test_time.timestamp()
        assert ticks[0].tick_data.end_timestamp == test_time.timestamp()
        assert ticks[0].tick_data.auto_materialize_evaluation_id == 1
        assert ticks[0].tick_data.failure_count == 0
        assert 'WHERE IS THE CODE' in str(ticks[0].tick_data.error)
        assert 'Auto-materialization will resume once the code server is available' in str(ticks[0].tick_data.error)
        _assert_run_requests_match(error_asset_scenario.expected_run_requests, ticks[0].tick_data.run_requests)
        cursor = _get_raw_cursor(instance)
        assert cursor is not None
        last_cursor = cursor
    for trial_num in range(3):
        test_time = test_time.add(seconds=15)
        with pendulum.test(test_time):
            debug_crash_flags = {crash_location: Exception(f'Oops {trial_num}')}
            with pytest.raises(Exception, match=f'Oops {trial_num}'):
                error_asset_scenario.do_daemon_scenario(instance, scenario_name='auto_materialize_policy_max_materializations_not_exceeded', debug_crash_flags=debug_crash_flags)
            ticks = instance.get_ticks(origin_id=FIXED_AUTO_MATERIALIZATION_ORIGIN_ID, selector_id=FIXED_AUTO_MATERIALIZATION_SELECTOR_ID)
            assert len(ticks) == trial_num + 2
            assert ticks[0].status == TickStatus.FAILURE
            assert ticks[0].timestamp == test_time.timestamp()
            assert ticks[0].tick_data.end_timestamp == test_time.timestamp()
            assert ticks[0].tick_data.auto_materialize_evaluation_id == 1
            assert ticks[0].tick_data.failure_count == trial_num + 1
            assert f'Oops {trial_num}' in str(ticks[0].tick_data.error)
            _assert_run_requests_match(error_asset_scenario.expected_run_requests, ticks[0].tick_data.run_requests)
            retry_cursor = _get_raw_cursor(instance)
            assert retry_cursor == last_cursor
    test_time = test_time.add(seconds=45)
    with pendulum.test(test_time):
        debug_crash_flags = {'RUN_IDS_ADDED_TO_EVALUATIONS': Exception('Oops new tick')}
        with pytest.raises(Exception, match='Oops new tick'):
            error_asset_scenario.do_daemon_scenario(instance, scenario_name='auto_materialize_policy_max_materializations_not_exceeded', debug_crash_flags=debug_crash_flags)
        ticks = instance.get_ticks(origin_id=FIXED_AUTO_MATERIALIZATION_ORIGIN_ID, selector_id=FIXED_AUTO_MATERIALIZATION_SELECTOR_ID)
        assert len(ticks) == 5
        assert ticks[0].status == TickStatus.FAILURE
        assert ticks[0].timestamp == test_time.timestamp()
        assert ticks[0].tick_data.end_timestamp == test_time.timestamp()
        assert ticks[0].tick_data.auto_materialize_evaluation_id == 2
        assert 'Oops new tick' in str(ticks[0].tick_data.error)
        assert ticks[0].tick_data.failure_count == 1
        moved_on_cursor = _get_raw_cursor(instance)
        assert moved_on_cursor != last_cursor
    test_time = test_time.add(seconds=45)
    with pendulum.test(test_time):
        error_asset_scenario.do_daemon_scenario(instance, scenario_name='auto_materialize_policy_max_materializations_not_exceeded', debug_crash_flags={})
    ticks = instance.get_ticks(origin_id=FIXED_AUTO_MATERIALIZATION_ORIGIN_ID, selector_id=FIXED_AUTO_MATERIALIZATION_SELECTOR_ID)
    assert len(ticks) == 6
    assert ticks[0].status != TickStatus.FAILURE
    assert ticks[0].timestamp == test_time.timestamp()
    assert ticks[0].tick_data.end_timestamp == test_time.timestamp()
    assert ticks[0].tick_data.auto_materialize_evaluation_id == 2
spawn_ctx = multiprocessing.get_context('spawn')

def _test_asset_daemon_in_subprocess(scenario_name, instance_ref: InstanceRef, execution_datetime: 'DateTime', debug_crash_flags: SingleInstigatorDebugCrashFlags) -> None:
    if False:
        while True:
            i = 10
    scenario = daemon_scenarios[scenario_name]
    with DagsterInstance.from_ref(instance_ref) as instance:
        try:
            scenario._replace(current_time=execution_datetime).do_daemon_scenario(instance, scenario_name=scenario_name, debug_crash_flags=debug_crash_flags)
        finally:
            cleanup_test_instance(instance)

@pytest.mark.parametrize('crash_location', ['EVALUATIONS_FINISHED', 'ASSET_EVALUATIONS_ADDED', 'RUN_REQUESTS_CREATED', 'CURSOR_UPDATED', 'RUN_IDS_ADDED_TO_EVALUATIONS', 'EXECUTION_PLAN_CREATED_1', 'RUN_CREATED', 'RUN_SUBMITTED', 'RUN_CREATED_1', 'RUN_SUBMITTED_1'])
def test_asset_daemon_crash_recovery(daemon_not_paused_instance, crash_location):
    if False:
        while True:
            i = 10
    instance = daemon_not_paused_instance
    scenario = daemon_scenarios['auto_materialize_policy_max_materializations_not_exceeded']
    asset_daemon_process = spawn_ctx.Process(target=_test_asset_daemon_in_subprocess, args=['auto_materialize_policy_max_materializations_not_exceeded', instance.get_ref(), scenario.current_time, {crash_location: get_terminate_signal()}])
    asset_daemon_process.start()
    asset_daemon_process.join(timeout=60)
    ticks = instance.get_ticks(origin_id=FIXED_AUTO_MATERIALIZATION_ORIGIN_ID, selector_id=FIXED_AUTO_MATERIALIZATION_SELECTOR_ID)
    assert len(ticks) == 1
    assert ticks[0]
    assert ticks[0].status == TickStatus.STARTED
    assert ticks[0].timestamp == scenario.current_time.timestamp()
    assert not ticks[0].tick_data.end_timestamp == scenario.current_time.timestamp()
    assert not len(ticks[0].tick_data.run_ids)
    assert ticks[0].tick_data.auto_materialize_evaluation_id == 1
    freeze_datetime = scenario.current_time.add(seconds=1)
    asset_daemon_process = spawn_ctx.Process(target=_test_asset_daemon_in_subprocess, args=['auto_materialize_policy_max_materializations_not_exceeded', instance.get_ref(), freeze_datetime, None])
    asset_daemon_process.start()
    asset_daemon_process.join(timeout=60)
    ticks = instance.get_ticks(origin_id=FIXED_AUTO_MATERIALIZATION_ORIGIN_ID, selector_id=FIXED_AUTO_MATERIALIZATION_SELECTOR_ID)
    cursor_written = crash_location not in ('EVALUATIONS_FINISHED', 'ASSET_EVALUATIONS_ADDED', 'RUN_REQUESTS_CREATED')
    assert len(ticks) == 1 if cursor_written else 2
    assert ticks[0]
    assert ticks[0].status == TickStatus.SUCCESS
    assert ticks[0].timestamp == scenario.current_time.timestamp() if cursor_written else freeze_datetime.timestamp()
    assert ticks[0].tick_data.end_timestamp == freeze_datetime.timestamp()
    assert len(ticks[0].tick_data.run_ids) == 5
    assert ticks[0].tick_data.auto_materialize_evaluation_id == 1
    if len(ticks) == 2:
        assert ticks[1].status == TickStatus.SKIPPED
    _assert_run_requests_match(scenario.expected_run_requests, ticks[0].tick_data.run_requests)
    runs = instance.get_runs()
    assert len(runs) == 5

    def sort_run_key_fn(run):
        if False:
            print('Hello World!')
        return (min(run.asset_selection), run.tags.get(PARTITION_NAME_TAG))
    sorted_runs = sorted(runs[:len(scenario.expected_run_requests)], key=sort_run_key_fn)
    evaluations = instance.schedule_storage.get_auto_materialize_asset_evaluations(asset_key=AssetKey('hourly'), limit=100)
    assert len(evaluations) == 1
    assert evaluations[0].evaluation.asset_key == AssetKey('hourly')
    assert evaluations[0].evaluation.run_ids == {run.run_id for run in sorted_runs}

@pytest.mark.parametrize('crash_location', ['EVALUATIONS_FINISHED', 'ASSET_EVALUATIONS_ADDED', 'RUN_REQUESTS_CREATED', 'RUN_IDS_ADDED_TO_EVALUATIONS', 'RUN_CREATED', 'RUN_SUBMITTED', 'RUN_CREATED_2', 'RUN_SUBMITTED_2'])
def test_asset_daemon_exception_recovery(daemon_not_paused_instance, crash_location):
    if False:
        return 10
    instance = daemon_not_paused_instance
    scenario = daemon_scenarios['auto_materialize_policy_max_materializations_not_exceeded']
    asset_daemon_process = spawn_ctx.Process(target=_test_asset_daemon_in_subprocess, args=['auto_materialize_policy_max_materializations_not_exceeded', instance.get_ref(), scenario.current_time, {crash_location: Exception('OOPS')}])
    asset_daemon_process.start()
    asset_daemon_process.join(timeout=60)
    ticks = instance.get_ticks(origin_id=FIXED_AUTO_MATERIALIZATION_ORIGIN_ID, selector_id=FIXED_AUTO_MATERIALIZATION_SELECTOR_ID)
    assert len(ticks) == 1
    assert ticks[0]
    assert ticks[0].status == TickStatus.FAILURE
    assert ticks[0].timestamp == scenario.current_time.timestamp()
    assert ticks[0].tick_data.end_timestamp == scenario.current_time.timestamp()
    assert ticks[0].tick_data.auto_materialize_evaluation_id == 1
    tick_data_written = crash_location not in ('EVALUATIONS_FINISHED', 'ASSET_EVALUATIONS_ADDED')
    cursor_written = crash_location not in ('EVALUATIONS_FINISHED', 'ASSET_EVALUATIONS_ADDED', 'RUN_REQUESTS_CREATED')
    if not tick_data_written:
        assert not len(ticks[0].tick_data.reserved_run_ids)
    else:
        assert len(ticks[0].tick_data.reserved_run_ids) == 5
    cursor = _get_raw_cursor(instance)
    assert bool(cursor) == cursor_written
    freeze_datetime = scenario.current_time.add(seconds=1)
    asset_daemon_process = spawn_ctx.Process(target=_test_asset_daemon_in_subprocess, args=['auto_materialize_policy_max_materializations_not_exceeded', instance.get_ref(), freeze_datetime, None])
    asset_daemon_process.start()
    asset_daemon_process.join(timeout=60)
    ticks = instance.get_ticks(origin_id=FIXED_AUTO_MATERIALIZATION_ORIGIN_ID, selector_id=FIXED_AUTO_MATERIALIZATION_SELECTOR_ID)
    assert len(ticks) == 2
    assert ticks[0]
    assert ticks[0].status == TickStatus.SUCCESS
    assert ticks[0].timestamp == freeze_datetime.timestamp()
    assert ticks[0].tick_data.end_timestamp == freeze_datetime.timestamp()
    assert len(ticks[0].tick_data.run_ids) == 5
    assert ticks[0].tick_data.auto_materialize_evaluation_id == 1
    _assert_run_requests_match(scenario.expected_run_requests, ticks[0].tick_data.run_requests)
    runs = instance.get_runs()
    assert len(runs) == 5

    def sort_run_key_fn(run):
        if False:
            print('Hello World!')
        return (min(run.asset_selection), run.tags.get(PARTITION_NAME_TAG))
    sorted_runs = sorted(runs[:len(scenario.expected_run_requests)], key=sort_run_key_fn)
    evaluations = instance.schedule_storage.get_auto_materialize_asset_evaluations(asset_key=AssetKey('hourly'), limit=100)
    assert len(evaluations) == 1
    assert evaluations[0].evaluation.asset_key == AssetKey('hourly')
    assert evaluations[0].evaluation.run_ids == {run.run_id for run in sorted_runs}
    cursor = _get_raw_cursor(instance)
    assert cursor