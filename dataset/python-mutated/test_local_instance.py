import gc
import os
import tempfile
import time
import types
from concurrent.futures import ThreadPoolExecutor
import pytest
import yaml
from dagster import DagsterEventType, DagsterInvalidConfigError, In, Out, Output, _check as check, job, op
from dagster._core.definitions.events import RetryRequested
from dagster._core.execution.stats import StepEventStatus
from dagster._core.instance import DagsterInstance, InstanceRef, InstanceType
from dagster._core.launcher import DefaultRunLauncher
from dagster._core.run_coordinator import DefaultRunCoordinator
from dagster._core.storage.dagster_run import DagsterRun, DagsterRunStatus
from dagster._core.storage.event_log import SqliteEventLogStorage
from dagster._core.storage.local_compute_log_manager import LocalComputeLogManager
from dagster._core.storage.root import LocalArtifactStorage
from dagster._core.storage.runs import SqliteRunStorage
from dagster._core.test_utils import environ
from packaging import version
from sqlalchemy import __version__ as sqlalchemy_version

def test_fs_stores():
    if False:
        return 10

    @job
    def simple():
        if False:
            i = 10
            return i + 15

        @op
        def easy(context):
            if False:
                i = 10
                return i + 15
            context.log.info('easy')
            return 'easy'
        easy()
    with tempfile.TemporaryDirectory() as temp_dir:
        with environ({'DAGSTER_HOME': temp_dir}):
            run_store = SqliteRunStorage.from_local(temp_dir)
            event_store = SqliteEventLogStorage(temp_dir)
            compute_log_manager = LocalComputeLogManager(temp_dir)
            instance = DagsterInstance(instance_type=InstanceType.PERSISTENT, local_artifact_storage=LocalArtifactStorage(temp_dir), run_storage=run_store, event_storage=event_store, compute_log_manager=compute_log_manager, run_coordinator=DefaultRunCoordinator(), run_launcher=DefaultRunLauncher(), ref=InstanceRef.from_dir(temp_dir), settings={'telemetry': {'enabled': False}})
            result = simple.execute_in_process(instance=instance)
            assert run_store.has_run(result.run_id)
            assert instance.get_run_by_id(result.run_id).status == DagsterRunStatus.SUCCESS
            assert DagsterEventType.PIPELINE_SUCCESS in [event.dagster_event.event_type for event in event_store.get_logs_for_run(result.run_id) if event.is_dagster_event]
            stats = event_store.get_stats_for_run(result.run_id)
            assert stats.steps_succeeded == 1
            assert stats.end_time is not None

def test_init_compute_log_with_bad_config():
    if False:
        print('Hello World!')
    with tempfile.TemporaryDirectory() as tmpdir_path:
        with open(os.path.join(tmpdir_path, 'dagster.yaml'), 'w', encoding='utf8') as fd:
            yaml.dump({'compute_logs': {'garbage': 'flargh'}}, fd, default_flow_style=False)
        with pytest.raises(DagsterInvalidConfigError, match='Received unexpected config entry "garbage"'):
            DagsterInstance.from_ref(InstanceRef.from_dir(tmpdir_path))

def test_init_compute_log_with_bad_config_override():
    if False:
        for i in range(10):
            print('nop')
    with tempfile.TemporaryDirectory() as tmpdir_path:
        with pytest.raises(DagsterInvalidConfigError, match='Received unexpected config entry "garbage"'):
            DagsterInstance.from_ref(InstanceRef.from_dir(tmpdir_path, overrides={'compute_logs': {'garbage': 'flargh'}}))

def test_init_compute_log_with_bad_config_module():
    if False:
        for i in range(10):
            print('nop')
    with tempfile.TemporaryDirectory() as tmpdir_path:
        with open(os.path.join(tmpdir_path, 'dagster.yaml'), 'w', encoding='utf8') as fd:
            yaml.dump({'compute_logs': {'module': 'flargh', 'class': 'Woble', 'config': {}}}, fd, default_flow_style=False)
        with pytest.raises(check.CheckError, match="Couldn't import module"):
            DagsterInstance.from_ref(InstanceRef.from_dir(tmpdir_path)).compute_log_manager
MOCK_HAS_RUN_CALLED = False

def test_get_run_by_id():
    if False:
        while True:
            i = 10
    with tempfile.TemporaryDirectory() as tmpdir_path:
        instance = DagsterInstance.from_ref(InstanceRef.from_dir(tmpdir_path))
        assert instance.get_runs() == []
        dagster_run = DagsterRun('foo_job', 'new_run')
        assert instance.get_run_by_id(dagster_run.run_id) is None
        instance.add_run(dagster_run)
        assert instance.get_runs() == [dagster_run]
        assert instance.get_run_by_id(dagster_run.run_id) == dagster_run
    with tempfile.TemporaryDirectory() as tmpdir_path:
        instance = DagsterInstance.from_ref(InstanceRef.from_dir(tmpdir_path))
        run = DagsterRun(job_name='foo_job', run_id='bar_run')

        def _has_run(self, run_id):
            if False:
                while True:
                    i = 10
            global MOCK_HAS_RUN_CALLED
            if not self._run_storage.has_run(run_id) and (not MOCK_HAS_RUN_CALLED):
                self._run_storage.add_run(DagsterRun(job_name='foo_job', run_id=run_id))
                return False
            else:
                return self._run_storage.has_run(run_id)
        instance.has_run = types.MethodType(_has_run, instance)
        assert instance.get_run_by_id(run.run_id) is None
    global MOCK_HAS_RUN_CALLED
    MOCK_HAS_RUN_CALLED = False
    with tempfile.TemporaryDirectory() as tmpdir_path:
        instance = DagsterInstance.from_ref(InstanceRef.from_dir(tmpdir_path))
        run = DagsterRun(job_name='foo_job', run_id='bar_run')

        def _has_run(self, run_id):
            if False:
                return 10
            global MOCK_HAS_RUN_CALLED
            if not self._run_storage.has_run(run_id) and (not MOCK_HAS_RUN_CALLED):
                self._run_storage.add_run(DagsterRun(job_name='foo_job', run_id=run_id))
                MOCK_HAS_RUN_CALLED = True
                return False
            elif self._run_storage.has_run(run_id) and MOCK_HAS_RUN_CALLED:
                MOCK_HAS_RUN_CALLED = False
                return True
            else:
                return False
        instance.has_run = types.MethodType(_has_run, instance)
        assert instance.get_run_by_id(run.run_id) is None

def test_run_step_stats():
    if False:
        for i in range(10):
            print('nop')
    _called = None

    @job
    def simple():
        if False:
            return 10

        @op
        def should_succeed(context):
            if False:
                for i in range(10):
                    print('nop')
            time.sleep(0.001)
            context.log.info('succeed')
            return 'yay'

        @op(ins={'_input': In(str)}, out=Out(str))
        def should_fail(context, _input):
            if False:
                while True:
                    i = 10
            context.log.info('fail')
            raise Exception('booo')

        @op
        def should_not_execute(_, x):
            if False:
                while True:
                    i = 10
            _called = True
            return x
        should_not_execute(should_fail(should_succeed()))
    with tempfile.TemporaryDirectory() as tmpdir_path:
        instance = DagsterInstance.from_ref(InstanceRef.from_dir(tmpdir_path))
        result = simple.execute_in_process(instance=instance, raise_on_error=False)
        step_stats = sorted(instance.get_run_step_stats(result.run_id), key=lambda x: x.end_time)
        assert len(step_stats) == 2
        assert step_stats[0].step_key == 'should_succeed'
        assert step_stats[0].status == StepEventStatus.SUCCESS
        assert step_stats[0].end_time > step_stats[0].start_time
        assert step_stats[0].attempts == 1
        assert step_stats[1].step_key == 'should_fail'
        assert step_stats[1].status == StepEventStatus.FAILURE
        assert step_stats[1].end_time > step_stats[0].start_time
        assert step_stats[1].attempts == 1
        assert not _called

def test_run_step_stats_with_retries():
    if False:
        i = 10
        return i + 15
    _called = None
    _count = {'total': 0}

    @job
    def simple():
        if False:
            while True:
                i = 10

        @op
        def should_succeed(_):
            if False:
                for i in range(10):
                    print('nop')
            if _count['total'] < 2:
                _count['total'] += 1
                raise RetryRequested(max_retries=3)
            yield Output('yay')

        @op(ins={'_input': In(str)}, out=Out(str))
        def should_retry(context, _input):
            if False:
                i = 10
                return i + 15
            raise RetryRequested(max_retries=3)

        @op
        def should_not_execute(_, x):
            if False:
                for i in range(10):
                    print('nop')
            _called = True
            return x
        should_not_execute(should_retry(should_succeed()))
    with tempfile.TemporaryDirectory() as tmpdir_path:
        instance = DagsterInstance.from_ref(InstanceRef.from_dir(tmpdir_path))
        result = simple.execute_in_process(instance=instance, raise_on_error=False)
        step_stats = instance.get_run_step_stats(result.run_id, step_keys=['should_retry'])
        assert len(step_stats) == 1
        assert step_stats[0].step_key == 'should_retry'
        assert step_stats[0].status == StepEventStatus.FAILURE
        assert step_stats[0].end_time > step_stats[0].start_time
        assert step_stats[0].attempts == 4
        assert not _called

def test_threaded_ephemeral_instance(caplog):
    if False:
        i = 10
        return i + 15
    gc.disable()
    try:
        n = 5
        with DagsterInstance.ephemeral() as shared_instance:

            def _instantiate_ephemeral_instance(_):
                if False:
                    return 10
                with DagsterInstance.ephemeral() as instance:
                    instance.get_runs_count()
                    instance.all_asset_keys()
                    assert instance.root_directory
                    shared_instance.get_runs_count()
                    shared_instance.all_asset_keys()
                    assert shared_instance.root_directory
                return True
            with ThreadPoolExecutor(max_workers=n, thread_name_prefix='ephemeral_worker') as executor:
                results = executor.map(_instantiate_ephemeral_instance, range(n))
                assert all(results)
        if version.parse(sqlalchemy_version) > version.parse('1.3.24'):
            assert 'SQLite objects created in a thread can only be used in that same thread.' not in caplog.text
    finally:
        gc.enable()

def test_threadsafe_local_temp_instance():
    if False:
        while True:
            i = 10
    n = 25
    gc.collect()
    baseline = len(DagsterInstance._TEMP_DIRS)
    shared = DagsterInstance.local_temp()

    def _run(_):
        if False:
            while True:
                i = 10
        shared.root_directory
        with DagsterInstance.local_temp() as instance:
            instance.root_directory
        return True
    with ThreadPoolExecutor(max_workers=n) as executor:
        results = executor.map(_run, range(n))
        assert all(results)
    shared = None
    gc.collect()
    assert baseline == len(DagsterInstance._TEMP_DIRS)