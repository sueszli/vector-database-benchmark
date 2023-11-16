from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from unittest import mock
import pendulum
from dagster import daily_partitioned_config, job, op, repository
from dagster._core.definitions.decorators.schedule_decorator import schedule
from dagster._core.host_representation import external_job_data_from_def, external_repository_data_from_def
from dagster._core.host_representation.external_data import ExternalTimeWindowPartitionsDefinitionData
from dagster._core.snap.job_snapshot import create_job_snapshot_id
from dagster._core.test_utils import in_process_test_workspace, instance_for_test
from dagster._core.types.loadable_target_origin import LoadableTargetOrigin
from dagster._serdes import serialize_pp

@op
def foo_op(_):
    if False:
        return 10
    pass

@schedule(cron_schedule='@daily', job_name='foo_job', execution_timezone='US/Central')
def foo_schedule():
    if False:
        print('Hello World!')
    return {}

@daily_partitioned_config(start_date=datetime(2020, 1, 1), minute_offset=15)
def my_partitioned_config(_start: datetime, _end: datetime):
    if False:
        print('Hello World!')
    return {}

@job(config=my_partitioned_config)
def foo_job():
    if False:
        print('Hello World!')
    foo_op()

@repository
def a_repo():
    if False:
        print('Hello World!')
    return [foo_job]

def test_external_repository_data(snapshot):
    if False:
        for i in range(10):
            print('nop')

    @repository
    def repo():
        if False:
            for i in range(10):
                print('nop')
        return [foo_job, foo_schedule]
    external_repo_data = external_repository_data_from_def(repo)
    assert external_repo_data.get_external_job_data('foo_job')
    assert external_repo_data.get_external_schedule_data('foo_schedule')
    job_partition_set_data = external_repo_data.get_external_partition_set_data('foo_job_partition_set')
    assert job_partition_set_data
    assert isinstance(job_partition_set_data.external_partitions_data, ExternalTimeWindowPartitionsDefinitionData)
    now = pendulum.now()
    assert job_partition_set_data.external_partitions_data.get_partitions_definition().get_partition_keys(now) == my_partitioned_config.partitions_def.get_partition_keys(now)
    snapshot.assert_match(serialize_pp(external_repo_data))

def test_external_job_data(snapshot):
    if False:
        return 10
    snapshot.assert_match(serialize_pp(external_job_data_from_def(foo_job)))

@mock.patch('dagster._core.host_representation.job_index.create_job_snapshot_id')
def test_external_repo_shared_index(snapshot_mock):
    if False:
        print('Hello World!')
    snapshot_mock.side_effect = create_job_snapshot_id
    with instance_for_test() as instance:
        with in_process_test_workspace(instance, LoadableTargetOrigin(python_file=__file__)) as workspace:

            def _fetch_snap_id():
                if False:
                    print('Hello World!')
                location = workspace.code_locations[0]
                ex_repo = next(iter(location.get_repositories().values()))
                return ex_repo.get_all_external_jobs()[0].identifying_job_snapshot_id
            _fetch_snap_id()
            assert snapshot_mock.call_count == 1
            _fetch_snap_id()
            assert snapshot_mock.call_count == 1

@mock.patch('dagster._core.host_representation.job_index.create_job_snapshot_id')
def test_external_repo_shared_index_threaded(snapshot_mock):
    if False:
        for i in range(10):
            print('nop')
    snapshot_mock.side_effect = create_job_snapshot_id
    with instance_for_test() as instance:
        with in_process_test_workspace(instance, LoadableTargetOrigin(python_file=__file__)) as workspace:

            def _fetch_snap_id():
                if False:
                    print('Hello World!')
                location = workspace.code_locations[0]
                ex_repo = next(iter(location.get_repositories().values()))
                return ex_repo.get_all_external_jobs()[0].identifying_job_snapshot_id
            with ThreadPoolExecutor() as executor:
                wait([executor.submit(_fetch_snap_id) for _ in range(100)])
            assert snapshot_mock.call_count == 1