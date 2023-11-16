import sys
from contextlib import contextmanager
import pytest
from dagster import IntMetadataValue, TextMetadataValue, job, op, repository
from dagster._api.snapshot_repository import sync_get_streaming_external_repositories_data_grpc
from dagster._core.errors import DagsterUserCodeProcessError
from dagster._core.host_representation import ExternalRepositoryData, ManagedGrpcPythonEnvCodeLocationOrigin
from dagster._core.host_representation.external import ExternalRepository
from dagster._core.host_representation.external_data import ExternalJobData
from dagster._core.host_representation.handle import RepositoryHandle
from dagster._core.host_representation.origin import ExternalRepositoryOrigin
from dagster._core.instance import DagsterInstance
from dagster._core.test_utils import instance_for_test
from dagster._core.types.loadable_target_origin import LoadableTargetOrigin
from dagster._serdes.serdes import deserialize_value
from .utils import get_bar_repo_code_location

def test_streaming_external_repositories_api_grpc(instance):
    if False:
        print('Hello World!')
    with get_bar_repo_code_location(instance) as code_location:
        external_repo_datas = sync_get_streaming_external_repositories_data_grpc(code_location.client, code_location)
        assert len(external_repo_datas) == 1
        external_repository_data = external_repo_datas['bar_repo']
        assert isinstance(external_repository_data, ExternalRepositoryData)
        assert external_repository_data.name == 'bar_repo'
        assert external_repository_data.metadata == {'string': TextMetadataValue('foo'), 'integer': IntMetadataValue(123)}

def test_streaming_external_repositories_error(instance):
    if False:
        return 10
    with get_bar_repo_code_location(instance) as code_location:
        code_location.repository_names = {'does_not_exist'}
        assert code_location.repository_names == {'does_not_exist'}
        with pytest.raises(DagsterUserCodeProcessError, match='Could not find a repository called "does_not_exist"'):
            sync_get_streaming_external_repositories_data_grpc(code_location.client, code_location)

@op
def do_something():
    if False:
        while True:
            i = 10
    return 1

@job
def giant_job():
    if False:
        i = 10
        return i + 15
    for _i in range(20000):
        do_something()

@repository
def giant_repo():
    if False:
        return 10
    return {'jobs': {'giant': giant_job}}

@contextmanager
def get_giant_repo_grpc_code_location(instance):
    if False:
        while True:
            i = 10
    with ManagedGrpcPythonEnvCodeLocationOrigin(loadable_target_origin=LoadableTargetOrigin(executable_path=sys.executable, attribute='giant_repo', module_name='dagster_tests.api_tests.test_api_snapshot_repository'), location_name='giant_repo_location').create_single_location(instance) as location:
        yield location

@pytest.mark.skip('https://github.com/dagster-io/dagster/issues/6940')
def test_giant_external_repository_streaming_grpc():
    if False:
        i = 10
        return i + 15
    with instance_for_test() as instance:
        with get_giant_repo_grpc_code_location(instance) as code_location:
            external_repos_data = sync_get_streaming_external_repositories_data_grpc(code_location.client, code_location)
            assert len(external_repos_data) == 1
            external_repository_data = external_repos_data['giant_repo']
            assert isinstance(external_repository_data, ExternalRepositoryData)
            assert external_repository_data.name == 'giant_repo'

def test_defer_snapshots(instance: DagsterInstance):
    if False:
        return 10
    with get_bar_repo_code_location(instance) as code_location:
        repo_origin = ExternalRepositoryOrigin(code_location.origin, 'bar_repo')
        ser_repo_data = code_location.client.external_repository(repo_origin, defer_snapshots=True)
        _state = {}

        def _ref_to_data(ref):
            if False:
                for i in range(10):
                    print('nop')
            _state['cnt'] = _state.get('cnt', 0) + 1
            reply = code_location.client.external_job(repo_origin, ref.name)
            return deserialize_value(reply.serialized_job_data, ExternalJobData)
        external_repository_data = deserialize_value(ser_repo_data, ExternalRepositoryData)
        assert external_repository_data.external_job_refs and len(external_repository_data.external_job_refs) == 6
        assert external_repository_data.external_job_datas is None
        repo = ExternalRepository(external_repository_data, RepositoryHandle(repository_name='bar_repo', code_location=code_location), ref_to_data_fn=_ref_to_data)
        jobs = repo.get_all_external_jobs()
        assert len(jobs) == 6
        assert _state.get('cnt', 0) == 0
        job = jobs[0]
        _ = job.computed_job_snapshot_id
        assert _state.get('cnt', 0) == 0
        _ = job.name
        assert _state.get('cnt', 0) == 0
        _ = job.active_presets
        assert _state.get('cnt', 0) == 0
        _ = job.job_snapshot
        assert _state.get('cnt', 0) == 1
        _ = job.job_snapshot
        assert _state.get('cnt', 0) == 1
        job = repo.get_all_external_jobs()[0]
        _ = job.job_snapshot
        assert _state.get('cnt', 0) == 1