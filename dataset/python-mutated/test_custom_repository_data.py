import sys
from typing import Iterator
import pytest
from dagster import file_relative_path, job, op, repository
from dagster._core.definitions.job_definition import JobDefinition
from dagster._core.definitions.repository_definition import RepositoryData
from dagster._core.instance import DagsterInstance
from dagster._core.test_utils import instance_for_test
from dagster._core.types.loadable_target_origin import LoadableTargetOrigin
from dagster._core.workspace.context import WorkspaceProcessContext
from dagster._core.workspace.load_target import GrpcServerTarget
from dagster._grpc.server import GrpcServerProcess

def define_do_something(num_calls):
    if False:
        return 10

    @op(name='do_something_' + str(num_calls))
    def do_something():
        if False:
            print('Hello World!')
        return num_calls
    return do_something

@op
def do_input(x):
    if False:
        while True:
            i = 10
    return x

def define_foo_job(num_calls: int) -> JobDefinition:
    if False:
        i = 10
        return i + 15
    do_something = define_do_something(num_calls)

    @job(name='foo_' + str(num_calls))
    def foo_job():
        if False:
            return 10
        do_input(do_something())
    return foo_job

class TestDynamicRepositoryData(RepositoryData):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._num_calls = 0

    def get_all_jobs(self):
        if False:
            while True:
                i = 10
        self._num_calls = self._num_calls + 1
        return [define_foo_job(self._num_calls)]

    def get_top_level_resources(self):
        if False:
            while True:
                i = 10
        return {}

    def get_env_vars_by_top_level_resource(self):
        if False:
            i = 10
            return i + 15
        return {}

    def get_resource_key_mapping(self):
        if False:
            print('Hello World!')
        return {}

@repository
def bar_repo():
    if False:
        i = 10
        return i + 15
    return TestDynamicRepositoryData()

@pytest.fixture(name='instance')
def instance_fixture() -> Iterator[DagsterInstance]:
    if False:
        for i in range(10):
            print('nop')
    with instance_for_test() as instance:
        yield instance

@pytest.fixture(name='workspace_process_context')
def workspace_process_context_fixture(instance: DagsterInstance) -> Iterator[WorkspaceProcessContext]:
    if False:
        while True:
            i = 10
    loadable_target_origin = LoadableTargetOrigin(executable_path=sys.executable, python_file=file_relative_path(__file__, 'test_custom_repository_data.py'))
    with GrpcServerProcess(instance_ref=instance.get_ref(), loadable_target_origin=loadable_target_origin, wait_on_exit=True) as server_process:
        with WorkspaceProcessContext(instance, GrpcServerTarget(host='localhost', socket=server_process.socket, port=server_process.port, location_name='test')) as workspace_process_context:
            yield workspace_process_context

def test_repository_data_can_reload_without_restarting(workspace_process_context: WorkspaceProcessContext):
    if False:
        i = 10
        return i + 15
    request_context = workspace_process_context.create_request_context()
    code_location = request_context.get_code_location('test')
    repo = code_location.get_repository('bar_repo')
    assert repo.has_external_job('foo_2')
    assert not repo.has_external_job('foo_1')
    external_job = repo.get_full_external_job('foo_2')
    assert external_job.has_node_invocation('do_something_2')
    workspace_process_context.reload_code_location('test')
    request_context = workspace_process_context.create_request_context()
    code_location = request_context.get_code_location('test')
    repo = code_location.get_repository('bar_repo')
    assert repo.has_external_job('foo_4')
    assert not repo.has_external_job('foo_3')
    external_job = repo.get_full_external_job('foo_4')
    assert external_job.has_node_invocation('do_something_4')