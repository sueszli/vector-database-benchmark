import os
import pytest
from dagster import DagsterInstance, job, op, reconstructable, repository
from dagster._core.execution.api import execute_job
from dagster._core.execution.plan.resume_retry import ReexecutionStrategy
from dagster._core.storage.dagster_run import DagsterRunStatus
from dagster._core.storage.tags import RESUME_RETRY_TAG
from dagster._core.test_utils import environ, instance_for_test, poll_for_finished_run, step_did_not_run, step_succeeded
from dagster._core.workspace.context import WorkspaceProcessContext
from dagster._core.workspace.load_target import PythonFileTarget
CONDITIONAL_FAIL_ENV = 'DAGSTER_CONDIIONAL_FAIL'

@op
def before_failure():
    if False:
        for i in range(10):
            print('nop')
    return 'hello'

@op
def conditional_fail(_, input_value):
    if False:
        i = 10
        return i + 15
    if os.environ.get(CONDITIONAL_FAIL_ENV):
        raise Exception('env set, failing!')
    return input_value

@op
def after_failure(_, input_value):
    if False:
        return 10
    return input_value

@job(tags={'foo': 'bar'})
def conditional_fail_job():
    if False:
        while True:
            i = 10
    after_failure(conditional_fail(before_failure()))

@repository
def repo():
    if False:
        for i in range(10):
            print('nop')
    return [conditional_fail_job]

@pytest.fixture(name='instance', scope='module')
def instance_fixture():
    if False:
        return 10
    with instance_for_test() as instance:
        yield instance

@pytest.fixture(name='workspace', scope='module')
def workspace_fixture(instance):
    if False:
        print('Hello World!')
    with WorkspaceProcessContext(instance, PythonFileTarget(python_file=__file__, attribute=None, working_directory=None, location_name='repo_loc')) as workspace_process_context:
        yield workspace_process_context.create_request_context()

@pytest.fixture(name='code_location', scope='module')
def code_location_fixture(workspace):
    if False:
        print('Hello World!')
    return workspace.get_code_location('repo_loc')

@pytest.fixture(name='external_job', scope='module')
def external_job_fixture(code_location):
    if False:
        print('Hello World!')
    return code_location.get_repository('repo').get_full_external_job('conditional_fail_job')

@pytest.fixture(name='failed_run', scope='module')
def failed_run_fixture(instance):
    if False:
        for i in range(10):
            print('nop')
    with environ({CONDITIONAL_FAIL_ENV: '1'}):
        result = execute_job(reconstructable(conditional_fail_job), instance=instance, tags={'fizz': 'buzz', 'foo': 'not bar!'})
    assert not result.success
    return instance.get_run_by_id(result.run_id)

def test_create_reexecuted_run_from_failure(instance: DagsterInstance, workspace, code_location, external_job, failed_run):
    if False:
        i = 10
        return i + 15
    run = instance.create_reexecuted_run(parent_run=failed_run, code_location=code_location, external_job=external_job, strategy=ReexecutionStrategy.FROM_FAILURE)
    assert run.tags[RESUME_RETRY_TAG] == 'true'
    assert set(run.step_keys_to_execute) == {'conditional_fail', 'after_failure'}
    instance.launch_run(run.run_id, workspace)
    run = poll_for_finished_run(instance, run.run_id)
    assert run.status == DagsterRunStatus.SUCCESS
    assert step_did_not_run(instance, run, 'before_failure')
    assert step_succeeded(instance, run, 'conditional_fail')
    assert step_succeeded(instance, run, 'after_failure')

def test_create_reexecuted_run_from_failure_tags(instance: DagsterInstance, code_location, external_job, failed_run):
    if False:
        while True:
            i = 10
    run = instance.create_reexecuted_run(parent_run=failed_run, code_location=code_location, external_job=external_job, strategy=ReexecutionStrategy.FROM_FAILURE)
    assert run.tags['foo'] == 'bar'
    assert 'fizz' not in run.tags
    run = instance.create_reexecuted_run(parent_run=failed_run, code_location=code_location, external_job=external_job, strategy=ReexecutionStrategy.FROM_FAILURE, use_parent_run_tags=True)
    assert run.tags['foo'] == 'not bar!'
    assert run.tags['fizz'] == 'buzz'
    run = instance.create_reexecuted_run(parent_run=failed_run, code_location=code_location, external_job=external_job, strategy=ReexecutionStrategy.FROM_FAILURE, use_parent_run_tags=True, extra_tags={'fizz': 'not buzz!!'})
    assert run.tags['foo'] == 'not bar!'
    assert run.tags['fizz'] == 'not buzz!!'

def test_create_reexecuted_run_all_steps(instance: DagsterInstance, workspace, code_location, external_job, failed_run):
    if False:
        for i in range(10):
            print('nop')
    run = instance.create_reexecuted_run(parent_run=failed_run, code_location=code_location, external_job=external_job, strategy=ReexecutionStrategy.ALL_STEPS)
    assert RESUME_RETRY_TAG not in run.tags
    instance.launch_run(run.run_id, workspace)
    run = poll_for_finished_run(instance, run.run_id)
    assert run.status == DagsterRunStatus.SUCCESS
    assert step_succeeded(instance, run, 'before_failure')
    assert step_succeeded(instance, run, 'conditional_fail')
    assert step_succeeded(instance, run, 'after_failure')