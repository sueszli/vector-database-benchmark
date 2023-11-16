import re
import sys
import types
import pytest
from dagster import DagsterInvariantViolationError, GraphDefinition, JobDefinition, execute_job, job, op, reconstructable, repository
from dagster._core.code_pointer import FileCodePointer
from dagster._core.definitions.reconstruct import ReconstructableJob
from dagster._core.origin import DEFAULT_DAGSTER_ENTRY_POINT, JobPythonOrigin, RepositoryPythonOrigin
from dagster._core.snap import JobSnapshot, create_job_snapshot_id
from dagster._core.test_utils import instance_for_test
from dagster._utils import file_relative_path
from dagster._utils.hosted_user_process import recon_job_from_origin

@op
def the_op():
    if False:
        for i in range(10):
            print('nop')
    return 1

@job
def the_job():
    if False:
        return 10
    the_op()

def get_the_pipeline():
    if False:
        while True:
            i = 10
    return the_job

def not_the_pipeline():
    if False:
        print('Hello World!')
    return None

def get_with_args(_x):
    if False:
        print('Hello World!')
    return the_job
lambda_version = lambda : the_job

def pid(pipeline_def):
    if False:
        for i in range(10):
            print('nop')
    return create_job_snapshot_id(JobSnapshot.from_job_def(pipeline_def))

@job
def some_job():
    if False:
        return 10
    pass

@repository
def some_repo():
    if False:
        i = 10
        return i + 15
    return [some_job]

def test_function():
    if False:
        return 10
    recon_pipe = reconstructable(get_the_pipeline)
    assert pid(recon_pipe.get_definition()) == pid(the_job)

def test_decorator():
    if False:
        print('Hello World!')
    recon_pipe = reconstructable(the_job)
    assert pid(recon_pipe.get_definition()) == pid(the_job)

def test_lambda():
    if False:
        i = 10
        return i + 15
    with pytest.raises(DagsterInvariantViolationError, match='Reconstructable target can not be a lambda'):
        reconstructable(lambda_version)

def test_not_defined_in_module(mocker):
    if False:
        while True:
            i = 10
    mocker.patch('inspect.getmodule', return_value=types.ModuleType('__main__'))
    with pytest.raises(DagsterInvariantViolationError, match=re.escape('reconstructable() can not reconstruct jobs defined in interactive environments')):
        reconstructable(get_the_pipeline)

def test_manual_instance():
    if False:
        return 10
    defn = JobDefinition(graph_def=GraphDefinition(node_defs=[the_op], name='test'))
    with pytest.raises(DagsterInvariantViolationError, match='Reconstructable target was not a function returning a job definition, or a job definition produced by a decorated function.'):
        reconstructable(defn)

def test_args_fails():
    if False:
        i = 10
        return i + 15
    with pytest.raises(DagsterInvariantViolationError, match='Reconstructable target must be callable with no arguments'):
        reconstructable(get_with_args)

def test_bad_target():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(DagsterInvariantViolationError, match=re.escape('Loadable attributes must be either a JobDefinition, GraphDefinition, or RepositoryDefinition. Got None.')):
        reconstructable(not_the_pipeline)

def test_inner_scope():
    if False:
        while True:
            i = 10

    def get_the_pipeline_inner():
        if False:
            print('Hello World!')
        return the_job
    with pytest.raises(DagsterInvariantViolationError, match='Use a function or decorated function defined at module scope'):
        reconstructable(get_the_pipeline_inner)

def test_inner_decorator():
    if False:
        return 10

    @job
    def pipe():
        if False:
            i = 10
            return i + 15
        the_op()
    with pytest.raises(DagsterInvariantViolationError, match='Use a function or decorated function defined at module scope'):
        reconstructable(pipe)

def test_op_selection():
    if False:
        i = 10
        return i + 15
    recon_pipe = reconstructable(get_the_pipeline)
    sub_pipe_full = recon_pipe.get_subset(op_selection={'the_op'})
    assert sub_pipe_full.op_selection == {'the_op'}
    sub_pipe_unresolved = recon_pipe.get_subset(op_selection={'the_op+'})
    assert sub_pipe_unresolved.op_selection == {'the_op+'}

def test_reconstructable_module():
    if False:
        for i in range(10):
            print('nop')
    original_sys_path = sys.path
    try:
        sys.path.insert(0, file_relative_path(__file__, '.'))
        from foo import bar_job
        reconstructable(bar_job)
    finally:
        sys.path = original_sys_path

def test_reconstruct_from_origin():
    if False:
        for i in range(10):
            print('nop')
    origin = JobPythonOrigin(job_name='foo_pipe', repository_origin=RepositoryPythonOrigin(executable_path='my_python', code_pointer=FileCodePointer(python_file='foo.py', fn_name='bar', working_directory='/'), container_image='my_image', entry_point=DEFAULT_DAGSTER_ENTRY_POINT, container_context={'docker': {'registry': 'my_reg'}}))
    recon_job = recon_job_from_origin(origin)
    assert recon_job.job_name == origin.job_name
    assert recon_job.repository.pointer == origin.repository_origin.code_pointer
    assert recon_job.repository.container_image == origin.repository_origin.container_image
    assert recon_job.repository.executable_path == origin.repository_origin.executable_path
    assert recon_job.repository.container_context == origin.repository_origin.container_context

def test_reconstructable_memoize():
    if False:
        return 10
    recon_job = reconstructable(some_job)
    recon_job.get_definition()
    starting_misses = ReconstructableJob.get_definition.cache_info().misses
    with instance_for_test() as instance:
        result = execute_job(recon_job, instance=instance)
    assert result.success
    assert ReconstructableJob.get_definition.cache_info().misses == starting_misses
    assert ReconstructableJob.get_definition.cache_info().hits > 1