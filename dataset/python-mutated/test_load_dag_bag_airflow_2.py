import os
import tempfile
import pytest
from airflow import __version__ as airflow_version
from airflow.models import DagBag
from dagster_airflow import make_dagster_definitions_from_airflow_dags_path, make_dagster_definitions_from_airflow_example_dags, make_dagster_job_from_airflow_dag
from dagster_airflow_tests.marks import requires_local_db, requires_no_db
from ..airflow_utils import test_make_from_dagbag_inputs_airflow_2

@pytest.mark.skipif(airflow_version < '2.0.0', reason='requires airflow 2')
@pytest.mark.parametrize('path_and_content_tuples, fn_arg_path, expected_job_names', test_make_from_dagbag_inputs_airflow_2)
@requires_no_db
def test_make_definition(path_and_content_tuples, fn_arg_path, expected_job_names):
    if False:
        while True:
            i = 10
    with tempfile.TemporaryDirectory() as tmpdir_path:
        for (path, content) in path_and_content_tuples:
            with open(os.path.join(tmpdir_path, path), 'wb') as f:
                f.write(bytes(content.encode('utf-8')))
        definitions = make_dagster_definitions_from_airflow_dags_path(tmpdir_path) if fn_arg_path is None else make_dagster_definitions_from_airflow_dags_path(os.path.join(tmpdir_path, fn_arg_path))
        repo = definitions.get_repository_def()
        for job_name in expected_job_names:
            assert repo.has_job(job_name)
            job = definitions.get_job_def(job_name)
            result = job.execute_in_process()
            assert result.success
            for event in result.all_events:
                assert event.event_type_value != 'STEP_FAILURE'
        assert set(repo.job_names) == set(expected_job_names)

@pytest.fixture(scope='module')
def airflow_examples_repo():
    if False:
        i = 10
        return i + 15
    definitions = make_dagster_definitions_from_airflow_example_dags()
    return definitions.get_repository_def()

def get_examples_airflow_repo_params():
    if False:
        print('Hello World!')
    definitions = make_dagster_definitions_from_airflow_example_dags()
    repo = definitions.get_repository_def()
    params = []
    no_job_run_dags = ['example_kubernetes_executor', 'example_passing_params_via_test_command', 'example_python_operator', 'example_dag_decorator', 'example_trigger_target_dag', 'example_trigger_controller_dag', 'example_subdag_operator', 'example_sensors', 'example_dynamic_task_mapping', 'example_dynamic_task_mapping_with_no_taskflow_operators']
    for job_name in repo.job_names:
        params.append(pytest.param(job_name, True if job_name in no_job_run_dags else False, id=job_name))
    return params

@pytest.mark.skipif(airflow_version < '2.0.0', reason='requires airflow 2')
@pytest.mark.parametrize('job_name, exclude_from_execution_tests', get_examples_airflow_repo_params())
@requires_local_db
def test_airflow_example_dags(airflow_examples_repo, job_name, exclude_from_execution_tests):
    if False:
        print('Hello World!')
    assert airflow_examples_repo.has_job(job_name)
    if not exclude_from_execution_tests:
        job = airflow_examples_repo.get_job(job_name)
        result = job.execute_in_process()
        assert result.success
        for event in result.all_events:
            assert event.event_type_value != 'STEP_FAILURE'
RETRY_DAG = '\nfrom airflow import models\n\nfrom airflow.operators.bash import BashOperator\n\nimport datetime\n\ndefault_args = {"start_date": datetime.datetime(2023, 2, 1), "retries": 3}\n\nwith models.DAG(\n    dag_id="retry_dag", default_args=default_args, schedule=\'0 0 * * *\', tags=[\'example\'],\n) as retry_dag:\n    foo = BashOperator(\n        task_id="foo", bash_command="echo foo", retries=1\n    )\n\n    bar = BashOperator(\n        task_id="bar", bash_command="echo bar"\n    )\n'

@pytest.mark.skipif(airflow_version < '2.0.0', reason='requires airflow 2')
@requires_local_db
def test_retry_conversion():
    if False:
        return 10
    with tempfile.TemporaryDirectory(suffix='retries') as tmpdir_path:
        with open(os.path.join(tmpdir_path, 'dag.py'), 'wb') as f:
            f.write(bytes(RETRY_DAG.encode('utf-8')))
        dag_bag = DagBag(dag_folder=tmpdir_path)
        retry_dag = dag_bag.get_dag(dag_id='retry_dag')
        job = make_dagster_job_from_airflow_dag(dag=retry_dag)
        result = job.execute_in_process()
        assert result.success
        for event in result.all_events:
            assert event.event_type_value != 'STEP_FAILURE'