from airflow import models
import internal_unit_testing
import pytest

@pytest.fixture(autouse=True, scope='function')
def set_variables(airflow_database):
    if False:
        return 10
    models.Variable.set('project_id', 'example-project')
    yield
    models.Variable.delete('project_id')

def test_dag_import():
    if False:
        i = 10
        return i + 15
    'Test that the DAG file can be successfully imported.\n\n    This tests that the DAG can be parsed, but does not run it in an Airflow\n    environment. This is a recommended confidence check by the official Airflow\n    docs: https://airflow.incubator.apache.org/tutorial.html#testing\n    '
    from . import dataproc_workflow_template_instantiate_operator_tutorial as module
    internal_unit_testing.assert_has_valid_dag(module)