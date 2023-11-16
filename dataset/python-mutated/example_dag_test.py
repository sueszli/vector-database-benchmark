from airflow import models
import internal_unit_testing
import pytest
PROJECT_ID = 'your-project-id'

@pytest.fixture(autouse=True, scope='function')
def set_variables(airflow_database):
    if False:
        return 10
    models.Variable.set('gcp_project', PROJECT_ID)
    yield
    models.Variable.delete('gcp_project')

def test_dag_import():
    if False:
        for i in range(10):
            print('nop')
    from . import example_dag
    internal_unit_testing.assert_has_valid_dag(example_dag)