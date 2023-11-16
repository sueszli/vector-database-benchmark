from airflow import models
import internal_unit_testing
import pytest

@pytest.fixture(autouse=True, scope='function')
def set_variables(airflow_database):
    if False:
        while True:
            i = 10
    models.Variable.set('gcs_bucket', 'example-bucket')
    models.Variable.set('gcp_project', 'example-project')
    models.Variable.set('gce_region', 'us-central1')
    yield
    models.Variable.delete('gcs_bucket')
    models.Variable.delete('gcp_project')
    models.Variable.delete('gce_region')

def test_dag_import():
    if False:
        for i in range(10):
            print('nop')
    'Test that the DAG file can be successfully imported.\n\n    This tests that the DAG can be parsed, but does not run it in an Airflow\n    environment. This is a recommended confidence check by the official Airflow\n    docs: https://airflow.incubator.apache.org/tutorial.html#testing\n    '
    from . import hadoop_tutorial as module
    internal_unit_testing.assert_has_valid_dag(module)