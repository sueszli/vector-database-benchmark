from airflow import models
import internal_unit_testing
import pytest

@pytest.fixture(autouse=True, scope='function')
def set_variables(airflow_database):
    if False:
        while True:
            i = 10
    models.Variable.set('gcs_bucket', 'example_bucket')
    models.Variable.set('gcp_project', 'example-project')
    models.Variable.set('gce_zone', 'us-central1-f')
    models.Variable.set('email', 'notify@example.com')
    yield
    models.Variable.delete('gcs_bucket')
    models.Variable.delete('gcp_project')
    models.Variable.delete('gce_zone')
    models.Variable.delete('email')

def test_dag_import():
    if False:
        return 10
    'Test that the DAG file can be successfully imported.\n\n    This tests that the DAG can be parsed, but does not run it in an Airflow\n    environment. This is a recommended confidence check by the official Airflow\n    docs: https://airflow.incubator.apache.org/tutorial.html#testing\n    '
    from . import bq_notify as module
    internal_unit_testing.assert_has_valid_dag(module)