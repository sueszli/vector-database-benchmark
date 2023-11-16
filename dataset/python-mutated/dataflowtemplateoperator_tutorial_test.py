from airflow import models
import internal_unit_testing
import pytest

@pytest.fixture(autouse=True, scope='function')
def set_variables(airflow_database):
    if False:
        while True:
            i = 10
    models.Variable.set('bucket_path', 'gs://example_bucket')
    models.Variable.set('project_id', 'example-project')
    models.Variable.set('gce_zone', 'us-central1-f')
    yield
    models.Variable.delete('bucket_path')
    models.Variable.delete('project_id')
    models.Variable.delete('gce_zone')

def test_dag_import():
    if False:
        while True:
            i = 10
    'Test that the DAG file can be successfully imported.\n\n    This tests that the DAG can be parsed, but does not run it in an Airflow\n    environment. This is a recommended confidence check by the official Airflow\n    docs: https://airflow.incubator.apache.org/tutorial.html#testing\n    '
    from . import dataflowtemplateoperator_tutorial as module
    internal_unit_testing.assert_has_valid_dag(module)