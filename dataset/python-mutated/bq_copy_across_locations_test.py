import os
import os.path
from airflow import models
import internal_unit_testing
import pytest

@pytest.fixture(autouse=True, scope='function')
def set_variables(airflow_database):
    if False:
        while True:
            i = 10
    example_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'bq_copy_eu_to_us_sample.csv')
    models.Variable.set('table_list_file_path', example_file_path)
    models.Variable.set('gcs_source_bucket', 'example-project')
    models.Variable.set('gcs_dest_bucket', 'us-central1-f')
    yield
    models.Variable.delete('table_list_file_path')
    models.Variable.delete('gcs_source_bucket')
    models.Variable.delete('gcs_dest_bucket')

def test_dag():
    if False:
        i = 10
        return i + 15
    'Test that the DAG file can be successfully imported.\n\n    This tests that the DAG can be parsed, but does not run it in an Airflow\n    environment. This is a recommended confidence check by the official Airflow\n    docs: https://airflow.incubator.apache.org/tutorial.html#testing\n    '
    from . import bq_copy_across_locations as module
    internal_unit_testing.assert_has_valid_dag(module)