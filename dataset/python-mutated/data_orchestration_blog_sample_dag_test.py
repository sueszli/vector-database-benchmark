from airflow import models
import internal_unit_testing
import pytest
PROJECT_ID = 'your-project-id'
DATASET = 'your-bq-output-dataset'
TABLE = 'your-bq-output-table'

@pytest.fixture(autouse=True, scope='function')
def set_variables(airflow_database):
    if False:
        while True:
            i = 10
    models.Variable.set('gcp_project', PROJECT_ID)
    models.Variable.set('bigquery_dataset', DATASET)
    models.Variable.set('bigquery_table', TABLE)
    yield
    models.Variable.delete('gcp_project')
    models.Variable.delete('bigquery_dataset')
    models.Variable.delete('bigquery_table')

def test_dag_import():
    if False:
        print('Hello World!')
    from . import data_orchestration_blog_sample_dag
    internal_unit_testing.assert_has_valid_dag(data_orchestration_blog_sample_dag)