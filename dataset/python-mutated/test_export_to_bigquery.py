import uuid
import google.auth
from google.cloud import bigquery
import pytest
import export_to_bigquery
GCLOUD_TESTS_PREFIX = 'python_samples_tests'

@pytest.fixture
def project_id():
    if False:
        i = 10
        return i + 15
    (_, project_id) = google.auth.default()
    return project_id

@pytest.fixture
def unique_id():
    if False:
        i = 10
        return i + 15
    uuid_hex = uuid.uuid4().hex[:8]
    return f'{GCLOUD_TESTS_PREFIX}_{uuid_hex}'

@pytest.fixture
def bigquery_resources(project_id, unique_id):
    if False:
        while True:
            i = 10
    bigquery_client = bigquery.Client()
    dataset_id = unique_id
    table_id = unique_id
    dataset = bigquery.Dataset(f'{project_id}.{dataset_id}')
    dataset.location = 'US'
    bigquery_client.create_dataset(dataset, timeout=30)
    table = bigquery.Table(f'{project_id}.{dataset_id}.{table_id}')
    bigquery_client.create_table(table)
    yield (dataset_id, table_id)
    bigquery_client.delete_dataset(dataset_id, delete_contents=True)

def test_export_data_to_bigquery(capsys, project_id, bigquery_resources):
    if False:
        i = 10
        return i + 15
    (dataset_id, table_id) = bigquery_resources
    export_to_bigquery.export_to_bigquery(project_id, project_id, dataset_id, table_id)
    (out, err) = capsys.readouterr()
    assert 'Exported data to BigQuery' in out