import os
import uuid
from google.cloud import bigquery
from google.cloud import storage
import pytest
import quickstart_analyzeiampolicylongrunning
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
BUCKET = f'analysis-{int(uuid.uuid4())}'
DATASET = f'analysis_{int(uuid.uuid4())}'

@pytest.fixture(scope='module')
def storage_client():
    if False:
        for i in range(10):
            print('nop')
    yield storage.Client()

@pytest.fixture(scope='module')
def bigquery_client():
    if False:
        for i in range(10):
            print('nop')
    yield bigquery.Client()

@pytest.fixture(scope='module')
def analysis_bucket(storage_client):
    if False:
        return 10
    bucket = storage_client.create_bucket(BUCKET)
    yield BUCKET
    try:
        bucket.delete(force=True)
    except Exception as e:
        print(f'Failed to delete bucket{BUCKET}')
        raise e

@pytest.fixture(scope='module')
def dataset(bigquery_client):
    if False:
        print('Hello World!')
    dataset_id = f'{PROJECT}.{DATASET}'
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = 'US'
    dataset = bigquery_client.create_dataset(dataset)
    yield DATASET
    bigquery_client.delete_dataset(dataset_id, delete_contents=True, not_found_ok=False)

def test_analyze_iam_policy_longrunning(analysis_bucket, dataset, capsys):
    if False:
        print('Hello World!')
    dump_file_path = f'gs://{analysis_bucket}/analysis-dump.txt'
    quickstart_analyzeiampolicylongrunning.analyze_iam_policy_longrunning_gcs(PROJECT, dump_file_path)
    (out, _) = capsys.readouterr()
    assert 'True' in out
    dataset_id = f'projects/{PROJECT}/datasets/{dataset}'
    quickstart_analyzeiampolicylongrunning.analyze_iam_policy_longrunning_bigquery(PROJECT, dataset_id, 'analysis_')
    (out, _) = capsys.readouterr()
    assert 'True' in out