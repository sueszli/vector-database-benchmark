import os
import uuid
from google.cloud import asset_v1
from google.cloud import bigquery
from google.cloud import storage
import pytest
import quickstart_exportassets
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
BUCKET = f'assets-{uuid.uuid4().hex}'
DATASET = f'assets_{int(uuid.uuid4())}'

@pytest.fixture(scope='module')
def storage_client():
    if False:
        print('Hello World!')
    yield storage.Client()

@pytest.fixture(scope='module')
def bigquery_client():
    if False:
        while True:
            i = 10
    yield bigquery.Client()

@pytest.fixture(scope='module')
def asset_bucket(storage_client):
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
        while True:
            i = 10
    dataset_id = f'{PROJECT}.{DATASET}'
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = 'US'
    dataset = bigquery_client.create_dataset(dataset)
    yield DATASET
    bigquery_client.delete_dataset(dataset_id, delete_contents=True, not_found_ok=False)

def test_export_assets(asset_bucket, dataset, capsys):
    if False:
        while True:
            i = 10
    content_type = asset_v1.ContentType.IAM_POLICY
    dump_file_path = f'gs://{asset_bucket}/assets-dump.txt'
    quickstart_exportassets.export_assets(PROJECT, dump_file_path, content_type=content_type)
    (out, _) = capsys.readouterr()
    assert dump_file_path in out
    content_type = asset_v1.ContentType.RESOURCE
    dataset_id = f'projects/{PROJECT}/datasets/{dataset}'
    quickstart_exportassets.export_assets_bigquery(PROJECT, dataset_id, 'assettable', content_type)
    (out, _) = capsys.readouterr()
    assert dataset_id in out
    content_type_r = asset_v1.ContentType.RELATIONSHIP
    quickstart_exportassets.export_assets_bigquery(PROJECT, dataset_id, 'assettable', content_type_r)
    (out_r, _) = capsys.readouterr()
    assert dataset_id in out_r