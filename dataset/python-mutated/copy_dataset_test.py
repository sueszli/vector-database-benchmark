import datetime
import uuid
import pytest
from . import copy_dataset

def temp_suffix():
    if False:
        for i in range(10):
            print('nop')
    now = datetime.datetime.now()
    return f"{now.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

@pytest.fixture(scope='module')
def destination_dataset_id(bigquery_client, project_id):
    if False:
        return 10
    dataset_id = f'bqdts_dest_{temp_suffix()}'
    bigquery_client.create_dataset(f'{project_id}.{dataset_id}')
    yield dataset_id
    bigquery_client.delete_dataset(dataset_id, delete_contents=True)

@pytest.fixture(scope='module')
def source_dataset_id(bigquery_client, project_id):
    if False:
        i = 10
        return i + 15
    dataset_id = f'bqdts_src_{temp_suffix()}'
    bigquery_client.create_dataset(f'{project_id}.{dataset_id}')
    yield dataset_id
    bigquery_client.delete_dataset(dataset_id, delete_contents=True)

def test_copy_dataset(capsys, transfer_client, project_id, destination_dataset_id, source_dataset_id, to_delete_configs):
    if False:
        print('Hello World!')
    assert transfer_client is not None
    transfer_config = copy_dataset.copy_dataset({'destination_project_id': project_id, 'destination_dataset_id': destination_dataset_id, 'source_project_id': project_id, 'source_dataset_id': source_dataset_id})
    to_delete_configs.append(transfer_config.name)
    (out, _) = capsys.readouterr()
    assert transfer_config.name in out