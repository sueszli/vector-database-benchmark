import os
import uuid
from google.api_core import retry
import pytest
import fhir_stores
cloud_region = 'us-central1'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
service_account_json = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
dataset_id = f'test_dataset_{uuid.uuid4()}'
fhir_store_id = f'test_fhir_store-{uuid.uuid4()}'
test_fhir_store_id = f'test_fhir_store-{uuid.uuid4()}'
client = fhir_stores.get_client(service_account_json)

class OperationNotComplete(Exception):
    """Operation is not yet complete"""

@retry.Retry(predicate=retry.if_exception_type(OperationNotComplete))
def wait_for_operation(operation_name: str):
    if False:
        print('Hello World!')
    operation = client.projects().locations().datasets().operations().get(name=operation_name).execute()
    if not operation.get('done', False):
        raise OperationNotComplete(operation)

@pytest.fixture(scope='module')
def test_dataset():
    if False:
        for i in range(10):
            print('nop')
    operation = fhir_stores.create_dataset(service_account_json, project_id, cloud_region, dataset_id)
    wait_for_operation(operation['name'])
    yield
    fhir_stores.delete_dataset(service_account_json, project_id, cloud_region, dataset_id)

@pytest.fixture(scope='module')
def test_fhir_store():
    if False:
        print('Hello World!')
    resp = fhir_stores.create_fhir_store(service_account_json, project_id, cloud_region, dataset_id, test_fhir_store_id)
    yield resp
    fhir_stores.delete_fhir_store(service_account_json, project_id, cloud_region, dataset_id, test_fhir_store_id)

def test_create_delete_fhir_store(test_dataset, capsys):
    if False:
        while True:
            i = 10
    fhir_stores.create_fhir_store(service_account_json, project_id, cloud_region, dataset_id, fhir_store_id)
    fhir_stores.delete_fhir_store(service_account_json, project_id, cloud_region, dataset_id, fhir_store_id)
    (out, _) = capsys.readouterr()
    assert 'Created FHIR store' in out
    assert 'Deleted FHIR store' in out