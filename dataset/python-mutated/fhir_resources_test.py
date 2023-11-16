import os
import uuid
import backoff
from google.api_core import retry
import pytest
from requests.exceptions import HTTPError
import fhir_stores
import fhir_resources
cloud_region = 'us-central1'
base_url = 'https://healthcare.googleapis.com/v1beta1'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
service_account_json = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
dataset_id = f'test_dataset_{uuid.uuid4()}'
fhir_store_id = f'test_fhir_store-{uuid.uuid4()}'
resource_type = 'Patient'
client = fhir_stores.get_client(service_account_json)
BACKOFF_MAX_TIME = 750

def fatal_code(e):
    if False:
        print('Hello World!')
    return 400 <= e.response.status_code < 500

class OperationNotComplete(Exception):
    """Operation is not yet complete"""

@retry.Retry(predicate=retry.if_exception_type(OperationNotComplete))
def wait_for_operation(operation_name: str):
    if False:
        return 10
    operation = client.projects().locations().datasets().operations().get(name=operation_name).execute()
    if not operation.get('done', False):
        raise OperationNotComplete(operation)

@pytest.fixture(scope='module')
def test_dataset():
    if False:
        i = 10
        return i + 15
    operation = fhir_stores.create_dataset(service_account_json, project_id, cloud_region, dataset_id)
    wait_for_operation(operation['name'])
    yield
    fhir_stores.delete_dataset(service_account_json, project_id, cloud_region, dataset_id)

@pytest.fixture(scope='module')
def test_fhir_store():
    if False:
        return 10
    fhir_store = fhir_stores.create_fhir_store(service_account_json, project_id, cloud_region, dataset_id, fhir_store_id)
    yield fhir_store
    fhir_stores.delete_fhir_store(service_account_json, project_id, cloud_region, dataset_id, fhir_store_id)

@pytest.fixture(scope='module')
def test_patient():
    if False:
        i = 10
        return i + 15
    patient_response = fhir_resources.create_patient(service_account_json, base_url, project_id, cloud_region, dataset_id, fhir_store_id)
    patient_resource_id = patient_response.json()['id']
    yield patient_resource_id
    fhir_resources.delete_resource(service_account_json, base_url, project_id, cloud_region, dataset_id, fhir_store_id, resource_type, patient_resource_id)

def test_create_patient(test_dataset, test_fhir_store, capsys):
    if False:
        for i in range(10):
            print('nop')
    fhir_resources.create_patient(service_account_json, base_url, project_id, cloud_region, dataset_id, fhir_store_id)
    (out, _) = capsys.readouterr()
    print(out)
    assert 'Created Patient' in out

@pytest.mark.skip(reason='flaky test sometimes returns 403 errors, need to investigate')
def test_conditional_patch_resource(test_dataset, test_fhir_store, test_patient, capsys):
    if False:
        for i in range(10):
            print('nop')
    encounter_response = fhir_resources.create_encounter(service_account_json, base_url, project_id, cloud_region, dataset_id, fhir_store_id, test_patient)
    encounter_resource_id = encounter_response.json()['id']
    observation_response = fhir_resources.create_observation(service_account_json, base_url, project_id, cloud_region, dataset_id, fhir_store_id, test_patient, encounter_resource_id)
    observation_resource_id = observation_response.json()['id']
    fhir_resources.conditional_patch_resource(service_account_json, base_url, project_id, cloud_region, dataset_id, fhir_store_id)
    fhir_resources.delete_resource(service_account_json, base_url, project_id, cloud_region, dataset_id, fhir_store_id, 'Observation', observation_resource_id)
    (out, _) = capsys.readouterr()
    print(out)
    assert 'Conditionally patched' in out

@pytest.mark.skip(reason='flaky test sometimes returns 412 errors, need to investigate')
def test_conditional_update_resource(test_dataset, test_fhir_store, test_patient, capsys):
    if False:
        i = 10
        return i + 15
    encounter_response = fhir_resources.create_encounter(service_account_json, base_url, project_id, cloud_region, dataset_id, fhir_store_id, test_patient)
    encounter_resource_id = encounter_response.json()['id']
    observation_response = fhir_resources.create_observation(service_account_json, base_url, project_id, cloud_region, dataset_id, fhir_store_id, test_patient, encounter_resource_id)
    observation_resource_id = observation_response.json()['id']
    fhir_resources.conditional_update_resource(service_account_json, base_url, project_id, cloud_region, dataset_id, fhir_store_id, test_patient, encounter_resource_id)
    fhir_resources.delete_resource(service_account_json, base_url, project_id, cloud_region, dataset_id, fhir_store_id, 'Observation', observation_resource_id)
    (out, _) = capsys.readouterr()
    assert 'Conditionally updated' in out

def test_conditional_delete_resource(test_dataset, test_fhir_store, test_patient, capsys):
    if False:
        for i in range(10):
            print('nop')

    @backoff.on_exception(backoff.expo, HTTPError, max_time=BACKOFF_MAX_TIME, giveup=fatal_code)
    def create_encounter():
        if False:
            i = 10
            return i + 15
        encounter_response = fhir_resources.create_encounter(service_account_json, base_url, project_id, cloud_region, dataset_id, fhir_store_id, test_patient)
        return encounter_response.json()['id']
    encounter_resource_id = create_encounter()

    @backoff.on_exception(backoff.expo, HTTPError, max_time=BACKOFF_MAX_TIME, giveup=fatal_code)
    def create_observation():
        if False:
            for i in range(10):
                print('nop')
        fhir_resources.create_observation(service_account_json, base_url, project_id, cloud_region, dataset_id, fhir_store_id, test_patient, encounter_resource_id)
    create_observation()

    @backoff.on_exception(backoff.expo, HTTPError, max_time=BACKOFF_MAX_TIME, giveup=fatal_code)
    def conditional_delete_resource():
        if False:
            print('Hello World!')
        fhir_resources.conditional_delete_resource(service_account_json, base_url, project_id, cloud_region, dataset_id, fhir_store_id)
    conditional_delete_resource()
    (out, _) = capsys.readouterr()
    print(out)
    assert 'Conditionally deleted' in out

def test_delete_patient(test_dataset, test_fhir_store, test_patient, capsys):
    if False:
        i = 10
        return i + 15
    fhir_resources.delete_resource(service_account_json, base_url, project_id, cloud_region, dataset_id, fhir_store_id, resource_type, test_patient)
    (out, _) = capsys.readouterr()
    print(out)
    assert 'Deleted Patient resource' in out