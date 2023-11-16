import os
import sys
import uuid
import backoff
from googleapiclient.errors import HttpError
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
from create_dataset import create_dataset
from delete_dataset import delete_dataset
import hl7v2_stores
location = 'us-central1'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
dataset_id = f'test_dataset_{uuid.uuid4()}'
hl7v2_store_id = f'test_hl7v2_store-{uuid.uuid4()}'

def retry_if_server_exception(exception):
    if False:
        return 10
    return isinstance(exception, HttpError)

@pytest.fixture(scope='module')
def test_dataset():
    if False:
        print('Hello World!')

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def create():
        if False:
            i = 10
            return i + 15
        try:
            create_dataset(project_id, location, dataset_id)
        except HttpError as err:
            if err.resp.status == 409:
                print(f'Got exception {err.resp.status} while creating dataset')
            else:
                raise
    create()
    yield

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def clean_up():
        if False:
            print('Hello World!')
        try:
            delete_dataset(project_id, location, dataset_id)
        except HttpError as err:
            if err.resp.status == 404 or err.resp.status == 403:
                print(f'Got exception {err.resp.status} while deleting dataset')
            else:
                raise
    clean_up()

@pytest.fixture(scope='module')
def test_hl7v2_store():
    if False:
        i = 10
        return i + 15

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def create():
        if False:
            i = 10
            return i + 15
        try:
            hl7v2_stores.create_hl7v2_store(project_id, location, dataset_id, hl7v2_store_id)
        except HttpError as err:
            if err.resp.status == 409:
                print('Got exception {} while creating HL7v2 store'.format(err.resp.status))
            else:
                raise
    create()
    yield

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def clean_up():
        if False:
            print('Hello World!')
        try:
            hl7v2_stores.delete_hl7v2_store(project_id, location, dataset_id, hl7v2_store_id)
        except HttpError as err:
            if err.resp.status == 404 or err.resp.status == 403:
                print('Got exception {} while deleting HL7v2 store'.format(err.resp.status))
            else:
                raise
    clean_up()

@pytest.fixture(scope='module')
def crud_hl7v2_store_id():
    if False:
        while True:
            i = 10
    yield hl7v2_store_id

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def clean_up():
        if False:
            return 10
        try:
            hl7v2_stores.delete_hl7v2_store(project_id, location, dataset_id, hl7v2_store_id)
        except HttpError as err:
            if err.resp.status == 404 or err.resp.status == 403:
                print('Got exception {} while deleting HL7v2 store'.format(err.resp.status))
            else:
                raise
    clean_up()

def test_CRUD_hl7v2_store(test_dataset, crud_hl7v2_store_id, capsys):
    if False:
        while True:
            i = 10

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def create():
        if False:
            print('Hello World!')
        hl7v2_stores.create_hl7v2_store(project_id, location, dataset_id, hl7v2_store_id)
    create()
    hl7v2_stores.get_hl7v2_store(project_id, location, dataset_id, hl7v2_store_id)
    hl7v2_stores.list_hl7v2_stores(project_id, location, dataset_id)
    hl7v2_stores.delete_hl7v2_store(project_id, location, dataset_id, hl7v2_store_id)
    (out, _) = capsys.readouterr()
    assert 'Created HL7v2 store' in out
    assert 'Name' in out
    assert 'hl7V2Stores' in out
    assert 'Deleted HL7v2 store' in out

def test_patch_hl7v2_store(test_dataset, test_hl7v2_store, capsys):
    if False:
        for i in range(10):
            print('nop')
    hl7v2_stores.patch_hl7v2_store(project_id, location, dataset_id, hl7v2_store_id)
    (out, _) = capsys.readouterr()
    assert 'Patched HL7v2 store' in out

def test_get_set_hl7v2_store_iam_policy(test_dataset, test_hl7v2_store, capsys):
    if False:
        print('Hello World!')
    get_response = hl7v2_stores.get_hl7v2_store_iam_policy(project_id, location, dataset_id, hl7v2_store_id)
    set_response = hl7v2_stores.set_hl7v2_store_iam_policy(project_id, location, dataset_id, hl7v2_store_id, 'serviceAccount:python-docs-samples-tests@appspot.gserviceaccount.com', 'roles/viewer')
    (out, _) = capsys.readouterr()
    assert 'etag' in get_response
    assert 'bindings' in set_response
    assert len(set_response['bindings']) == 1
    assert 'python-docs-samples-tests' in str(set_response['bindings'])
    assert 'roles/viewer' in str(set_response['bindings'])