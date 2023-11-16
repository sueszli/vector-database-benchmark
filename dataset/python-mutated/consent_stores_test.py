import os
import sys
import uuid
import backoff
from googleapiclient.errors import HttpError
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
from create_dataset import create_dataset
from delete_dataset import delete_dataset
import consent_stores
location = 'us-central1'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
dataset_id = f'test_dataset-{uuid.uuid4()}'
consent_store_id = f'test_consent_store_{uuid.uuid4()}'
default_consent_ttl = '86400s'

@pytest.fixture(scope='module')
def test_dataset():
    if False:
        return 10

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def create():
        if False:
            for i in range(10):
                print('nop')
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
            i = 10
            return i + 15
        try:
            delete_dataset(project_id, location, dataset_id)
        except HttpError as err:
            if err.resp.status == 403:
                print(f'Got exception {err.resp.status} while deleting dataset')
            else:
                raise
    clean_up()

@pytest.fixture(scope='module')
def test_consent_store():
    if False:
        while True:
            i = 10

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def create():
        if False:
            i = 10
            return i + 15
        try:
            consent_stores.create_consent_store(project_id, location, dataset_id, consent_store_id)
        except HttpError as err:
            if err.resp.status == 409:
                print('Got exception {} while creating consent store'.format(err.resp.status))
            else:
                raise
    create()
    yield

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def clean_up():
        if False:
            i = 10
            return i + 15
        try:
            consent_stores.delete_consent_store(project_id, location, dataset_id, consent_store_id)
        except HttpError as err:
            if err.resp.status == 404 or err.resp.status == 403:
                print('Got exception {} while deleting consent store'.format(err.resp.status))
            else:
                raise
    clean_up()

@pytest.fixture(scope='module')
def crud_consent_store_id():
    if False:
        return 10
    yield consent_store_id

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def clean_up():
        if False:
            return 10
        try:
            consent_stores.delete_consent_store(project_id, location, dataset_id, consent_store_id)
        except HttpError as err:
            if err.resp.status == 404 or err.resp.status == 403:
                print('Got exception {} while deleting consent store'.format(err.resp.status))
            else:
                raise
    clean_up()

def test_CRUD_consent_store(test_dataset: str, crud_consent_store_id: str, capsys):
    if False:
        for i in range(10):
            print('nop')

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def create():
        if False:
            return 10
        consent_stores.create_consent_store(project_id, location, dataset_id, crud_consent_store_id)
    create()
    consent_stores.get_consent_store(project_id, location, dataset_id, crud_consent_store_id)
    consent_stores.list_consent_stores(project_id, location, dataset_id)
    consent_stores.delete_consent_store(project_id, location, dataset_id, crud_consent_store_id)
    (out, _) = capsys.readouterr()
    assert 'Created consent store' in out
    assert 'name' in out
    assert 'consentStores' in out
    assert 'Deleted consent store' in out

def test_patch_consent_store(test_dataset: str, test_consent_store: str, capsys):
    if False:
        print('Hello World!')
    consent_stores.patch_consent_store(project_id, location, dataset_id, consent_store_id, default_consent_ttl)
    (out, _) = capsys.readouterr()
    assert 'Patched consent store' in out

def test_get_set_consent_store_iam_policy(test_dataset: str, test_consent_store: str, capsys):
    if False:
        print('Hello World!')
    get_response = consent_stores.get_consent_store_iam_policy(project_id, location, dataset_id, consent_store_id)
    set_response = consent_stores.set_consent_store_iam_policy(project_id, location, dataset_id, consent_store_id, 'serviceAccount:python-docs-samples-tests@appspot.gserviceaccount.com', 'roles/viewer')
    (out, _) = capsys.readouterr()
    assert 'etag' in get_response
    assert 'bindings' in set_response
    assert len(set_response['bindings']) == 1
    assert 'python-docs-samples-tests' in str(set_response['bindings'])
    assert 'roles/viewer' in str(set_response['bindings'])