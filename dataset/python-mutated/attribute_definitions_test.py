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
import attribute_definitions
location = 'us-central1'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
dataset_id = f'test-dataset-{uuid.uuid4()}'
consent_store_id = f'test-consent-store-{uuid.uuid4()}'
resource_attribute_definition_id = 'test_resource_attribute_definition_id_{}'.format(uuid.uuid4().hex[:5])
request_attribute_definition_id = 'test_request_attribute_definition_id_{}'.format(uuid.uuid4().hex[:5])
description = 'whether the data is de-identifiable'

@pytest.fixture(scope='module')
def test_dataset():
    if False:
        for i in range(10):
            print('nop')

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def create():
        if False:
            while True:
                i = 10
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
            if err.resp.status == 403:
                print(f'Got exception {err.resp.status} while deleting dataset')
            else:
                raise
    clean_up()

@pytest.fixture(scope='module')
def test_consent_store():
    if False:
        print('Hello World!')

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def create():
        if False:
            for i in range(10):
                print('nop')
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
            while True:
                i = 10
        try:
            consent_stores.delete_consent_store(project_id, location, dataset_id, consent_store_id)
        except HttpError as err:
            if err.resp.status == 404 or err.resp.status == 403:
                print('Got exception {} while deleting consent store'.format(err.resp.status))
            else:
                raise
    clean_up()

def test_CRUD_resource_attribute_definition(test_dataset: str, test_consent_store: str, capsys):
    if False:
        return 10

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def create():
        if False:
            while True:
                i = 10
        attribute_definitions.create_resource_attribute_definition(project_id, location, dataset_id, consent_store_id, resource_attribute_definition_id)
    create()
    attribute_definitions.get_attribute_definition(project_id, location, dataset_id, consent_store_id, resource_attribute_definition_id)
    attribute_definitions.list_attribute_definitions(project_id, location, dataset_id, consent_store_id)
    attribute_definitions.patch_attribute_definition(project_id, location, dataset_id, consent_store_id, resource_attribute_definition_id, description)
    attribute_definitions.delete_attribute_definition(project_id, location, dataset_id, consent_store_id, resource_attribute_definition_id)
    (out, _) = capsys.readouterr()
    assert 'Created RESOURCE attribute definition' in out
    assert 'Got attribute definition' in out
    assert 'name' in out
    assert 'Patched attribute definition' in out
    assert 'Deleted attribute definition' in out

def test_CRUD_request_attribute_definition(test_dataset: str, test_consent_store: str, capsys):
    if False:
        while True:
            i = 10

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def create():
        if False:
            print('Hello World!')
        attribute_definitions.create_request_attribute_definition(project_id, location, dataset_id, consent_store_id, request_attribute_definition_id)
    create()
    attribute_definitions.get_attribute_definition(project_id, location, dataset_id, consent_store_id, request_attribute_definition_id)
    attribute_definitions.list_attribute_definitions(project_id, location, dataset_id, consent_store_id)
    attribute_definitions.patch_attribute_definition(project_id, location, dataset_id, consent_store_id, request_attribute_definition_id, description)
    attribute_definitions.delete_attribute_definition(project_id, location, dataset_id, consent_store_id, request_attribute_definition_id)
    (out, _) = capsys.readouterr()
    assert 'Created REQUEST attribute definition' in out
    assert 'Got attribute definition' in out
    assert 'name' in out
    assert 'Patched attribute definition' in out
    assert 'Deleted attribute definition' in out