import os
import sys
import uuid
import backoff
from google.cloud import pubsub_v1
from googleapiclient.errors import HttpError
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
from create_dataset import create_dataset
from delete_dataset import delete_dataset
import dicom_stores
location = 'us-central1'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
dataset_id = f'test_dataset-{uuid.uuid4()}'
dicom_store_id = f'test_dicom_store_{uuid.uuid4()}'
pubsub_topic = f'test_pubsub_topic_{uuid.uuid4()}'
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')
bucket = os.environ['CLOUD_STORAGE_BUCKET']
dcm_file_name = 'dicom_00000001_000.dcm'
content_uri = bucket + '/' + dcm_file_name
dcm_file = os.path.join(RESOURCES, dcm_file_name)

@pytest.fixture(scope='module')
def test_dataset():
    if False:
        i = 10
        return i + 15

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
            return 10
        try:
            delete_dataset(project_id, location, dataset_id)
        except HttpError as err:
            if err.resp.status == 403:
                print(f'Got exception {err.resp.status} while deleting dataset')
            else:
                raise
    clean_up()

@pytest.fixture(scope='module')
def test_dicom_store():
    if False:
        print('Hello World!')

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def create():
        if False:
            print('Hello World!')
        try:
            dicom_stores.create_dicom_store(project_id, location, dataset_id, dicom_store_id)
        except HttpError as err:
            if err.resp.status == 409:
                print('Got exception {} while creating DICOM store'.format(err.resp.status))
            else:
                raise
    create()
    yield

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def clean_up():
        if False:
            print('Hello World!')
        try:
            dicom_stores.delete_dicom_store(project_id, location, dataset_id, dicom_store_id)
        except HttpError as err:
            if err.resp.status == 404 or err.resp.status == 403:
                print('Got exception {} while deleting DICOM store'.format(err.resp.status))
            else:
                raise
    clean_up()

@pytest.fixture(scope='module')
def crud_dicom_store_id():
    if False:
        while True:
            i = 10
    yield dicom_store_id

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def clean_up():
        if False:
            return 10
        try:
            dicom_stores.delete_dicom_store(project_id, location, dataset_id, dicom_store_id)
        except HttpError as err:
            if err.resp.status == 404 or err.resp.status == 403:
                print('Got exception {} while deleting DICOM store'.format(err.resp.status))
            else:
                raise
    clean_up()

@pytest.fixture(scope='module')
def test_pubsub_topic():
    if False:
        i = 10
        return i + 15
    pubsub_client = pubsub_v1.PublisherClient()
    topic_path = pubsub_client.topic_path(project_id, pubsub_topic)
    pubsub_client.create_topic(request={'name': topic_path})
    yield pubsub_topic
    pubsub_client.delete_topic(request={'topic': topic_path})

def test_CRUD_dicom_store(test_dataset, crud_dicom_store_id, capsys):
    if False:
        print('Hello World!')

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def create():
        if False:
            for i in range(10):
                print('nop')
        dicom_stores.create_dicom_store(project_id, location, dataset_id, crud_dicom_store_id)
    create()
    dicom_stores.get_dicom_store(project_id, location, dataset_id, crud_dicom_store_id)
    dicom_stores.list_dicom_stores(project_id, location, dataset_id)
    dicom_stores.delete_dicom_store(project_id, location, dataset_id, crud_dicom_store_id)
    (out, _) = capsys.readouterr()
    assert 'Created DICOM store' in out
    assert 'name' in out
    assert 'dicomStores' in out
    assert 'Deleted DICOM store' in out

def test_patch_dicom_store(test_dataset, test_dicom_store, test_pubsub_topic, capsys):
    if False:
        print('Hello World!')
    dicom_stores.patch_dicom_store(project_id, location, dataset_id, dicom_store_id, test_pubsub_topic)
    (out, _) = capsys.readouterr()
    assert 'Patched DICOM store' in out

def test_import_dicom_instance(test_dataset, test_dicom_store, capsys):
    if False:
        while True:
            i = 10
    dicom_stores.import_dicom_instance(project_id, location, dataset_id, dicom_store_id, content_uri)
    (out, _) = capsys.readouterr()
    assert 'Imported DICOM instance' in out

def test_export_dicom_instance(test_dataset, test_dicom_store, capsys):
    if False:
        print('Hello World!')
    dicom_stores.export_dicom_instance(project_id, location, dataset_id, dicom_store_id, bucket)
    (out, _) = capsys.readouterr()
    assert 'Exported DICOM instance' in out

def test_get_set_dicom_store_iam_policy(test_dataset, test_dicom_store, capsys):
    if False:
        return 10
    get_response = dicom_stores.get_dicom_store_iam_policy(project_id, location, dataset_id, dicom_store_id)
    set_response = dicom_stores.set_dicom_store_iam_policy(project_id, location, dataset_id, dicom_store_id, 'serviceAccount:python-docs-samples-tests@appspot.gserviceaccount.com', 'roles/viewer')
    (out, _) = capsys.readouterr()
    assert 'etag' in get_response
    assert 'bindings' in set_response
    assert len(set_response['bindings']) == 1
    assert 'python-docs-samples-tests' in str(set_response['bindings'])
    assert 'roles/viewer' in str(set_response['bindings'])