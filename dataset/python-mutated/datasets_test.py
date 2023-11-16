import os
import uuid
import backoff
from google.api_core.exceptions import RetryError
from googleapiclient.errors import HttpError
import pytest
from retrying import retry
from create_dataset import create_dataset
from deidentify_dataset import deidentify_dataset
from delete_dataset import delete_dataset
from get_dataset import get_dataset
from get_dataset_iam_policy import get_dataset_iam_policy
from list_datasets import list_datasets
from patch_dataset import patch_dataset
from set_dataset_iam_policy import set_dataset_iam_policy
location = 'us-central1'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
dataset_id = f'1test-dataset-{uuid.uuid4()}'
tmp_dataset_id = f'1tmp-test-dataset-{uuid.uuid4()}'
destination_dataset_id = f'1test-destination-dataset-{uuid.uuid4()}'
time_zone = 'UTC'
WAIT_EXPONENTIAL_MULTIPLIER = 1000
WAIT_EXPONENTIAL_MAX = 120000
STOP_MAX_ATTEMPT_NUMBER = 20
MAX_BACKOFF_TIME = 750

def is_retryable_exception(exception):
    if False:
        for i in range(10):
            print('nop')
    return isinstance(exception, (HttpError, RetryError))

@pytest.fixture(scope='module')
def test_dataset():
    if False:
        while True:
            i = 10
    'Yields a dataset for other tests to use.'

    @retry(wait_exponential_multiplier=WAIT_EXPONENTIAL_MULTIPLIER, wait_exponential_max=WAIT_EXPONENTIAL_MAX, stop_max_attempt_number=STOP_MAX_ATTEMPT_NUMBER, retry_on_exception=is_retryable_exception)
    def create():
        if False:
            i = 10
            return i + 15
        try:
            create_dataset(project_id, location, dataset_id)
        except HttpError as err:
            if err.resp.status == 409:
                print(f'Got {err.resp.status} error while creating dataset. Dataset already exists.')
            else:
                raise err
        except TimeoutError as err:
            raise err
    create()
    yield
    clean_up_dataset(dataset_id)

def clean_up_dataset(dataset_id):
    if False:
        i = 10
        return i + 15

    @retry(wait_exponential_multiplier=WAIT_EXPONENTIAL_MULTIPLIER, wait_exponential_max=WAIT_EXPONENTIAL_MAX, stop_max_attempt_number=STOP_MAX_ATTEMPT_NUMBER, retry_on_exception=is_retryable_exception)
    def clean_up():
        if False:
            i = 10
            return i + 15
        try:
            delete_dataset(project_id, location, dataset_id)
        except HttpError as err:
            if err.resp.status == 404 or err.resp.status == 403:
                print(f'Got exception {err.resp.status} while deleting dataset. Dataset was likely already deleted.')
            else:
                raise
    clean_up()

@pytest.fixture(scope='module')
def dest_dataset_id():
    if False:
        for i in range(10):
            print('nop')
    yield destination_dataset_id
    clean_up_dataset(destination_dataset_id)

@retry(wait_exponential_multiplier=WAIT_EXPONENTIAL_MULTIPLIER, wait_exponential_max=WAIT_EXPONENTIAL_MAX, stop_max_attempt_number=STOP_MAX_ATTEMPT_NUMBER, retry_on_exception=is_retryable_exception)
def test_create_dataset(capsys):
    if False:
        while True:
            i = 10
    create_dataset(project_id, location, tmp_dataset_id)
    (out, _) = capsys.readouterr()
    assert tmp_dataset_id in out
    clean_up_dataset(tmp_dataset_id)

@backoff.on_exception(backoff.expo, HttpError, max_tries=10)
def test_get_dataset(capsys, test_dataset):
    if False:
        i = 10
        return i + 15
    get_dataset(project_id, location, dataset_id)
    (out, _) = capsys.readouterr()
    assert dataset_id in out

@backoff.on_exception(backoff.expo, HttpError, max_tries=10)
def test_list_datasets(capsys, test_dataset):
    if False:
        return 10
    list_datasets(project_id, location)
    (out, _) = capsys.readouterr()
    assert 'Dataset' in out

@backoff.on_exception(backoff.expo, HttpError, max_tries=10)
def test_patch_dataset(capsys, test_dataset):
    if False:
        return 10
    patch_dataset(project_id, location, dataset_id, time_zone)
    (out, _) = capsys.readouterr()
    assert time_zone in out

@backoff.on_exception(backoff.expo, HttpError, max_tries=10)
def test_deidentify_dataset(capsys, test_dataset, dest_dataset_id):
    if False:
        i = 10
        return i + 15
    deidentify_dataset(project_id, location, dataset_id, dest_dataset_id)
    (out, _) = capsys.readouterr()
    assert dest_dataset_id in out

@backoff.on_exception(backoff.expo, HttpError, max_tries=10)
def test_get_set_dataset_iam_policy(capsys, test_dataset):
    if False:
        return 10
    get_response = get_dataset_iam_policy(project_id, location, dataset_id)
    set_response = set_dataset_iam_policy(project_id, location, dataset_id, 'serviceAccount:python-docs-samples-tests@appspot.gserviceaccount.com', 'roles/viewer')
    (out, _) = capsys.readouterr()
    assert 'etag' in get_response
    assert 'bindings' in set_response
    assert len(set_response['bindings']) == 1
    assert 'python-docs-samples-tests' in str(set_response['bindings'])
    assert 'roles/viewer' in str(set_response['bindings'])

@backoff.on_exception(backoff.expo, HttpError, max_tries=10)
def test_delete_dataset(capsys, test_dataset):
    if False:
        return 10
    delete_dataset(project_id, location, dataset_id)
    (out, _) = capsys.readouterr()
    assert 'Deleted' in out