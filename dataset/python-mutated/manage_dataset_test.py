import os
import backoff
from google.api_core.exceptions import RetryError, ServerError
import pytest
import manage_dataset
import testing_lib
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')

@pytest.mark.skip(reason='service is limited due to covid')
@pytest.fixture(scope='module')
def dataset():
    if False:
        i = 10
        return i + 15
    dataset = testing_lib.create_dataset(PROJECT_ID)
    yield dataset
    testing_lib.delete_dataset(dataset.name)

@pytest.fixture(scope='module')
def cleaner():
    if False:
        while True:
            i = 10
    try:
        testing_lib.delete_old_datasets(PROJECT_ID)
    except RetryError as e:
        print(f'delete_old_datasets failed: detail {e}')
    resource_names = []
    yield resource_names
    for resource_name in resource_names:
        testing_lib.delete_dataset(resource_name)

@pytest.mark.skip(reason='service is limited due to covid')
def test_create_dataset(cleaner, capsys):
    if False:
        i = 10
        return i + 15

    @backoff.on_exception(backoff.expo, ServerError, max_time=testing_lib.RETRY_DEADLINE)
    def run_sample():
        if False:
            for i in range(10):
                print('nop')
        return manage_dataset.create_dataset(PROJECT_ID)
    response = run_sample()
    cleaner.append(response.name)
    (out, _) = capsys.readouterr()
    assert 'The dataset resource name:' in out

@pytest.mark.skip(reason='service is limited due to covid')
def test_list_dataset(capsys, dataset):
    if False:
        while True:
            i = 10

    @backoff.on_exception(backoff.expo, ServerError, max_time=testing_lib.RETRY_DEADLINE)
    def run_sample():
        if False:
            return 10
        manage_dataset.list_datasets(PROJECT_ID)
    run_sample()
    (out, _) = capsys.readouterr()
    assert dataset.name in out

@pytest.mark.skip(reason='service is limited due to covid')
def test_get_dataset(capsys, dataset):
    if False:
        for i in range(10):
            print('nop')

    @backoff.on_exception(backoff.expo, ServerError, max_time=testing_lib.RETRY_DEADLINE)
    def run_sample():
        if False:
            while True:
                i = 10
        manage_dataset.get_dataset(dataset.name)
    run_sample()
    (out, _) = capsys.readouterr()
    assert 'The dataset resource name:' in out

@pytest.mark.skip(reason='service is limited due to covid')
def test_delete_dataset(capsys, dataset):
    if False:
        print('Hello World!')

    @backoff.on_exception(backoff.expo, ServerError, max_time=testing_lib.RETRY_DEADLINE)
    def run_sample():
        if False:
            print('Hello World!')
        manage_dataset.delete_dataset(dataset.name)
    run_sample()
    (out, _) = capsys.readouterr()
    assert 'Dataset deleted.' in out