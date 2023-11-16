import os
import backoff
from google.api_core.exceptions import ServerError
from google.cloud import datalabeling
import pytest
import import_data
import testing_lib
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
INPUT_GCS_URI = 'gs://cloud-samples-data/datalabeling/image/image_dataset.csv'

@pytest.fixture(scope='module')
def dataset():
    if False:
        i = 10
        return i + 15
    dataset = testing_lib.create_dataset(PROJECT_ID)
    yield dataset
    testing_lib.delete_dataset(dataset.name)

@pytest.mark.skip(reason='service is limited due to covid')
def test_import_data(capsys, dataset):
    if False:
        for i in range(10):
            print('nop')

    @backoff.on_exception(backoff.expo, ServerError, max_time=testing_lib.RETRY_DEADLINE)
    def run_sample():
        if False:
            i = 10
            return i + 15
        import_data.import_data(dataset.name, datalabeling.DataType.IMAGE, INPUT_GCS_URI)
    run_sample()
    (out, _) = capsys.readouterr()
    assert 'Dataset resource name: ' in out