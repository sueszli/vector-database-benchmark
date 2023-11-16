import datetime
import os
from google.api_core.retry import Retry
from google.cloud import automl
import pytest
import delete_dataset
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
BUCKET_ID = f'{PROJECT_ID}-lcm'

@pytest.fixture(scope='function')
def dataset_id():
    if False:
        for i in range(10):
            print('nop')
    client = automl.AutoMlClient()
    project_location = f'projects/{PROJECT_ID}/locations/us-central1'
    display_name = 'test_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    metadata = automl.TextExtractionDatasetMetadata()
    dataset = automl.Dataset(display_name=display_name, text_extraction_dataset_metadata=metadata)
    response = client.create_dataset(parent=project_location, dataset=dataset)
    dataset_id = response.result().name.split('/')[-1]
    yield dataset_id

@Retry()
def test_delete_dataset(capsys, dataset_id):
    if False:
        for i in range(10):
            print('nop')
    delete_dataset.delete_dataset(PROJECT_ID, dataset_id)
    (out, _) = capsys.readouterr()
    assert 'Dataset deleted.' in out