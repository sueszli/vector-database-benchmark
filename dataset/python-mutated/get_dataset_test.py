import os
from google.api_core.retry import Retry
import get_dataset
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
DATASET_ID = os.environ['ENTITY_EXTRACTION_DATASET_ID']

@Retry()
def test_get_dataset(capsys):
    if False:
        for i in range(10):
            print('nop')
    get_dataset.get_dataset(PROJECT_ID, DATASET_ID)
    (out, _) = capsys.readouterr()
    assert 'Dataset name: ' in out