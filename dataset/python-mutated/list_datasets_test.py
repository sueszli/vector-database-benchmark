import os
from google.api_core.retry import Retry
import list_datasets
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
DATASET_ID = os.environ['ENTITY_EXTRACTION_DATASET_ID']

@Retry()
def test_list_dataset(capsys):
    if False:
        for i in range(10):
            print('nop')
    list_datasets.list_datasets(PROJECT_ID)
    (out, _) = capsys.readouterr()
    assert f'Dataset id: {DATASET_ID}' in out