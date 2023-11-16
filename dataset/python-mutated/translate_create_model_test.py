import os
from google.api_core.retry import Retry
import translate_create_model
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
DATASET_ID = 'TRL00000000000000000'

@Retry()
def test_translate_create_model(capsys):
    if False:
        print('Hello World!')
    try:
        translate_create_model.create_model(PROJECT_ID, DATASET_ID, 'translate_test_create_model')
        (out, _) = capsys.readouterr()
        assert 'Dataset does not exist.' in out
    except Exception as e:
        assert 'Dataset does not exist.' in e.message