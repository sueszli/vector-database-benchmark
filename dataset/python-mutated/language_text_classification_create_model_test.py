import os
from google.api_core.retry import Retry
import language_text_classification_create_model
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
DATASET_ID = 'TCN00000000000000000000'

@Retry()
def test_text_classification_create_model(capsys):
    if False:
        i = 10
        return i + 15
    try:
        language_text_classification_create_model.create_model(PROJECT_ID, DATASET_ID, 'lang_text_test_create_model')
        (out, _) = capsys.readouterr()
        assert 'Dataset does not exist.' in out
    except Exception as e:
        assert 'Dataset does not exist.' in e.message