import os
from google.api_core.retry import Retry
import get_model
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
MODEL_ID = os.environ['ENTITY_EXTRACTION_MODEL_ID']

@Retry()
def test_get_model(capsys):
    if False:
        while True:
            i = 10
    get_model.get_model(PROJECT_ID, MODEL_ID)
    (out, _) = capsys.readouterr()
    assert 'Model id: ' in out