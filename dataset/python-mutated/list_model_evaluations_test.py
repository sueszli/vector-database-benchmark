import os
from google.api_core.retry import Retry
import list_model_evaluations
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
MODEL_ID = os.environ['ENTITY_EXTRACTION_MODEL_ID']

@Retry()
def test_list_model_evaluations(capsys):
    if False:
        while True:
            i = 10
    list_model_evaluations.list_model_evaluations(PROJECT_ID, MODEL_ID)
    (out, _) = capsys.readouterr()
    assert 'Model evaluation name: ' in out