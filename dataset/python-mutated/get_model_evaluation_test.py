import os
from google.api_core.retry import Retry
from google.cloud import automl
import pytest
import get_model_evaluation
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
MODEL_ID = os.environ['ENTITY_EXTRACTION_MODEL_ID']

@Retry()
@pytest.fixture(scope='function')
def model_evaluation_id():
    if False:
        while True:
            i = 10
    client = automl.AutoMlClient()
    model_full_id = client.model_path(PROJECT_ID, 'us-central1', MODEL_ID)
    evaluation = None
    for e in client.list_model_evaluations(parent=model_full_id, filter=''):
        evaluation = e
        break
    model_evaluation_id = evaluation.name.split(f'{MODEL_ID}/modelEvaluations/')[1].split('\n')[0]
    yield model_evaluation_id

@Retry()
def test_get_model_evaluation(capsys, model_evaluation_id):
    if False:
        for i in range(10):
            print('nop')
    get_model_evaluation.get_model_evaluation(PROJECT_ID, MODEL_ID, model_evaluation_id)
    (out, _) = capsys.readouterr()
    assert 'Model evaluation name: ' in out