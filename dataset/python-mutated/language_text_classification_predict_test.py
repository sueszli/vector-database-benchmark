import os
from google.api_core.retry import Retry
from google.cloud import automl
import pytest
import language_text_classification_predict
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
MODEL_ID = os.environ['TEXT_CLASSIFICATION_MODEL_ID']

@pytest.fixture(scope='function')
def verify_model_state():
    if False:
        return 10
    client = automl.AutoMlClient()
    model_full_id = client.model_path(PROJECT_ID, 'us-central1', MODEL_ID)
    model = client.get_model(name=model_full_id)
    if model.deployment_state == automl.Model.DeploymentState.UNDEPLOYED:
        response = client.deploy_model(name=model_full_id)
        response.result()

@Retry()
def test_text_classification_predict(capsys, verify_model_state):
    if False:
        i = 10
        return i + 15
    verify_model_state
    text = 'Fruit and nut flavour'
    language_text_classification_predict.predict(PROJECT_ID, MODEL_ID, text)
    (out, _) = capsys.readouterr()
    assert 'Predicted class name: ' in out