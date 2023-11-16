import os
from google.api_core.retry import Retry
from google.cloud import automl
import pytest
import translate_predict
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
MODEL_ID = os.environ['TRANSLATION_MODEL_ID']

@pytest.fixture(scope='function', autouse=True)
def setup():
    if False:
        for i in range(10):
            print('nop')
    client = automl.AutoMlClient()
    model_full_id = client.model_path(PROJECT_ID, 'us-central1', MODEL_ID)
    model = client.get_model(name=model_full_id)
    if model.deployment_state == automl.Model.DeploymentState.UNDEPLOYED:
        response = client.deploy_model(name=model_full_id)
        response.result()

@Retry()
def test_translate_predict(capsys):
    if False:
        while True:
            i = 10
    translate_predict.predict(PROJECT_ID, MODEL_ID, 'resources/input.txt')
    (out, _) = capsys.readouterr()
    assert 'Translated content: ' in out