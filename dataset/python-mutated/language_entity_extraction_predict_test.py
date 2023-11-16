import os
import backoff
from google.api_core.exceptions import InternalServerError
from google.api_core.exceptions import ServiceUnavailable
from google.cloud import automl
import pytest
import language_entity_extraction_predict
PROJECT_ID = os.environ['AUTOML_PROJECT_ID']
MODEL_ID = os.environ['ENTITY_EXTRACTION_MODEL_ID']

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

@backoff.on_exception(backoff.expo, (InternalServerError, ServiceUnavailable), max_tries=3)
def test_predict(capsys, verify_model_state):
    if False:
        while True:
            i = 10
    verify_model_state
    text = 'Constitutional mutations in the WT1 gene in patients with Denys-Drash syndrome.'
    language_entity_extraction_predict.predict(PROJECT_ID, MODEL_ID, text)
    (out, _) = capsys.readouterr()
    assert 'Text Extract Entity Types: ' in out