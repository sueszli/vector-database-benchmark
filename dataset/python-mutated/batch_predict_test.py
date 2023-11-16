import os
from google.cloud.automl_v1beta1 import Model
import pytest
import automl_tables_model
import automl_tables_predict
import model_test
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
REGION = 'us-central1'
STATIC_MODEL = model_test.STATIC_MODEL
GCS_INPUT = f'gs://{PROJECT}-automl-tables-test/bank-marketing.csv'
GCS_OUTPUT = f'gs://{PROJECT}-automl-tables-test/TABLE_TEST_OUTPUT/'
BQ_INPUT = f'bq://{PROJECT}.automl_test.bank_marketing'
BQ_OUTPUT = f'bq://{PROJECT}'
PARAMS = {}

@pytest.mark.slow
def test_batch_predict(capsys):
    if False:
        print('Hello World!')
    ensure_model_online()
    automl_tables_predict.batch_predict(PROJECT, REGION, STATIC_MODEL, GCS_INPUT, GCS_OUTPUT, PARAMS)
    (out, _) = capsys.readouterr()
    assert 'Batch prediction complete' in out

@pytest.mark.slow
def test_batch_predict_bq(capsys):
    if False:
        print('Hello World!')
    ensure_model_online()
    automl_tables_predict.batch_predict_bq(PROJECT, REGION, STATIC_MODEL, BQ_INPUT, BQ_OUTPUT, PARAMS)
    (out, _) = capsys.readouterr()
    assert 'Batch prediction complete' in out

def ensure_model_online():
    if False:
        for i in range(10):
            print('nop')
    model = model_test.ensure_model_ready()
    if model.deployment_state != Model.DeploymentState.DEPLOYED:
        automl_tables_model.deploy_model(PROJECT, REGION, model.display_name)
    return automl_tables_model.get_model(PROJECT, REGION, model.display_name)