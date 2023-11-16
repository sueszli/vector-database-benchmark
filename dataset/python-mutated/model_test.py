import os
import random
import string
import time
from google.api_core import exceptions, retry
import automl_tables_model
import dataset_test
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
REGION = 'us-central1'
STATIC_MODEL = 'do_not_delete_this_model_0'
GCS_DATASET = 'gs://cloud-ml-tables-data/bank-marketing.csv'
ID = '{rand}_{time}'.format(rand=''.join([random.choice(string.ascii_letters + string.digits) for n in range(4)]), time=int(time.time()))

def _id(name):
    if False:
        for i in range(10):
            print('nop')
    return f'{name}_{ID}'

@retry.Retry()
def test_list_models():
    if False:
        print('Hello World!')
    ensure_model_ready()
    assert next((m for m in automl_tables_model.list_models(PROJECT, REGION) if m.display_name == STATIC_MODEL), None) is not None

@retry.Retry()
def test_list_model_evaluations():
    if False:
        while True:
            i = 10
    model = ensure_model_ready()
    mes = automl_tables_model.list_model_evaluations(PROJECT, REGION, model.display_name)
    assert len(mes) > 0
    for me in mes:
        assert me.name.startswith(model.name)

@retry.Retry()
def test_get_model_evaluations():
    if False:
        while True:
            i = 10
    model = ensure_model_ready()
    me = automl_tables_model.list_model_evaluations(PROJECT, REGION, model.display_name)[0]
    mep = automl_tables_model.get_model_evaluation(PROJECT, REGION, model.name.rpartition('/')[2], me.name.rpartition('/')[2])
    assert mep.name == me.name

def ensure_model_ready():
    if False:
        return 10
    name = STATIC_MODEL
    try:
        return automl_tables_model.get_model(PROJECT, REGION, name)
    except exceptions.NotFound:
        pass
    dataset = dataset_test.ensure_dataset_ready()
    return automl_tables_model.create_model(PROJECT, REGION, dataset.display_name, name, 1000)