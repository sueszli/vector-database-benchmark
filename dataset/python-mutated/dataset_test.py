import os
import random
import string
import time
from google.api_core import exceptions, retry
import pytest
import automl_tables_dataset
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
REGION = 'us-central1'
STATIC_DATASET = 'do_not_delete_this_table_python'
GCS_DATASET = 'gs://python-docs-samples-tests-automl-tables-test/bank-marketing.csv'
ID = '{rand}_{time}'.format(rand=''.join([random.choice(string.ascii_letters + string.digits) for n in range(4)]), time=int(time.time()))

def _id(name):
    if False:
        print('Hello World!')
    return f'{name}_{ID}'

def ensure_dataset_ready():
    if False:
        while True:
            i = 10
    dataset = None
    name = STATIC_DATASET
    try:
        dataset = automl_tables_dataset.get_dataset(PROJECT, REGION, name)
    except exceptions.NotFound:
        dataset = automl_tables_dataset.create_dataset(PROJECT, REGION, name)
    if dataset.example_count is None or dataset.example_count == 0:
        automl_tables_dataset.import_data(PROJECT, REGION, name, GCS_DATASET)
        dataset = automl_tables_dataset.get_dataset(PROJECT, REGION, name)
    automl_tables_dataset.update_dataset(PROJECT, REGION, dataset.display_name, target_column_spec_name='Deposit')
    return dataset

@retry.Retry()
@pytest.mark.slow
def test_dataset_create_import_delete(capsys):
    if False:
        print('Hello World!')
    name = _id('d_cr_dl')
    dataset = automl_tables_dataset.create_dataset(PROJECT, REGION, name)
    assert dataset is not None
    assert dataset.display_name == name
    automl_tables_dataset.import_data(PROJECT, REGION, name, GCS_DATASET, dataset.name)
    (out, _) = capsys.readouterr()
    assert 'Data imported.' in out
    automl_tables_dataset.delete_dataset(PROJECT, REGION, name)
    with pytest.raises(exceptions.NotFound):
        automl_tables_dataset.get_dataset(PROJECT, REGION, name)

@retry.Retry()
def test_dataset_update(capsys):
    if False:
        for i in range(10):
            print('nop')
    dataset = ensure_dataset_ready()
    automl_tables_dataset.update_dataset(PROJECT, REGION, dataset.display_name, target_column_spec_name='Deposit', weight_column_spec_name='Balance')
    (out, _) = capsys.readouterr()
    assert 'Target column updated.' in out
    assert 'Weight column updated.' in out

@retry.Retry()
def test_list_datasets():
    if False:
        for i in range(10):
            print('nop')
    ensure_dataset_ready()
    assert next((d for d in automl_tables_dataset.list_datasets(PROJECT, REGION) if d.display_name == STATIC_DATASET), None) is not None