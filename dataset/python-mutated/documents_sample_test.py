import os
from discoveryengine import import_documents_sample
from discoveryengine import list_documents_sample
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
location = 'global'
data_store_id = 'test-structured-data-engine'
gcs_uri = 'gs://cloud-samples-data/gen-app-builder/search/empty.json'
bigquery_dataset = 'genappbuilder_test'
bigquery_table = 'import_documents_test'

def test_import_documents_gcs():
    if False:
        for i in range(10):
            print('nop')
    operation_name = import_documents_sample.import_documents_sample(project_id=project_id, location=location, data_store_id=data_store_id, gcs_uri=gcs_uri)
    assert 'operations/import-documents' in operation_name

def test_import_documents_bigquery():
    if False:
        while True:
            i = 10
    operation_name = import_documents_sample.import_documents_sample(project_id=project_id, location=location, data_store_id=data_store_id, bigquery_dataset=bigquery_dataset, bigquery_table=bigquery_table)
    assert 'operations/import-documents' in operation_name

def test_list_documents():
    if False:
        print('Hello World!')
    response = list_documents_sample.list_documents_sample(project_id=project_id, location=location, data_store_id=data_store_id)
    assert response