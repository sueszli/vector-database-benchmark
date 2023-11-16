import os
from contentwarehouse.snippets import create_document_sample
from contentwarehouse.snippets import create_document_schema_sample
from contentwarehouse.snippets import delete_document_sample
from contentwarehouse.snippets import delete_document_schema_sample
from contentwarehouse.snippets import get_document_sample
from contentwarehouse.snippets import test_utilities
from contentwarehouse.snippets import update_document_sample
from google.cloud import contentwarehouse
from google.cloud.contentwarehouse import Document
import pytest
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
location = 'us'
raw_document_path = 'gs://cloud-samples-data/documentai/codelabs/warehouse/order-invoice.pdf'
raw_document_file_type = contentwarehouse.RawDocumentFileType.RAW_DOCUMENT_FILE_TYPE_PDF
user_id = os.environ['user_id']
reference_id = '001'

@pytest.mark.dependency(name='create_schema')
def test_create_document_schema(request: pytest.fixture) -> None:
    if False:
        while True:
            i = 10
    project_number = test_utilities.get_project_number(project_id)
    response = create_document_schema_sample.sample_create_document_schema(project_number=project_number, location=location)
    assert 'display_name' in response
    document_schema_id = response.name.split('/')[-1]
    request.config.cache.set('document_schema_id', document_schema_id)

@pytest.mark.dependency(name='create_doc', depends=['create_schema'])
def test_create_document(request: pytest.fixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    project_number = test_utilities.get_project_number(project_id)
    document_schema_id = request.config.cache.get('document_schema_id', None)
    response = create_document_sample.sample_create_document(project_number=project_number, location=location, raw_document_path=raw_document_path, raw_document_file_type=raw_document_file_type, document_schema_id=document_schema_id, user_id=user_id, reference_id=reference_id)
    assert 'document' in response
    request.config.cache.set('document_name', response.document.name)

@pytest.mark.dependency(name='get_doc', depends=['create_doc'])
def test_get_document(request: pytest.fixture) -> None:
    if False:
        while True:
            i = 10
    document_name = request.config.cache.get('document_name', None)
    response = get_document_sample.sample_get_document(document_name=document_name, user_id=user_id)
    assert 'name' in response
    request.config.cache.set('document', Document.to_json(response, including_default_value_fields=False, preserving_proto_field_name=False))

@pytest.mark.dependency(name='update_doc', depends=['get_doc'])
def test_update_document(request: pytest.fixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    document_name = request.config.cache.get('document_name', None)
    document_json = request.config.cache.get('document', None)
    document = Document.from_json(document_json)
    response = update_document_sample.sample_update_document(document_name=document_name, document=document, user_id=user_id)
    assert 'document' in response

@pytest.mark.dependency(name='delete_doc', depends=['update_doc'])
def test_delete_document(request: pytest.fixture) -> None:
    if False:
        while True:
            i = 10
    document_name = request.config.cache.get('document_name', None)
    response = delete_document_sample.sample_delete_document(document_name=document_name, user_id=user_id)
    assert response is None

@pytest.mark.dependency(name='delete_schema', depends=['delete_doc'])
def test_delete_document_schema(request: pytest.fixture) -> None:
    if False:
        return 10
    project_number = test_utilities.get_project_number(project_id)
    document_schema_id = request.config.cache.get('document_schema_id', None)
    response = delete_document_schema_sample.sample_delete_document_schema(project_number=project_number, location=location, document_schema_id=document_schema_id)
    assert response is None