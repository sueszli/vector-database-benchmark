import os
from contentwarehouse.snippets import create_document_schema_sample
from contentwarehouse.snippets import delete_document_schema_sample
from contentwarehouse.snippets import get_document_schema_sample
from contentwarehouse.snippets import test_utilities
import pytest
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
location = 'us'

@pytest.mark.dependency(name='create')
def test_create_document_schema(request: pytest.fixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    project_number = test_utilities.get_project_number(project_id)
    response = create_document_schema_sample.sample_create_document_schema(project_number=project_number, location=location)
    assert 'display_name' in response
    document_schema_id = response.name.split('/')[-1]
    request.config.cache.set('document_schema_id', document_schema_id)

@pytest.mark.dependency(name='get', depends=['create'])
def test_get_document_schema(request: pytest.fixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    project_number = test_utilities.get_project_number(project_id)
    document_schema_id = request.config.cache.get('document_schema_id', None)
    response = get_document_schema_sample.sample_get_document_schema(project_number=project_number, location=location, document_schema_id=document_schema_id)
    assert 'display_name' in response

@pytest.mark.dependency(name='delete', depends=['get'])
def test_delete_document_schema(request: pytest.fixture) -> None:
    if False:
        return 10
    project_number = test_utilities.get_project_number(project_id)
    document_schema_id = request.config.cache.get('document_schema_id', None)
    response = delete_document_schema_sample.sample_delete_document_schema(project_number=project_number, location=location, document_schema_id=document_schema_id)
    assert response is None