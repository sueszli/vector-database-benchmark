import os
from contentwarehouse.snippets import list_document_schema_sample
from contentwarehouse.snippets import test_utilities
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
location = 'us'

def test_list_document_schemas() -> None:
    if False:
        while True:
            i = 10
    project_number = test_utilities.get_project_number(project_id)
    response = list_document_schema_sample.sample_list_document_schemas(project_number=project_number, location=location)
    for schema in response:
        assert 'display_name' in schema