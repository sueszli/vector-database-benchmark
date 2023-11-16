import os
from contentwarehouse.snippets import test_utilities
from contentwarehouse.snippets import update_document_schema_sample
import pytest
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
location = 'us'
document_schema_id = '0gc5eijqsb18g'

def test_update_document_schema_sample(capsys: pytest.CaptureFixture) -> None:
    if False:
        print('Hello World!')
    project_number = test_utilities.get_project_number(project_id)
    update_document_schema_sample.update_document_schema(project_number=project_number, location=location, document_schema_id=document_schema_id)
    (out, _) = capsys.readouterr()
    assert 'Updated Document Schema' in out