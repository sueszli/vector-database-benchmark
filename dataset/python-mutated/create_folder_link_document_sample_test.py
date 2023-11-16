import os
from contentwarehouse.snippets import create_folder_link_document_sample
from contentwarehouse.snippets import test_utilities
import pytest
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
location = 'us'
user_id = 'user:xxxx@example.com'

def test_create_folder_link_document(capsys: pytest.CaptureFixture) -> None:
    if False:
        return 10
    project_number = test_utilities.get_project_number(project_id)
    create_folder_link_document_sample.create_folder_link_document(project_number=project_number, location=location, user_id=user_id)
    (out, _) = capsys.readouterr()
    assert 'Rule Engine Output' in out
    assert 'Folder Created' in out
    assert 'Rule Engine Output' in out
    assert 'Document Created' in out
    assert 'Link Created' in out
    assert 'Validate Link Created' in out