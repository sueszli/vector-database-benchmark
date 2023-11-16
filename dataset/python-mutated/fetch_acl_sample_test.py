import os
from contentwarehouse.snippets import fetch_acl_sample
from contentwarehouse.snippets import test_utilities
from google.api_core.exceptions import InvalidArgument, PermissionDenied
import pytest
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
location = 'us'
document_id = '2bq3m8uih3j78'
user_id = 'user:xxxx@example.com'

def test_fetch_project_acl(capsys: pytest.CaptureFixture) -> None:
    if False:
        print('Hello World!')
    project_number = test_utilities.get_project_number(project_id)
    with pytest.raises(InvalidArgument):
        fetch_acl_sample.fetch_acl(project_number=project_number, location=location, user_id=user_id)
        (out, _) = capsys.readouterr()

def test_fetch_document_acl(capsys: pytest.CaptureFixture) -> None:
    if False:
        print('Hello World!')
    project_number = test_utilities.get_project_number(project_id)
    with pytest.raises(PermissionDenied):
        fetch_acl_sample.fetch_acl(project_number=project_number, location=location, user_id=user_id, document_id=document_id)
        (out, _) = capsys.readouterr()