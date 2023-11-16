import os
from contentwarehouse.snippets import set_acl_sample
from contentwarehouse.snippets import test_utilities
from google.api_core.exceptions import InvalidArgument
import pytest
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
location = 'us'
document_id = '2bq3m8uih3j78'
user_id = 'user:xxxx@example.com'
policy = {'bindings': [{'role': 'roles/contentwarehouse.documentAdmin', 'members': ['xxxx@example.com']}]}

def test_set_project_acl(capsys: pytest.CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    project_number = test_utilities.get_project_number(project_id)
    with pytest.raises(InvalidArgument):
        set_acl_sample.set_acl(project_number=project_number, location=location, policy=policy, user_id=user_id)
        capsys.readouterr()

def test_set_document_acl(capsys: pytest.CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    project_number = test_utilities.get_project_number(project_id)
    with pytest.raises(InvalidArgument):
        set_acl_sample.set_acl(project_number=project_number, location=location, policy=policy, user_id=user_id, document_id=document_id)
        capsys.readouterr()