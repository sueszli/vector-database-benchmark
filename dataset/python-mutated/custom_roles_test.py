import os
from typing import Iterator
import uuid
import pytest
import custom_roles
GCLOUD_PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']

@pytest.fixture(scope='module')
def custom_role() -> Iterator[str]:
    if False:
        for i in range(10):
            print('nop')
    role_name = 'pythonTestCustomRole' + str(uuid.uuid4().hex)
    custom_roles.create_role(role_name, GCLOUD_PROJECT, 'Python Test Custom Role', 'This is a python test custom role', ['iam.roles.get'], 'GA')
    yield role_name
    custom_roles.delete_role(role_name, GCLOUD_PROJECT)

def test_query_testable_permissions(capsys: pytest.CaptureFixture) -> None:
    if False:
        print('Hello World!')
    custom_roles.query_testable_permissions('//cloudresourcemanager.googleapis.com/projects/' + GCLOUD_PROJECT)
    (out, _) = capsys.readouterr()
    assert '\n' in out

def test_list_roles(capsys: pytest.CaptureFixture) -> None:
    if False:
        print('Hello World!')
    custom_roles.list_roles(GCLOUD_PROJECT)
    (out, _) = capsys.readouterr()
    assert 'roles/' in out

def test_get_role(capsys: pytest.CaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    custom_roles.get_role('roles/appengine.appViewer')
    (out, _) = capsys.readouterr()
    assert 'roles/' in out

def test_edit_role(custom_role: dict, capsys: pytest.CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    custom_roles.edit_role(custom_role, GCLOUD_PROJECT, 'Python Test Custom Role', 'Updated', ['iam.roles.get'], 'GA')
    (out, _) = capsys.readouterr()
    assert 'Updated role:' in out

def test_disable_role(custom_role: dict, capsys: pytest.CaptureFixture) -> None:
    if False:
        return 10
    custom_roles.disable_role(custom_role, GCLOUD_PROJECT)
    (out, _) = capsys.readouterr()
    assert 'Disabled role:' in out