import os
import pytest
import grantable_roles

def test_grantable_roles(capsys: pytest.CaptureFixture) -> None:
    if False:
        print('Hello World!')
    project = os.environ['GOOGLE_CLOUD_PROJECT']
    resource = '//cloudresourcemanager.googleapis.com/projects/' + project
    grantable_roles.view_grantable_roles(resource)
    (out, _) = capsys.readouterr()
    assert 'Title:' in out