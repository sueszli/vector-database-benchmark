import os
import re
import pytest
from snippets.get_deny_policy import get_deny_policy
from snippets.list_deny_policies import list_deny_policy
from snippets.update_deny_policy import update_deny_policy
PROJECT_ID = os.environ['IAM_PROJECT_ID']
GOOGLE_APPLICATION_CREDENTIALS = os.environ['IAM_CREDENTIALS']

def test_retrieve_policy(capsys: 'pytest.CaptureFixture[str]', deny_policy: str) -> None:
    if False:
        while True:
            i = 10
    get_deny_policy(PROJECT_ID, deny_policy)
    (out, _) = capsys.readouterr()
    assert re.search(f'Retrieved the deny policy: {deny_policy}', out)

def test_list_policies(capsys: 'pytest.CaptureFixture[str]', deny_policy: str) -> None:
    if False:
        return 10
    list_deny_policy(PROJECT_ID)
    (out, _) = capsys.readouterr()
    assert re.search(deny_policy, out)
    assert re.search('Listed all deny policies', out)

def test_update_deny_policy(capsys: 'pytest.CaptureFixture[str]', deny_policy: str) -> None:
    if False:
        while True:
            i = 10
    policy = get_deny_policy(PROJECT_ID, deny_policy)
    update_deny_policy(PROJECT_ID, deny_policy, policy.etag)
    (out, _) = capsys.readouterr()
    assert re.search(f'Updated the deny policy: {deny_policy}', out)