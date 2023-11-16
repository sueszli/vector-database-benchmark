import os
from typing import Iterator
import uuid
from googleapiclient import errors
import pytest
from retrying import retry
import access
import service_accounts
GCLOUD_PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
GCP_ROLE = 'roles/owner'

def retry_if_conflict(exception: Exception) -> bool:
    if False:
        i = 10
        return i + 15
    return isinstance(exception, errors.HttpError) and 'There were concurrent policy changes' in str(exception)

@pytest.fixture(scope='module')
def test_member() -> Iterator[str]:
    if False:
        return 10
    name = 'python-test-' + str(uuid.uuid4()).split('-')[0]
    email = name + '@' + GCLOUD_PROJECT + '.iam.gserviceaccount.com'
    member = 'serviceAccount:' + email
    service_accounts.create_service_account(GCLOUD_PROJECT, name, 'Py Test Account')
    yield member
    service_accounts.delete_service_account(email)

def test_get_policy(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        print('Hello World!')
    access.get_policy(GCLOUD_PROJECT, version=3)
    (out, _) = capsys.readouterr()
    assert 'etag' in out

def test_modify_policy_add_role(test_member: str, capsys: pytest.LogCaptureFixture) -> None:
    if False:
        return 10

    @retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5, retry_on_exception=retry_if_conflict)
    def test_call() -> None:
        if False:
            while True:
                i = 10
        policy = access.get_policy(GCLOUD_PROJECT, version=3)
        access.modify_policy_add_role(policy, GCLOUD_PROJECT, test_member)
        (out, _) = capsys.readouterr()
        assert 'etag' in out
    test_call()

def test_modify_policy_remove_member(test_member: str, capsys: pytest.LogCaptureFixture) -> None:
    if False:
        i = 10
        return i + 15

    @retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5, retry_on_exception=retry_if_conflict)
    def test_call() -> None:
        if False:
            i = 10
            return i + 15
        policy = access.get_policy(GCLOUD_PROJECT, version=3)
        access.modify_policy_remove_member(policy, GCP_ROLE, test_member)
        (out, _) = capsys.readouterr()
        assert 'iam.gserviceaccount.com' in out
    test_call()

def test_set_policy(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        while True:
            i = 10

    @retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5, retry_on_exception=retry_if_conflict)
    def test_call() -> None:
        if False:
            print('Hello World!')
        policy = access.get_policy(GCLOUD_PROJECT, version=3)
        access.set_policy(GCLOUD_PROJECT, policy)
        (out, _) = capsys.readouterr()
        assert 'etag' in out
    test_call()

def test_permissions(capsys: pytest.LogCaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    access.test_permissions(GCLOUD_PROJECT)
    (out, _) = capsys.readouterr()
    assert 'permissions' in out