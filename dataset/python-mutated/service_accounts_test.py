import os
import uuid
from googleapiclient.errors import HttpError
import pytest
import service_accounts

def test_service_accounts(capsys: pytest.CaptureFixture) -> None:
    if False:
        while True:
            i = 10
    project_id = os.environ['GOOGLE_CLOUD_PROJECT']
    name = f'test-{uuid.uuid4().hex[:25]}'
    try:
        acct = service_accounts.create_service_account(project_id, name, 'Py Test Account')
        assert 'uniqueId' in acct
        unique_id = acct['uniqueId']
        service_accounts.list_service_accounts(project_id)
        service_accounts.rename_service_account(unique_id, 'Updated Py Test Account')
        service_accounts.disable_service_account(unique_id)
        service_accounts.enable_service_account(unique_id)
        service_accounts.delete_service_account(unique_id)
    finally:
        try:
            service_accounts.delete_service_account(unique_id)
        except HttpError as e:
            if '403' in str(e) or '404' in str(e):
                print('Ignoring 404/403 error upon cleanup.')
            else:
                raise