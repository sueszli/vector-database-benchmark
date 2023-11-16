from re import search
from unittest import mock
from prowler.providers.gcp.services.iam.iam_service import Setting
GCP_PROJECT_ID = '123456789012'

class Test_iam_account_access_approval_enabled:

    def test_iam_no_settings(self):
        if False:
            for i in range(10):
                print('nop')
        accessapproval_client = mock.MagicMock
        accessapproval_client.settings = {}
        accessapproval_client.project_ids = [GCP_PROJECT_ID]
        accessapproval_client.region = 'global'
        with mock.patch('prowler.providers.gcp.services.iam.iam_account_access_approval_enabled.iam_account_access_approval_enabled.accessapproval_client', new=accessapproval_client):
            from prowler.providers.gcp.services.iam.iam_account_access_approval_enabled.iam_account_access_approval_enabled import iam_account_access_approval_enabled
            check = iam_account_access_approval_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert search('does not have Access Approval enabled', result[0].status_extended)
            assert result[0].resource_id == GCP_PROJECT_ID
            assert result[0].project_id == GCP_PROJECT_ID
            assert result[0].location == 'global'

    def test_iam_project_with_settings(self):
        if False:
            for i in range(10):
                print('nop')
        accessapproval_client = mock.MagicMock
        accessapproval_client.settings = {GCP_PROJECT_ID: Setting(name='test', project_id=GCP_PROJECT_ID)}
        accessapproval_client.project_ids = [GCP_PROJECT_ID]
        accessapproval_client.region = 'global'
        with mock.patch('prowler.providers.gcp.services.iam.iam_account_access_approval_enabled.iam_account_access_approval_enabled.accessapproval_client', new=accessapproval_client):
            from prowler.providers.gcp.services.iam.iam_account_access_approval_enabled.iam_account_access_approval_enabled import iam_account_access_approval_enabled
            check = iam_account_access_approval_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert search('has Access Approval enabled', result[0].status_extended)
            assert result[0].resource_id == GCP_PROJECT_ID
            assert result[0].project_id == GCP_PROJECT_ID
            assert result[0].location == 'global'