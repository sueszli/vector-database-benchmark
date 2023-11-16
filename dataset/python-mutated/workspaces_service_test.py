from unittest.mock import patch
from uuid import uuid4
import botocore
from boto3 import session
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.workspaces.workspaces_service import WorkSpaces
from prowler.providers.common.models import Audit_Metadata
AWS_ACCOUNT_NUMBER = '123456789012'
AWS_REGION = 'eu-west-1'
workspace_id = str(uuid4())
make_api_call = botocore.client.BaseClient._make_api_call

def mock_make_api_call(self, operation_name, kwarg):
    if False:
        i = 10
        return i + 15
    if operation_name == 'DescribeWorkspaces':
        return {'Workspaces': [{'WorkspaceId': workspace_id, 'UserVolumeEncryptionEnabled': True, 'RootVolumeEncryptionEnabled': True, 'SubnetId': 'subnet-1234567890'}]}
    if operation_name == 'DescribeTags':
        return {'TagList': [{'Key': 'test', 'Value': 'test'}]}
    return make_api_call(self, operation_name, kwarg)

def mock_generate_regional_clients(service, audit_info, _):
    if False:
        print('Hello World!')
    regional_client = audit_info.audit_session.client(service, region_name=AWS_REGION)
    regional_client.region = AWS_REGION
    return {AWS_REGION: regional_client}

@patch('botocore.client.BaseClient._make_api_call', new=mock_make_api_call)
@patch('prowler.providers.aws.lib.service.service.generate_regional_clients', new=mock_generate_regional_clients)
class Test_WorkSpaces_Service:

    def set_mocked_audit_info(self):
        if False:
            i = 10
            return i + 15
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    def test_service(self):
        if False:
            while True:
                i = 10
        audit_info = self.set_mocked_audit_info()
        workspaces = WorkSpaces(audit_info)
        assert workspaces.service == 'workspaces'

    def test_client(self):
        if False:
            while True:
                i = 10
        audit_info = self.set_mocked_audit_info()
        workspaces = WorkSpaces(audit_info)
        for reg_client in workspaces.regional_clients.values():
            assert reg_client.__class__.__name__ == 'WorkSpaces'

    def test__get_session__(self):
        if False:
            while True:
                i = 10
        audit_info = self.set_mocked_audit_info()
        workspaces = WorkSpaces(audit_info)
        assert workspaces.session.__class__.__name__ == 'Session'

    def test__describe_workspaces__(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = self.set_mocked_audit_info()
        workspaces = WorkSpaces(audit_info)
        assert len(workspaces.workspaces) == 1
        assert workspaces.workspaces[0].id == workspace_id
        assert workspaces.workspaces[0].region == AWS_REGION
        assert workspaces.workspaces[0].tags == [{'Key': 'test', 'Value': 'test'}]
        assert workspaces.workspaces[0].user_volume_encryption_enabled
        assert workspaces.workspaces[0].root_volume_encryption_enabled