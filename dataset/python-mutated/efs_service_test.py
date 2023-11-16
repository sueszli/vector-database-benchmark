import json
from unittest.mock import patch
import botocore
from boto3 import client, session
from moto import mock_efs
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.efs.efs_service import EFS
from prowler.providers.common.models import Audit_Metadata
AWS_REGION = 'eu-west-1'
AWS_ACCOUNT_NUMBER = '123456789012'
make_api_call = botocore.client.BaseClient._make_api_call
file_system_id = 'fs-c7a0456e'
creation_token = 'console-d215fa78-1f83-4651-b026-facafd8a7da7'
backup_policy_status = 'ENABLED'
filesystem_policy = {'Id': '1', 'Statement': [{'Effect': 'Allow', 'Action': ['elasticfilesystem:ClientMount'], 'Principal': {'AWS': f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root'}}]}

def mock_make_api_call(self, operation_name, kwarg):
    if False:
        for i in range(10):
            print('nop')
    if operation_name == 'DescribeFileSystemPolicy':
        return {'FileSystemId': file_system_id, 'Policy': json.dumps(filesystem_policy)}
    if operation_name == 'DescribeBackupPolicy':
        return {'BackupPolicy': {'Status': backup_policy_status}}
    return make_api_call(self, operation_name, kwarg)

def mock_generate_regional_clients(service, audit_info, _):
    if False:
        for i in range(10):
            print('nop')
    regional_client = audit_info.audit_session.client(service, region_name=AWS_REGION)
    regional_client.region = AWS_REGION
    return {AWS_REGION: regional_client}

@patch('botocore.client.BaseClient._make_api_call', new=mock_make_api_call)
@patch('prowler.providers.aws.lib.service.service.generate_regional_clients', new=mock_generate_regional_clients)
class Test_EFS:

    def set_mocked_audit_info(self):
        if False:
            return 10
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    def test__get_session__(self):
        if False:
            print('Hello World!')
        access_analyzer = EFS(self.set_mocked_audit_info())
        assert access_analyzer.session.__class__.__name__ == 'Session'

    def test__get_service__(self):
        if False:
            i = 10
            return i + 15
        access_analyzer = EFS(self.set_mocked_audit_info())
        assert access_analyzer.service == 'efs'

    @mock_efs
    def test__describe_file_systems__(self):
        if False:
            print('Hello World!')
        efs_client = client('efs', AWS_REGION)
        efs = efs_client.create_file_system(CreationToken=creation_token, Encrypted=True, Tags=[{'Key': 'test', 'Value': 'test'}])
        filesystem = EFS(self.set_mocked_audit_info())
        assert len(filesystem.filesystems) == 1
        assert filesystem.filesystems[0].id == efs['FileSystemId']
        assert filesystem.filesystems[0].encrypted == efs['Encrypted']
        assert filesystem.filesystems[0].tags == [{'Key': 'test', 'Value': 'test'}]

    @mock_efs
    def test__describe_file_system_policies__(self):
        if False:
            while True:
                i = 10
        efs_client = client('efs', AWS_REGION)
        efs = efs_client.create_file_system(CreationToken=creation_token, Encrypted=True)
        filesystem = EFS(self.set_mocked_audit_info())
        assert len(filesystem.filesystems) == 1
        assert filesystem.filesystems[0].id == efs['FileSystemId']
        assert filesystem.filesystems[0].encrypted == efs['Encrypted']
        assert filesystem.filesystems[0].backup_policy == backup_policy_status
        assert filesystem.filesystems[0].policy == filesystem_policy