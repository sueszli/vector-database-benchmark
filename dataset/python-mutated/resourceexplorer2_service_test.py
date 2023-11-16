from unittest.mock import patch
import botocore
from boto3 import session
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.resourceexplorer2.resourceexplorer2_service import ResourceExplorer2
from prowler.providers.common.models import Audit_Metadata
AWS_ACCOUNT_NUMBER = '123456789012'
AWS_REGION = 'eu-west-1'
INDEX_ARN = 'arn:aws:resource-explorer-2:ap-south-1:123456789012:index/123456-2896-4fe8-93d2-15ec137e5c47'
INDEX_REGION = 'us-east-1'
make_api_call = botocore.client.BaseClient._make_api_call

def mock_make_api_call(self, operation_name, kwarg):
    if False:
        return 10
    '\n    Mock every AWS API call\n    '
    if operation_name == 'ListIndexes':
        return {'Indexes': [{'Arn': INDEX_ARN, 'Region': INDEX_REGION, 'Type': 'LOCAL'}]}
    return make_api_call(self, operation_name, kwarg)

def mock_generate_regional_clients(service, audit_info, _):
    if False:
        return 10
    regional_client = audit_info.audit_session.client(service, region_name=AWS_REGION)
    regional_client.region = AWS_REGION
    return {AWS_REGION: regional_client}

@patch('botocore.client.BaseClient._make_api_call', new=mock_make_api_call)
@patch('prowler.providers.aws.lib.service.service.generate_regional_clients', new=mock_generate_regional_clients)
class Test_ResourceExplorer2_Service:

    def set_mocked_audit_info(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=['us-east-1'], organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    def test__get_client__(self):
        if False:
            i = 10
            return i + 15
        audit_info = self.set_mocked_audit_info()
        resourceeplorer2 = ResourceExplorer2(audit_info)
        assert resourceeplorer2.regional_clients[AWS_REGION].__class__.__name__ == 'ResourceExplorer'

    def test__get_service__(self):
        if False:
            i = 10
            return i + 15
        audit_info = self.set_mocked_audit_info()
        resourceeplorer2 = ResourceExplorer2(audit_info)
        assert resourceeplorer2.service == 'resource-explorer-2'

    def test__list_indexes__(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = self.set_mocked_audit_info()
        resourceeplorer2 = ResourceExplorer2(audit_info)
        assert len(resourceeplorer2.indexes) == 1
        assert resourceeplorer2.indexes[0].arn == INDEX_ARN
        assert resourceeplorer2.indexes[0].region == INDEX_REGION
        assert resourceeplorer2.indexes[0].type == 'LOCAL'