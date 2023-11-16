from boto3 import session
from botocore.client import BaseClient
from mock import patch
from moto import mock_athena
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.athena.athena_service import Athena
from prowler.providers.common.models import Audit_Metadata
AWS_ACCOUNT_NUMBER = '123456789012'
AWS_REGION = 'eu-west-1'
make_api_call = BaseClient._make_api_call

def mock_make_api_call(self, operation_name, kwarg):
    if False:
        while True:
            i = 10
    '\n    Mock every AWS API call using Boto3\n\n    As you can see the operation_name has the get_work_group snake_case form but\n    we are using the GetWorkGroup form.\n    Rationale -> https://github.com/boto/botocore/blob/develop/botocore/client.py#L810:L816\n    '
    if operation_name == 'GetWorkGroup':
        return {'WorkGroup': {'Name': 'primary', 'State': 'ENABLED', 'Configuration': {'ResultConfiguration': {'EncryptionConfiguration': {'EncryptionOption': 'SSE_S3'}}, 'EnforceWorkGroupConfiguration': True}}}
    return make_api_call(self, operation_name, kwarg)

def mock_generate_regional_clients(service, audit_info, _):
    if False:
        for i in range(10):
            print('nop')
    regional_client = audit_info.audit_session.client(service, region_name=AWS_REGION)
    regional_client.region = AWS_REGION
    return {AWS_REGION: regional_client}

@patch('prowler.providers.aws.lib.service.service.generate_regional_clients', new=mock_generate_regional_clients)
class Test_Athena_Service:

    def set_mocked_audit_info(self):
        if False:
            return 10
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_athena
    def test__get_workgroups__not_encrypted(self):
        if False:
            while True:
                i = 10
        default_workgroup_name = 'primary'
        audit_info = self.set_mocked_audit_info()
        workgroup_arn = f'arn:{audit_info.audited_partition}:athena:{AWS_REGION}:{audit_info.audited_account}:workgroup/{default_workgroup_name}'
        athena = Athena(audit_info)
        assert len(athena.workgroups) == 1
        assert athena.workgroups[workgroup_arn]
        assert athena.workgroups[workgroup_arn].arn == workgroup_arn
        assert athena.workgroups[workgroup_arn].name == default_workgroup_name
        assert athena.workgroups[workgroup_arn].region == AWS_REGION
        assert athena.workgroups[workgroup_arn].tags == []
        assert athena.workgroups[workgroup_arn].encryption_configuration.encrypted is False
        assert athena.workgroups[workgroup_arn].encryption_configuration.encryption_option == ''
        assert athena.workgroups[workgroup_arn].enforce_workgroup_configuration is False

    @patch('botocore.client.BaseClient._make_api_call', new=mock_make_api_call)
    @mock_athena
    def test__get_workgroups__encrypted(self):
        if False:
            return 10
        default_workgroup_name = 'primary'
        audit_info = self.set_mocked_audit_info()
        workgroup_arn = f'arn:{audit_info.audited_partition}:athena:{AWS_REGION}:{audit_info.audited_account}:workgroup/{default_workgroup_name}'
        athena = Athena(audit_info)
        assert len(athena.workgroups) == 1
        assert athena.workgroups[workgroup_arn]
        assert athena.workgroups[workgroup_arn].arn == workgroup_arn
        assert athena.workgroups[workgroup_arn].name == default_workgroup_name
        assert athena.workgroups[workgroup_arn].region == AWS_REGION
        assert athena.workgroups[workgroup_arn].tags == []
        assert athena.workgroups[workgroup_arn].encryption_configuration.encrypted is True
        assert athena.workgroups[workgroup_arn].encryption_configuration.encryption_option == 'SSE_S3'
        assert athena.workgroups[workgroup_arn].enforce_workgroup_configuration is True