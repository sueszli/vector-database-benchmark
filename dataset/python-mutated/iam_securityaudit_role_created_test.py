from json import dumps
from re import search
from unittest import mock
from boto3 import client, session
from moto import mock_iam
from prowler.providers.aws.lib.audit_info.audit_info import AWS_Audit_Info
from prowler.providers.aws.services.iam.iam_service import IAM
from prowler.providers.common.models import Audit_Metadata
AWS_ACCOUNT_NUMBER = '123456789012'

class Test_iam_securityaudit_role_created:

    def set_mocked_audit_info(self):
        if False:
            while True:
                i = 10
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region='us-east-1', credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_iam
    def test_securityaudit_role_created(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = self.set_mocked_audit_info()
        iam = client('iam')
        role_name = 'test_securityaudit_role_created'
        assume_role_policy_document = {'Version': '2012-10-17', 'Statement': {'Sid': 'test', 'Effect': 'Allow', 'Principal': {'AWS': '*'}, 'Action': 'sts:AssumeRole'}}
        iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=dumps(assume_role_policy_document))
        iam.attach_role_policy(RoleName=role_name, PolicyArn='arn:aws:iam::aws:policy/SecurityAudit')
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.iam.iam_securityaudit_role_created.iam_securityaudit_role_created.iam_client', new=IAM(audit_info)):
                from prowler.providers.aws.services.iam.iam_securityaudit_role_created.iam_securityaudit_role_created import iam_securityaudit_role_created
                check = iam_securityaudit_role_created()
                result = check.execute()
                assert result[0].status == 'PASS'
                assert search(f'SecurityAudit policy attached to role {role_name}.', result[0].status_extended)
                assert result[0].resource_id == 'SecurityAudit'
                assert result[0].resource_arn == 'arn:aws:iam::aws:policy/SecurityAudit'
                assert result[0].region == 'us-east-1'

    @mock_iam
    def test_no_securityaudit_role_created(self):
        if False:
            i = 10
            return i + 15
        audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.iam.iam_securityaudit_role_created.iam_securityaudit_role_created.iam_client', new=IAM(audit_info)):
                from prowler.providers.aws.services.iam.iam_securityaudit_role_created.iam_securityaudit_role_created import iam_securityaudit_role_created
                check = iam_securityaudit_role_created()
                result = check.execute()
                assert result[0].status == 'FAIL'
                assert result[0].status_extended == 'SecurityAudit policy is not attached to any role.'
                assert result[0].resource_id == 'SecurityAudit'
                assert result[0].resource_arn == 'arn:aws:iam::aws:policy/SecurityAudit'
                assert result[0].region == 'us-east-1'