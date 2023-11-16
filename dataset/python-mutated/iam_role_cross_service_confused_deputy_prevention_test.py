from json import dumps
from unittest import mock
from boto3 import client, session
from moto import mock_iam
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.iam.iam_service import Role
from prowler.providers.common.models import Audit_Metadata
AWS_REGION = 'us-east-1'
AWS_ACCOUNT_ID = '123456789012'

class Test_iam_role_cross_service_confused_deputy_prevention:

    def set_mocked_audit_info(self):
        if False:
            print('Hello World!')
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_ID, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_ID}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=['us-east-1', 'eu-west-1'], organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_iam
    def test_no_roles(self):
        if False:
            i = 10
            return i + 15
        from prowler.providers.aws.services.iam.iam_service import IAM
        current_audit_info = self.set_mocked_audit_info()
        current_audit_info.audited_account = AWS_ACCOUNT_ID
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_role_cross_service_confused_deputy_prevention.iam_role_cross_service_confused_deputy_prevention.iam_client', new=IAM(current_audit_info)):
            from prowler.providers.aws.services.iam.iam_role_cross_service_confused_deputy_prevention.iam_role_cross_service_confused_deputy_prevention import iam_role_cross_service_confused_deputy_prevention
            check = iam_role_cross_service_confused_deputy_prevention()
            result = check.execute()
            assert len(result) == 0

    @mock_iam
    def test_only_aws_service_linked_roles(self):
        if False:
            while True:
                i = 10
        iam_client = mock.MagicMock
        iam_client.roles = []
        iam_client.roles.append(Role(name='AWSServiceRoleForAmazonGuardDuty', arn='arn:aws:iam::106908755756:role/aws-service-role/guardduty.amazonaws.com/AWSServiceRoleForAmazonGuardDuty', assume_role_policy={'Version': '2008-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': {'Service': 'ec2.amazonaws.com'}, 'Action': 'sts:AssumeRole'}]}, is_service_role=True))
        current_audit_info = self.set_mocked_audit_info()
        current_audit_info.audited_account = AWS_ACCOUNT_ID
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_role_cross_service_confused_deputy_prevention.iam_role_cross_service_confused_deputy_prevention.iam_client', new=iam_client):
            from prowler.providers.aws.services.iam.iam_role_cross_service_confused_deputy_prevention.iam_role_cross_service_confused_deputy_prevention import iam_role_cross_service_confused_deputy_prevention
            check = iam_role_cross_service_confused_deputy_prevention()
            result = check.execute()
            assert len(result) == 0

    @mock_iam
    def test_iam_service_role_without_cross_service_confused_deputy_prevention(self):
        if False:
            i = 10
            return i + 15
        iam_client = client('iam', region_name=AWS_REGION)
        policy_document = {'Version': '2008-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': {'Service': 'ec2.amazonaws.com'}, 'Action': 'sts:AssumeRole'}]}
        response = iam_client.create_role(RoleName='test', AssumeRolePolicyDocument=dumps(policy_document))
        from prowler.providers.aws.services.iam.iam_service import IAM
        current_audit_info = self.set_mocked_audit_info()
        current_audit_info.audited_account = AWS_ACCOUNT_ID
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_role_cross_service_confused_deputy_prevention.iam_role_cross_service_confused_deputy_prevention.iam_client', new=IAM(current_audit_info)):
            from prowler.providers.aws.services.iam.iam_role_cross_service_confused_deputy_prevention.iam_role_cross_service_confused_deputy_prevention import iam_role_cross_service_confused_deputy_prevention
            check = iam_role_cross_service_confused_deputy_prevention()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert result[0].status_extended == 'IAM Service Role test does not prevent against a cross-service confused deputy attack.'
            assert result[0].resource_id == 'test'
            assert result[0].resource_arn == response['Role']['Arn']

    @mock_iam
    def test_iam_service_role_with_cross_service_confused_deputy_prevention(self):
        if False:
            while True:
                i = 10
        iam_client = client('iam', region_name=AWS_REGION)
        policy_document = {'Version': '2008-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': {'Service': 'workspaces.amazonaws.com'}, 'Action': 'sts:AssumeRole', 'Condition': {'StringEquals': {'aws:SourceAccount': [AWS_ACCOUNT_ID]}}}]}
        response = iam_client.create_role(RoleName='test', AssumeRolePolicyDocument=dumps(policy_document))
        from prowler.providers.aws.services.iam.iam_service import IAM
        current_audit_info = self.set_mocked_audit_info()
        current_audit_info.audited_account = AWS_ACCOUNT_ID
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_role_cross_service_confused_deputy_prevention.iam_role_cross_service_confused_deputy_prevention.iam_client', new=IAM(current_audit_info)):
            from prowler.providers.aws.services.iam.iam_role_cross_service_confused_deputy_prevention.iam_role_cross_service_confused_deputy_prevention import iam_role_cross_service_confused_deputy_prevention
            check = iam_role_cross_service_confused_deputy_prevention()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert result[0].status_extended == 'IAM Service Role test prevents against a cross-service confused deputy attack.'
            assert result[0].resource_id == 'test'
            assert result[0].resource_arn == response['Role']['Arn']

    @mock_iam
    def test_iam_service_role_with_cross_service_confused_deputy_prevention_stringlike(self):
        if False:
            i = 10
            return i + 15
        iam_client = client('iam', region_name=AWS_REGION)
        policy_document = {'Version': '2008-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': {'Service': 'workspaces.amazonaws.com'}, 'Action': 'sts:AssumeRole', 'Condition': {'StringLike': {'aws:SourceAccount': [AWS_ACCOUNT_ID]}}}]}
        response = iam_client.create_role(RoleName='test', AssumeRolePolicyDocument=dumps(policy_document))
        from prowler.providers.aws.services.iam.iam_service import IAM
        current_audit_info = self.set_mocked_audit_info()
        current_audit_info.audited_account = AWS_ACCOUNT_ID
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_role_cross_service_confused_deputy_prevention.iam_role_cross_service_confused_deputy_prevention.iam_client', new=IAM(current_audit_info)):
            from prowler.providers.aws.services.iam.iam_role_cross_service_confused_deputy_prevention.iam_role_cross_service_confused_deputy_prevention import iam_role_cross_service_confused_deputy_prevention
            check = iam_role_cross_service_confused_deputy_prevention()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert result[0].status_extended == 'IAM Service Role test prevents against a cross-service confused deputy attack.'
            assert result[0].resource_id == 'test'
            assert result[0].resource_arn == response['Role']['Arn']

    @mock_iam
    def test_iam_service_role_with_cross_service_confused_deputy_prevention_PrincipalAccount(self):
        if False:
            return 10
        iam_client = client('iam', region_name=AWS_REGION)
        policy_document = {'Version': '2008-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': {'Service': 'workspaces.amazonaws.com'}, 'Action': 'sts:AssumeRole', 'Condition': {'StringLike': {'aws:PrincipalAccount': [AWS_ACCOUNT_ID]}}}]}
        response = iam_client.create_role(RoleName='test', AssumeRolePolicyDocument=dumps(policy_document))
        from prowler.providers.aws.services.iam.iam_service import IAM
        current_audit_info = self.set_mocked_audit_info()
        current_audit_info.audited_account = AWS_ACCOUNT_ID
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_role_cross_service_confused_deputy_prevention.iam_role_cross_service_confused_deputy_prevention.iam_client', new=IAM(current_audit_info)):
            from prowler.providers.aws.services.iam.iam_role_cross_service_confused_deputy_prevention.iam_role_cross_service_confused_deputy_prevention import iam_role_cross_service_confused_deputy_prevention
            check = iam_role_cross_service_confused_deputy_prevention()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert result[0].status_extended == 'IAM Service Role test prevents against a cross-service confused deputy attack.'
            assert result[0].resource_id == 'test'
            assert result[0].resource_arn == response['Role']['Arn']

    @mock_iam
    def test_iam_service_role_with_cross_service_confused_deputy_prevention_ResourceAccount(self):
        if False:
            print('Hello World!')
        iam_client = client('iam', region_name=AWS_REGION)
        policy_document = {'Version': '2008-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': {'Service': 'workspaces.amazonaws.com'}, 'Action': 'sts:AssumeRole', 'Condition': {'StringLike': {'aws:ResourceAccount': [AWS_ACCOUNT_ID]}}}]}
        response = iam_client.create_role(RoleName='test', AssumeRolePolicyDocument=dumps(policy_document))
        from prowler.providers.aws.services.iam.iam_service import IAM
        current_audit_info = self.set_mocked_audit_info()
        current_audit_info.audited_account = AWS_ACCOUNT_ID
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_role_cross_service_confused_deputy_prevention.iam_role_cross_service_confused_deputy_prevention.iam_client', new=IAM(current_audit_info)):
            from prowler.providers.aws.services.iam.iam_role_cross_service_confused_deputy_prevention.iam_role_cross_service_confused_deputy_prevention import iam_role_cross_service_confused_deputy_prevention
            check = iam_role_cross_service_confused_deputy_prevention()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert result[0].status_extended == 'IAM Service Role test prevents against a cross-service confused deputy attack.'
            assert result[0].resource_id == 'test'
            assert result[0].resource_arn == response['Role']['Arn']