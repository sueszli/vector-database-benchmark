from json import dumps
from re import search
from unittest import mock
from boto3 import client, session
from moto import mock_iam
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.common.models import Audit_Metadata
AWS_ACCOUNT_NUMBER = '123456789012'
AWS_REGION = 'us-east-1'

class Test_iam_no_custom_policy_permissive_role_assumption:

    def set_mocked_audit_info(self):
        if False:
            i = 10
            return i + 15
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None, region_name=AWS_REGION), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=AWS_REGION, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_iam
    def test_policy_allows_permissive_role_assumption_wildcard(self):
        if False:
            for i in range(10):
                print('nop')
        iam_client = client('iam')
        policy_name = 'policy1'
        policy_document = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': 'sts:*', 'Resource': '*'}]}
        arn = iam_client.create_policy(PolicyName=policy_name, PolicyDocument=dumps(policy_document))['Policy']['Arn']
        from prowler.providers.aws.services.iam.iam_service import IAM
        audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.iam.iam_no_custom_policy_permissive_role_assumption.iam_no_custom_policy_permissive_role_assumption.iam_client', new=IAM(audit_info)):
                from prowler.providers.aws.services.iam.iam_no_custom_policy_permissive_role_assumption.iam_no_custom_policy_permissive_role_assumption import iam_no_custom_policy_permissive_role_assumption
                check = iam_no_custom_policy_permissive_role_assumption()
                result = check.execute()
                assert result[0].status == 'FAIL'
                assert search(f'Custom Policy {policy_name} allows permissive STS Role assumption', result[0].status_extended)
                assert result[0].resource_arn == arn
                assert result[0].resource_id == policy_name

    @mock_iam
    def test_policy_allows_permissive_role_assumption_no_wilcard(self):
        if False:
            return 10
        iam_client = client('iam')
        policy_name = 'policy1'
        policy_document = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': 'sts:AssumeRole', 'Resource': '*'}]}
        arn = iam_client.create_policy(PolicyName=policy_name, PolicyDocument=dumps(policy_document))['Policy']['Arn']
        from prowler.providers.aws.services.iam.iam_service import IAM
        audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.iam.iam_no_custom_policy_permissive_role_assumption.iam_no_custom_policy_permissive_role_assumption.iam_client', new=IAM(audit_info)):
                from prowler.providers.aws.services.iam.iam_no_custom_policy_permissive_role_assumption.iam_no_custom_policy_permissive_role_assumption import iam_no_custom_policy_permissive_role_assumption
                check = iam_no_custom_policy_permissive_role_assumption()
                result = check.execute()
                assert result[0].status == 'FAIL'
                assert search(f'Custom Policy {policy_name} allows permissive STS Role assumption', result[0].status_extended)
                assert result[0].resource_arn == arn
                assert result[0].resource_id == policy_name

    @mock_iam
    def test_policy_assume_role_not_allow_permissive_role_assumption(self):
        if False:
            i = 10
            return i + 15
        iam_client = client('iam')
        policy_name = 'policy1'
        policy_document = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': 'sts:AssumeRole', 'Resource': 'arn:aws:iam::123456789012:user/JohnDoe'}]}
        arn = iam_client.create_policy(PolicyName=policy_name, PolicyDocument=dumps(policy_document))['Policy']['Arn']
        from prowler.providers.aws.services.iam.iam_service import IAM
        audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.iam.iam_no_custom_policy_permissive_role_assumption.iam_no_custom_policy_permissive_role_assumption.iam_client', new=IAM(audit_info)):
                from prowler.providers.aws.services.iam.iam_no_custom_policy_permissive_role_assumption.iam_no_custom_policy_permissive_role_assumption import iam_no_custom_policy_permissive_role_assumption
                check = iam_no_custom_policy_permissive_role_assumption()
                result = check.execute()
                assert result[0].status == 'PASS'
                assert search(f'Custom Policy {policy_name} does not allow permissive STS Role assumption', result[0].status_extended)
                assert result[0].resource_arn == arn
                assert result[0].resource_id == policy_name

    @mock_iam
    def test_policy_not_allow_permissive_role_assumption(self):
        if False:
            while True:
                i = 10
        iam_client = client('iam')
        policy_name = 'policy1'
        policy_document = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': 'logs:CreateLogGroup', 'Resource': '*'}]}
        arn = iam_client.create_policy(PolicyName=policy_name, PolicyDocument=dumps(policy_document))['Policy']['Arn']
        from prowler.providers.aws.services.iam.iam_service import IAM
        audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.iam.iam_no_custom_policy_permissive_role_assumption.iam_no_custom_policy_permissive_role_assumption.iam_client', new=IAM(audit_info)):
                from prowler.providers.aws.services.iam.iam_no_custom_policy_permissive_role_assumption.iam_no_custom_policy_permissive_role_assumption import iam_no_custom_policy_permissive_role_assumption
                check = iam_no_custom_policy_permissive_role_assumption()
                result = check.execute()
                assert result[0].status == 'PASS'
                assert search(f'Custom Policy {policy_name} does not allow permissive STS Role assumption', result[0].status_extended)
                assert result[0].resource_arn == arn
                assert result[0].resource_id == policy_name

    @mock_iam
    def test_policy_permissive_and_not_permissive(self):
        if False:
            while True:
                i = 10
        iam_client = client('iam')
        policy_name_non_permissive = 'policy1'
        policy_document_non_permissive = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': 'logs:*', 'Resource': '*'}]}
        policy_name_permissive = 'policy2'
        policy_document_permissive = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': 'sts:AssumeRole', 'Resource': '*'}]}
        arn_non_permissive = iam_client.create_policy(PolicyName=policy_name_non_permissive, PolicyDocument=dumps(policy_document_non_permissive))['Policy']['Arn']
        arn_permissive = iam_client.create_policy(PolicyName=policy_name_permissive, PolicyDocument=dumps(policy_document_permissive))['Policy']['Arn']
        from prowler.providers.aws.services.iam.iam_service import IAM
        audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.iam.iam_no_custom_policy_permissive_role_assumption.iam_no_custom_policy_permissive_role_assumption.iam_client', new=IAM(audit_info)):
                from prowler.providers.aws.services.iam.iam_no_custom_policy_permissive_role_assumption.iam_no_custom_policy_permissive_role_assumption import iam_no_custom_policy_permissive_role_assumption
                check = iam_no_custom_policy_permissive_role_assumption()
                result = check.execute()
                assert len(result) == 2
                assert result[0].status == 'PASS'
                assert result[0].resource_arn == arn_non_permissive
                assert search(f'Policy {policy_name_non_permissive} does not allow permissive STS Role assumption', result[0].status_extended)
                assert result[0].resource_id == policy_name_non_permissive
                assert result[1].status == 'FAIL'
                assert result[1].resource_arn == arn_permissive
                assert search(f'Policy {policy_name_permissive} allows permissive STS Role assumption', result[1].status_extended)
                assert result[1].resource_id == policy_name_permissive