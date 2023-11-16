from unittest import mock
from boto3 import client, session
from moto import mock_logs
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.common.models import Audit_Metadata
AWS_REGION = 'us-east-1'
AWS_ACCOUNT_NUMBER = '123456789012'

class Test_cloudwatch_log_group_retention_policy_specific_days_enabled:

    def set_mocked_audit_info(self):
        if False:
            return 10
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=['us-east-1', 'eu-west-1'], organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    def test_cloudwatch_no_log_groups(self):
        if False:
            i = 10
            return i + 15
        from prowler.providers.aws.services.cloudwatch.cloudwatch_service import Logs
        current_audit_info = self.set_mocked_audit_info()
        current_audit_info.audit_config = {'log_group_retention_days': 365}
        from prowler.providers.common.models import Audit_Metadata
        current_audit_info.audit_metadata = Audit_Metadata(services_scanned=0, expected_checks=['cloudwatch_log_group_no_secrets_in_logs'], completed_checks=0, audit_progress=0)
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.cloudwatch.cloudwatch_log_group_retention_policy_specific_days_enabled.cloudwatch_log_group_retention_policy_specific_days_enabled.logs_client', new=Logs(current_audit_info)):
            from prowler.providers.aws.services.cloudwatch.cloudwatch_log_group_retention_policy_specific_days_enabled.cloudwatch_log_group_retention_policy_specific_days_enabled import cloudwatch_log_group_retention_policy_specific_days_enabled
            check = cloudwatch_log_group_retention_policy_specific_days_enabled()
            result = check.execute()
            assert len(result) == 0

    @mock_logs
    def test_cloudwatch_log_group_without_retention_days_never_expires(self):
        if False:
            print('Hello World!')
        logs_client = client('logs', region_name=AWS_REGION)
        logs_client.create_log_group(logGroupName='test')
        from prowler.providers.aws.services.cloudwatch.cloudwatch_service import Logs
        current_audit_info = self.set_mocked_audit_info()
        current_audit_info.audit_config = {'log_group_retention_days': 365}
        from prowler.providers.common.models import Audit_Metadata
        current_audit_info.audit_metadata = Audit_Metadata(services_scanned=0, expected_checks=['cloudwatch_log_group_no_secrets_in_logs'], completed_checks=0, audit_progress=0)
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.cloudwatch.cloudwatch_log_group_retention_policy_specific_days_enabled.cloudwatch_log_group_retention_policy_specific_days_enabled.logs_client', new=Logs(current_audit_info)):
            from prowler.providers.aws.services.cloudwatch.cloudwatch_log_group_retention_policy_specific_days_enabled.cloudwatch_log_group_retention_policy_specific_days_enabled import cloudwatch_log_group_retention_policy_specific_days_enabled
            check = cloudwatch_log_group_retention_policy_specific_days_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert result[0].status_extended == 'Log Group test comply with 365 days retention period since it never expires.'
            assert result[0].resource_id == 'test'
            assert result[0].resource_arn == f'arn:aws:logs:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:log-group:test'
            assert result[0].region == AWS_REGION

    @mock_logs
    def test_cloudwatch_log_group_with_compliant_retention_days(self):
        if False:
            print('Hello World!')
        logs_client = client('logs', region_name=AWS_REGION)
        logs_client.create_log_group(logGroupName='test')
        logs_client.put_retention_policy(logGroupName='test', retentionInDays=400)
        from prowler.providers.aws.services.cloudwatch.cloudwatch_service import Logs
        current_audit_info = self.set_mocked_audit_info()
        current_audit_info.audit_config = {'log_group_retention_days': 365}
        from prowler.providers.common.models import Audit_Metadata
        current_audit_info.audit_metadata = Audit_Metadata(services_scanned=0, expected_checks=['cloudwatch_log_group_no_secrets_in_logs'], completed_checks=0, audit_progress=0)
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.cloudwatch.cloudwatch_log_group_retention_policy_specific_days_enabled.cloudwatch_log_group_retention_policy_specific_days_enabled.logs_client', new=Logs(current_audit_info)):
            from prowler.providers.aws.services.cloudwatch.cloudwatch_log_group_retention_policy_specific_days_enabled.cloudwatch_log_group_retention_policy_specific_days_enabled import cloudwatch_log_group_retention_policy_specific_days_enabled
            check = cloudwatch_log_group_retention_policy_specific_days_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert result[0].status_extended == 'Log Group test comply with 365 days retention period since it has 400 days.'
            assert result[0].resource_id == 'test'
            assert result[0].resource_arn == f'arn:aws:logs:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:log-group:test'
            assert result[0].region == AWS_REGION

    @mock_logs
    def test_cloudwatch_log_group_with_no_compliant_retention_days(self):
        if False:
            i = 10
            return i + 15
        logs_client = client('logs', region_name=AWS_REGION)
        logs_client.create_log_group(logGroupName='test')
        logs_client.put_retention_policy(logGroupName='test', retentionInDays=7)
        from prowler.providers.aws.services.cloudwatch.cloudwatch_service import Logs
        current_audit_info = self.set_mocked_audit_info()
        current_audit_info.audit_config = {'log_group_retention_days': 365}
        from prowler.providers.common.models import Audit_Metadata
        current_audit_info.audit_metadata = Audit_Metadata(services_scanned=0, expected_checks=['cloudwatch_log_group_no_secrets_in_logs'], completed_checks=0, audit_progress=0)
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.cloudwatch.cloudwatch_log_group_retention_policy_specific_days_enabled.cloudwatch_log_group_retention_policy_specific_days_enabled.logs_client', new=Logs(current_audit_info)):
            from prowler.providers.aws.services.cloudwatch.cloudwatch_log_group_retention_policy_specific_days_enabled.cloudwatch_log_group_retention_policy_specific_days_enabled import cloudwatch_log_group_retention_policy_specific_days_enabled
            check = cloudwatch_log_group_retention_policy_specific_days_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert result[0].status_extended == 'Log Group test has less than 365 days retention period (7 days).'
            assert result[0].resource_id == 'test'
            assert result[0].resource_arn == f'arn:aws:logs:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:log-group:test'
            assert result[0].region == AWS_REGION