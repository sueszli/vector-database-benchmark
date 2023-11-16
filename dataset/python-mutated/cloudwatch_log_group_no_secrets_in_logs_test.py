from re import search
from unittest import mock
from boto3 import client, session
from moto import mock_logs
from moto.core.utils import unix_time_millis
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.common.models import Audit_Metadata
AWS_REGION = 'us-east-1'
AWS_ACCOUNT_NUMBER = '123456789012'

class Test_cloudwatch_log_group_no_secrets_in_logs:

    def set_mocked_audit_info(self):
        if False:
            i = 10
            return i + 15
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=['us-east-1', 'eu-west-1'], organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    def test_cloudwatch_no_log_groups(self):
        if False:
            i = 10
            return i + 15
        from prowler.providers.aws.services.cloudwatch.cloudwatch_service import Logs
        current_audit_info = self.set_mocked_audit_info()
        from prowler.providers.common.models import Audit_Metadata
        current_audit_info.audit_metadata = Audit_Metadata(services_scanned=0, expected_checks=['cloudwatch_log_group_no_secrets_in_logs'], completed_checks=0, audit_progress=0)
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.cloudwatch.cloudwatch_log_group_no_secrets_in_logs.cloudwatch_log_group_no_secrets_in_logs.logs_client', new=Logs(current_audit_info)):
            from prowler.providers.aws.services.cloudwatch.cloudwatch_log_group_no_secrets_in_logs.cloudwatch_log_group_no_secrets_in_logs import cloudwatch_log_group_no_secrets_in_logs
            check = cloudwatch_log_group_no_secrets_in_logs()
            result = check.execute()
            assert len(result) == 0

    @mock_logs
    def test_cloudwatch_log_group_without_secrets(self):
        if False:
            print('Hello World!')
        logs_client = client('logs', region_name=AWS_REGION)
        logs_client.create_log_group(logGroupName='test')
        logs_client.create_log_stream(logGroupName='test', logStreamName='test stream')
        logs_client.put_log_events(logGroupName='test', logStreamName='test stream', logEvents=[{'timestamp': int(unix_time_millis()), 'message': 'non sensitive message'}])
        from prowler.providers.aws.services.cloudwatch.cloudwatch_service import Logs
        current_audit_info = self.set_mocked_audit_info()
        from prowler.providers.common.models import Audit_Metadata
        current_audit_info.audit_metadata = Audit_Metadata(services_scanned=0, expected_checks=['cloudwatch_log_group_no_secrets_in_logs'], completed_checks=0, audit_progress=0)
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.cloudwatch.cloudwatch_log_group_no_secrets_in_logs.cloudwatch_log_group_no_secrets_in_logs.logs_client', new=Logs(current_audit_info)):
            from prowler.providers.aws.services.cloudwatch.cloudwatch_log_group_no_secrets_in_logs.cloudwatch_log_group_no_secrets_in_logs import cloudwatch_log_group_no_secrets_in_logs
            check = cloudwatch_log_group_no_secrets_in_logs()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert result[0].status_extended == 'No secrets found in test log group.'
            assert result[0].resource_id == 'test'

    @mock_logs
    def test_cloudwatch_log_group_with_secrets(self):
        if False:
            print('Hello World!')
        logs_client = client('logs', region_name=AWS_REGION)
        logs_client.create_log_group(logGroupName='test')
        logs_client.create_log_stream(logGroupName='test', logStreamName='test stream')
        logs_client.put_log_events(logGroupName='test', logStreamName='test stream', logEvents=[{'timestamp': int(unix_time_millis()), 'message': 'password = password123'}])
        from prowler.providers.aws.services.cloudwatch.cloudwatch_service import Logs
        current_audit_info = self.set_mocked_audit_info()
        from prowler.providers.common.models import Audit_Metadata
        current_audit_info.audit_metadata = Audit_Metadata(services_scanned=0, expected_checks=['cloudwatch_log_group_no_secrets_in_logs'], completed_checks=0, audit_progress=0)
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.cloudwatch.cloudwatch_log_group_no_secrets_in_logs.cloudwatch_log_group_no_secrets_in_logs.logs_client', new=Logs(current_audit_info)):
            from prowler.providers.aws.services.cloudwatch.cloudwatch_log_group_no_secrets_in_logs.cloudwatch_log_group_no_secrets_in_logs import cloudwatch_log_group_no_secrets_in_logs
            check = cloudwatch_log_group_no_secrets_in_logs()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert search('Potential secrets found in log group', result[0].status_extended)
            assert result[0].resource_id == 'test'