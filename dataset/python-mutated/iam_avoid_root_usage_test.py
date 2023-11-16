import datetime
from csv import DictReader
from re import search
from unittest import mock
from boto3 import session
from moto import mock_iam
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.common.models import Audit_Metadata
AWS_ACCOUNT_NUMBER = '123456789012'
AWS_REGION = 'us-east-1'

class Test_iam_avoid_root_usage:

    def set_mocked_audit_info(self):
        if False:
            return 10
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None, region_name=AWS_REGION), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=AWS_REGION, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_iam
    def test_root_not_used(self):
        if False:
            i = 10
            return i + 15
        raw_credential_report = 'user,arn,user_creation_time,password_enabled,password_last_used,password_last_changed,password_next_rotation,mfa_active,access_key_1_active,access_key_1_last_rotated,access_key_1_last_used_date,access_key_1_last_used_region,access_key_1_last_used_service,access_key_2_active,access_key_2_last_rotated,access_key_2_last_used_date,access_key_2_last_used_region,access_key_2_last_used_service,cert_1_active,cert_1_last_rotated,cert_2_active,cert_2_last_rotated\n<root_account>,arn:aws:iam::123456789012:<root_account>,2022-04-17T14:59:38+00:00,true,no_information,not_supported,not_supported,false,true,N/A,N/A,N/A,N/A,false,N/A,N/A,N/A,N/A,false,N/A,false,N/A'
        credential_lines = raw_credential_report.split('\n')
        csv_reader = DictReader(credential_lines, delimiter=',')
        credential_list = list(csv_reader)
        from prowler.providers.aws.services.iam.iam_service import IAM
        audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.iam.iam_avoid_root_usage.iam_avoid_root_usage.iam_client', new=IAM(audit_info)) as service_client:
                from prowler.providers.aws.services.iam.iam_avoid_root_usage.iam_avoid_root_usage import iam_avoid_root_usage
                service_client.credential_report = credential_list
                check = iam_avoid_root_usage()
                result = check.execute()
                assert result[0].status == 'PASS'
                assert search("Root user in the account wasn't accessed in the last", result[0].status_extended)
                assert result[0].resource_id == '<root_account>'
                assert result[0].resource_arn == 'arn:aws:iam::123456789012:<root_account>'

    @mock_iam
    def test_root_password_recently_used(self):
        if False:
            return 10
        password_last_used = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S+00:00')
        raw_credential_report = f'user,arn,user_creation_time,password_enabled,password_last_used,password_last_changed,password_next_rotation,mfa_active,access_key_1_active,access_key_1_last_rotated,access_key_1_last_used_date,access_key_1_last_used_region,access_key_1_last_used_service,access_key_2_active,access_key_2_last_rotated,access_key_2_last_used_date,access_key_2_last_used_region,access_key_2_last_used_service,cert_1_active,cert_1_last_rotated,cert_2_active,cert_2_last_rotated\n<root_account>,arn:aws:iam::123456789012:<root_account>,2022-04-17T14:59:38+00:00,true,{password_last_used},not_supported,not_supported,false,true,N/A,N/A,N/A,N/A,false,N/A,N/A,N/A,N/A,false,N/A,false,N/A'
        credential_lines = raw_credential_report.split('\n')
        csv_reader = DictReader(credential_lines, delimiter=',')
        credential_list = list(csv_reader)
        from prowler.providers.aws.services.iam.iam_service import IAM
        audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.iam.iam_avoid_root_usage.iam_avoid_root_usage.iam_client', new=IAM(audit_info)) as service_client:
                from prowler.providers.aws.services.iam.iam_avoid_root_usage.iam_avoid_root_usage import iam_avoid_root_usage
                service_client.credential_report = credential_list
                check = iam_avoid_root_usage()
                result = check.execute()
                assert result[0].status == 'FAIL'
                assert search('Root user in the account was last accessed', result[0].status_extended)
                assert result[0].resource_id == '<root_account>'
                assert result[0].resource_arn == 'arn:aws:iam::123456789012:<root_account>'

    @mock_iam
    def test_root_access_key_1_recently_used(self):
        if False:
            return 10
        access_key_1_last_used = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S+00:00')
        raw_credential_report = f'user,arn,user_creation_time,password_enabled,password_last_used,password_last_changed,password_next_rotation,mfa_active,access_key_1_active,access_key_1_last_rotated,access_key_1_last_used_date,access_key_1_last_used_region,access_key_1_last_used_service,access_key_2_active,access_key_2_last_rotated,access_key_2_last_used_date,access_key_2_last_used_region,access_key_2_last_used_service,cert_1_active,cert_1_last_rotated,cert_2_active,cert_2_last_rotated\n<root_account>,arn:aws:iam::123456789012:<root_account>,2022-04-17T14:59:38+00:00,true,no_information,not_supported,not_supported,false,true,N/A,{access_key_1_last_used},N/A,N/A,false,N/A,N/A,N/A,N/A,false,N/A,false,N/A'
        credential_lines = raw_credential_report.split('\n')
        csv_reader = DictReader(credential_lines, delimiter=',')
        credential_list = list(csv_reader)
        from prowler.providers.aws.services.iam.iam_service import IAM
        audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.iam.iam_avoid_root_usage.iam_avoid_root_usage.iam_client', new=IAM(audit_info)) as service_client:
                from prowler.providers.aws.services.iam.iam_avoid_root_usage.iam_avoid_root_usage import iam_avoid_root_usage
                service_client.credential_report = credential_list
                check = iam_avoid_root_usage()
                result = check.execute()
                assert result[0].status == 'FAIL'
                assert search('Root user in the account was last accessed', result[0].status_extended)
                assert result[0].resource_id == '<root_account>'
                assert result[0].resource_arn == 'arn:aws:iam::123456789012:<root_account>'

    @mock_iam
    def test_root_access_key_2_recently_used(self):
        if False:
            i = 10
            return i + 15
        access_key_2_last_used = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S+00:00')
        raw_credential_report = f'user,arn,user_creation_time,password_enabled,password_last_used,password_last_changed,password_next_rotation,mfa_active,access_key_1_active,access_key_1_last_rotated,access_key_1_last_used_date,access_key_1_last_used_region,access_key_1_last_used_service,access_key_2_active,access_key_2_last_rotated,access_key_2_last_used_date,access_key_2_last_used_region,access_key_2_last_used_service,cert_1_active,cert_1_last_rotated,cert_2_active,cert_2_last_rotated\n<root_account>,arn:aws:iam::123456789012:<root_account>,2022-04-17T14:59:38+00:00,true,no_information,not_supported,not_supported,false,true,N/A,N/A,N/A,N/A,false,N/A,{access_key_2_last_used},N/A,N/A,false,N/A,false,N/A'
        credential_lines = raw_credential_report.split('\n')
        csv_reader = DictReader(credential_lines, delimiter=',')
        credential_list = list(csv_reader)
        from prowler.providers.aws.services.iam.iam_service import IAM
        audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.iam.iam_avoid_root_usage.iam_avoid_root_usage.iam_client', new=IAM(audit_info)) as service_client:
                from prowler.providers.aws.services.iam.iam_avoid_root_usage.iam_avoid_root_usage import iam_avoid_root_usage
                service_client.credential_report = credential_list
                check = iam_avoid_root_usage()
                result = check.execute()
                assert result[0].status == 'FAIL'
                assert search('Root user in the account was last accessed', result[0].status_extended)
                assert result[0].resource_id == '<root_account>'
                assert result[0].resource_arn == 'arn:aws:iam::123456789012:<root_account>'

    @mock_iam
    def test_root_password_used(self):
        if False:
            i = 10
            return i + 15
        password_last_used = (datetime.datetime.now() - datetime.timedelta(days=100)).strftime('%Y-%m-%dT%H:%M:%S+00:00')
        raw_credential_report = f'user,arn,user_creation_time,password_enabled,password_last_used,password_last_changed,password_next_rotation,mfa_active,access_key_1_active,access_key_1_last_rotated,access_key_1_last_used_date,access_key_1_last_used_region,access_key_1_last_used_service,access_key_2_active,access_key_2_last_rotated,access_key_2_last_used_date,access_key_2_last_used_region,access_key_2_last_used_service,cert_1_active,cert_1_last_rotated,cert_2_active,cert_2_last_rotated\n<root_account>,arn:aws:iam::123456789012:<root_account>,2022-04-17T14:59:38+00:00,true,{password_last_used},not_supported,not_supported,false,true,N/A,N/A,N/A,N/A,false,N/A,N/A,N/A,N/A,false,N/A,false,N/A'
        credential_lines = raw_credential_report.split('\n')
        csv_reader = DictReader(credential_lines, delimiter=',')
        credential_list = list(csv_reader)
        from prowler.providers.aws.services.iam.iam_service import IAM
        audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.iam.iam_avoid_root_usage.iam_avoid_root_usage.iam_client', new=IAM(audit_info)) as service_client:
                from prowler.providers.aws.services.iam.iam_avoid_root_usage.iam_avoid_root_usage import iam_avoid_root_usage
                service_client.credential_report = credential_list
                check = iam_avoid_root_usage()
                result = check.execute()
                assert result[0].status == 'PASS'
                assert search("Root user in the account wasn't accessed in the last 1 days", result[0].status_extended)
                assert result[0].resource_id == '<root_account>'
                assert result[0].resource_arn == 'arn:aws:iam::123456789012:<root_account>'

    @mock_iam
    def test_root_access_key_1_used(self):
        if False:
            i = 10
            return i + 15
        access_key_1_last_used = (datetime.datetime.now() - datetime.timedelta(days=100)).strftime('%Y-%m-%dT%H:%M:%S+00:00')
        raw_credential_report = f'user,arn,user_creation_time,password_enabled,password_last_used,password_last_changed,password_next_rotation,mfa_active,access_key_1_active,access_key_1_last_rotated,access_key_1_last_used_date,access_key_1_last_used_region,access_key_1_last_used_service,access_key_2_active,access_key_2_last_rotated,access_key_2_last_used_date,access_key_2_last_used_region,access_key_2_last_used_service,cert_1_active,cert_1_last_rotated,cert_2_active,cert_2_last_rotated\n<root_account>,arn:aws:iam::123456789012:<root_account>,2022-04-17T14:59:38+00:00,true,no_information,not_supported,not_supported,false,true,N/A,{access_key_1_last_used},N/A,N/A,false,N/A,N/A,N/A,N/A,false,N/A,false,N/A'
        credential_lines = raw_credential_report.split('\n')
        csv_reader = DictReader(credential_lines, delimiter=',')
        credential_list = list(csv_reader)
        from prowler.providers.aws.services.iam.iam_service import IAM
        audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.iam.iam_avoid_root_usage.iam_avoid_root_usage.iam_client', new=IAM(audit_info)) as service_client:
                from prowler.providers.aws.services.iam.iam_avoid_root_usage.iam_avoid_root_usage import iam_avoid_root_usage
                service_client.credential_report = credential_list
                check = iam_avoid_root_usage()
                result = check.execute()
                assert result[0].status == 'PASS'
                assert search("Root user in the account wasn't accessed in the last 1 days", result[0].status_extended)
                assert result[0].resource_id == '<root_account>'
                assert result[0].resource_arn == 'arn:aws:iam::123456789012:<root_account>'

    @mock_iam
    def test_root_access_key_2_used(self):
        if False:
            while True:
                i = 10
        access_key_2_last_used = (datetime.datetime.now() - datetime.timedelta(days=100)).strftime('%Y-%m-%dT%H:%M:%S+00:00')
        raw_credential_report = f'user,arn,user_creation_time,password_enabled,password_last_used,password_last_changed,password_next_rotation,mfa_active,access_key_1_active,access_key_1_last_rotated,access_key_1_last_used_date,access_key_1_last_used_region,access_key_1_last_used_service,access_key_2_active,access_key_2_last_rotated,access_key_2_last_used_date,access_key_2_last_used_region,access_key_2_last_used_service,cert_1_active,cert_1_last_rotated,cert_2_active,cert_2_last_rotated\n<root_account>,arn:aws:iam::123456789012:<root_account>,2022-04-17T14:59:38+00:00,true,no_information,not_supported,not_supported,false,true,N/A,N/A,N/A,N/A,false,N/A,{access_key_2_last_used},N/A,N/A,false,N/A,false,N/A'
        credential_lines = raw_credential_report.split('\n')
        csv_reader = DictReader(credential_lines, delimiter=',')
        credential_list = list(csv_reader)
        from prowler.providers.aws.services.iam.iam_service import IAM
        audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=audit_info):
            with mock.patch('prowler.providers.aws.services.iam.iam_avoid_root_usage.iam_avoid_root_usage.iam_client', new=IAM(audit_info)) as service_client:
                from prowler.providers.aws.services.iam.iam_avoid_root_usage.iam_avoid_root_usage import iam_avoid_root_usage
                service_client.credential_report = credential_list
                check = iam_avoid_root_usage()
                result = check.execute()
                assert result[0].status == 'PASS'
                assert search("Root user in the account wasn't accessed in the last 1 days", result[0].status_extended)
                assert result[0].resource_id == '<root_account>'
                assert result[0].resource_arn == 'arn:aws:iam::123456789012:<root_account>'