from re import search
from unittest import mock
from boto3 import client, session
from moto import mock_iam
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.common.models import Audit_Metadata
AWS_ACCOUNT_NUMBER = '123456789012'

class Test_iam_user_hardware_mfa_enabled_test:

    def set_mocked_audit_info(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=['us-east-1', 'eu-west-1'], organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_iam
    def test_user_no_mfa_devices(self):
        if False:
            for i in range(10):
                print('nop')
        iam_client = client('iam')
        user = 'test-user'
        arn = iam_client.create_user(UserName=user)['User']['Arn']
        from prowler.providers.aws.services.iam.iam_service import IAM
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_user_hardware_mfa_enabled.iam_user_hardware_mfa_enabled.iam_client', new=IAM(current_audit_info)) as service_client:
            from prowler.providers.aws.services.iam.iam_user_hardware_mfa_enabled.iam_user_hardware_mfa_enabled import iam_user_hardware_mfa_enabled
            service_client.users[0].mfa_devices = []
            check = iam_user_hardware_mfa_enabled()
            result = check.execute()
            assert result[0].status == 'FAIL'
            assert search(f'User {user} does not have any type of MFA enabled.', result[0].status_extended)
            assert result[0].resource_id == user
            assert result[0].resource_arn == arn

    @mock_iam
    def test_user_virtual_mfa_devices(self):
        if False:
            print('Hello World!')
        iam_client = client('iam')
        user = 'test-user'
        arn = iam_client.create_user(UserName=user)['User']['Arn']
        from prowler.providers.aws.services.iam.iam_service import IAM, MFADevice
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_user_hardware_mfa_enabled.iam_user_hardware_mfa_enabled.iam_client', new=IAM(current_audit_info)) as service_client:
            from prowler.providers.aws.services.iam.iam_user_hardware_mfa_enabled.iam_user_hardware_mfa_enabled import iam_user_hardware_mfa_enabled
            mfa_devices = [MFADevice(serial_number='123454', type='mfa'), MFADevice(serial_number='1234547', type='sms-mfa')]
            service_client.users[0].mfa_devices = mfa_devices
            check = iam_user_hardware_mfa_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert search(f'User {user} has a virtual MFA instead of a hardware MFA device enabled.', result[0].status_extended)
            assert result[0].resource_id == user
            assert result[0].resource_arn == arn

    @mock_iam
    def test_user_virtual_sms_mfa_devices(self):
        if False:
            while True:
                i = 10
        iam_client = client('iam')
        user = 'test-user'
        arn = iam_client.create_user(UserName=user)['User']['Arn']
        from prowler.providers.aws.services.iam.iam_service import IAM, MFADevice
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_user_hardware_mfa_enabled.iam_user_hardware_mfa_enabled.iam_client', new=IAM(current_audit_info)) as service_client:
            from prowler.providers.aws.services.iam.iam_user_hardware_mfa_enabled.iam_user_hardware_mfa_enabled import iam_user_hardware_mfa_enabled
            mfa_devices = [MFADevice(serial_number='123454', type='test-mfa'), MFADevice(serial_number='1234547', type='sms-mfa')]
            service_client.users[0].mfa_devices = mfa_devices
            check = iam_user_hardware_mfa_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert search(f'User {user} has a virtual MFA instead of a hardware MFA device enabled.', result[0].status_extended)
            assert result[0].resource_id == user
            assert result[0].resource_arn == arn