from os import path
from pathlib import Path
from unittest import mock
from boto3 import resource, session
from moto import mock_ec2
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.common.models import Audit_Metadata
AWS_REGION = 'us-east-1'
EXAMPLE_AMI_ID = 'ami-12c6146b'
AWS_ACCOUNT_NUMBER = '123456789012'
ACTUAL_DIRECTORY = Path(path.dirname(path.realpath(__file__)))
FIXTURES_DIR_NAME = 'fixtures'

class Test_ec2_instance_secrets_user_data:

    def set_mocked_audit_info(self):
        if False:
            while True:
                i = 10
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=['us-east-1', 'eu-west-1'], organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_ec2
    def test_no_ec2(self):
        if False:
            while True:
                i = 10
        from prowler.providers.aws.services.ec2.ec2_service import EC2
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.ec2.ec2_instance_secrets_user_data.ec2_instance_secrets_user_data.ec2_client', new=EC2(current_audit_info)):
            from prowler.providers.aws.services.ec2.ec2_instance_secrets_user_data.ec2_instance_secrets_user_data import ec2_instance_secrets_user_data
            check = ec2_instance_secrets_user_data()
            result = check.execute()
            assert len(result) == 0

    @mock_ec2
    def test_one_ec2_with_no_secrets(self):
        if False:
            while True:
                i = 10
        ec2 = resource('ec2', region_name=AWS_REGION)
        instance = ec2.create_instances(ImageId=EXAMPLE_AMI_ID, MinCount=1, MaxCount=1, UserData='This is some user_data')[0]
        from prowler.providers.aws.services.ec2.ec2_service import EC2
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.ec2.ec2_instance_secrets_user_data.ec2_instance_secrets_user_data.ec2_client', new=EC2(current_audit_info)):
            from prowler.providers.aws.services.ec2.ec2_instance_secrets_user_data.ec2_instance_secrets_user_data import ec2_instance_secrets_user_data
            check = ec2_instance_secrets_user_data()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert result[0].status_extended == f'No secrets found in EC2 instance {instance.id} User Data.'
            assert result[0].resource_id == instance.id
            assert result[0].resource_arn == f'arn:{current_audit_info.audited_partition}:ec2:{AWS_REGION}:{current_audit_info.audited_account}:instance/{instance.id}'
            assert result[0].resource_tags is None
            assert result[0].region == AWS_REGION

    @mock_ec2
    def test_one_ec2_with_secrets(self):
        if False:
            i = 10
            return i + 15
        ec2 = resource('ec2', region_name=AWS_REGION)
        instance = ec2.create_instances(ImageId=EXAMPLE_AMI_ID, MinCount=1, MaxCount=1, UserData='DB_PASSWORD=foobar123')[0]
        from prowler.providers.aws.services.ec2.ec2_service import EC2
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.ec2.ec2_instance_secrets_user_data.ec2_instance_secrets_user_data.ec2_client', new=EC2(current_audit_info)):
            from prowler.providers.aws.services.ec2.ec2_instance_secrets_user_data.ec2_instance_secrets_user_data import ec2_instance_secrets_user_data
            check = ec2_instance_secrets_user_data()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert result[0].status_extended == f'Potential secret found in EC2 instance {instance.id} User Data -> Secret Keyword on line 1.'
            assert result[0].resource_id == instance.id
            assert result[0].resource_arn == f'arn:{current_audit_info.audited_partition}:ec2:{AWS_REGION}:{current_audit_info.audited_account}:instance/{instance.id}'
            assert result[0].resource_tags is None
            assert result[0].region == AWS_REGION

    @mock_ec2
    def test_one_ec2_file_with_secrets(self):
        if False:
            return 10
        f = open(f'{ACTUAL_DIRECTORY}/{FIXTURES_DIR_NAME}/fixture', 'r')
        secrets = f.read()
        ec2 = resource('ec2', region_name=AWS_REGION)
        instance = ec2.create_instances(ImageId=EXAMPLE_AMI_ID, MinCount=1, MaxCount=1, UserData=secrets)[0]
        from prowler.providers.aws.services.ec2.ec2_service import EC2
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.ec2.ec2_instance_secrets_user_data.ec2_instance_secrets_user_data.ec2_client', new=EC2(current_audit_info)):
            from prowler.providers.aws.services.ec2.ec2_instance_secrets_user_data.ec2_instance_secrets_user_data import ec2_instance_secrets_user_data
            check = ec2_instance_secrets_user_data()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert result[0].status_extended == f'Potential secret found in EC2 instance {instance.id} User Data -> Secret Keyword on line 1, Hex High Entropy String on line 3, Secret Keyword on line 3, Secret Keyword on line 4.'
            assert result[0].resource_id == instance.id
            assert result[0].resource_arn == f'arn:{current_audit_info.audited_partition}:ec2:{AWS_REGION}:{current_audit_info.audited_account}:instance/{instance.id}'
            assert result[0].resource_tags is None
            assert result[0].region == AWS_REGION

    @mock_ec2
    def test_one_launch_configurations_without_user_data(self):
        if False:
            for i in range(10):
                print('nop')
        ec2 = resource('ec2', region_name=AWS_REGION)
        instance = ec2.create_instances(ImageId=EXAMPLE_AMI_ID, MinCount=1, MaxCount=1, UserData='')[0]
        from prowler.providers.aws.services.ec2.ec2_service import EC2
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.ec2.ec2_instance_secrets_user_data.ec2_instance_secrets_user_data.ec2_client', new=EC2(current_audit_info)):
            from prowler.providers.aws.services.ec2.ec2_instance_secrets_user_data.ec2_instance_secrets_user_data import ec2_instance_secrets_user_data
            check = ec2_instance_secrets_user_data()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert result[0].status_extended == f'No secrets found in EC2 instance {instance.id} since User Data is empty.'
            assert result[0].resource_id == instance.id
            assert result[0].resource_arn == f'arn:{current_audit_info.audited_partition}:ec2:{AWS_REGION}:{current_audit_info.audited_account}:instance/{instance.id}'
            assert result[0].resource_tags is None
            assert result[0].region == AWS_REGION

    @mock_ec2
    def test_one_ec2_file_with_secrets_gzip(self):
        if False:
            i = 10
            return i + 15
        f = open(f'{ACTUAL_DIRECTORY}/{FIXTURES_DIR_NAME}/fixture.gz', 'rb')
        secrets = f.read()
        ec2 = resource('ec2', region_name=AWS_REGION)
        instance = ec2.create_instances(ImageId=EXAMPLE_AMI_ID, MinCount=1, MaxCount=1, UserData=secrets)[0]
        from prowler.providers.aws.services.ec2.ec2_service import EC2
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.ec2.ec2_instance_secrets_user_data.ec2_instance_secrets_user_data.ec2_client', new=EC2(current_audit_info)):
            from prowler.providers.aws.services.ec2.ec2_instance_secrets_user_data.ec2_instance_secrets_user_data import ec2_instance_secrets_user_data
            check = ec2_instance_secrets_user_data()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert result[0].status_extended == f'Potential secret found in EC2 instance {instance.id} User Data -> Secret Keyword on line 1, Hex High Entropy String on line 3, Secret Keyword on line 3, Secret Keyword on line 4.'
            assert result[0].resource_id == instance.id
            assert result[0].resource_arn == f'arn:{current_audit_info.audited_partition}:ec2:{AWS_REGION}:{current_audit_info.audited_account}:instance/{instance.id}'
            assert result[0].resource_tags is None
            assert result[0].region == AWS_REGION