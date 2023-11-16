from os import path
from pathlib import Path
from unittest import mock
from boto3 import client, session
from moto import mock_autoscaling
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.common.models import Audit_Metadata
AWS_REGION = 'us-east-1'
AWS_ACCOUNT_NUMBER = '123456789012'
ACTUAL_DIRECTORY = Path(path.dirname(path.realpath(__file__)))
FIXTURES_DIR_NAME = 'fixtures'

class Test_autoscaling_find_secrets_ec2_launch_configuration:

    def set_mocked_audit_info(self):
        if False:
            while True:
                i = 10
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=['us-east-1', 'eu-west-1'], organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_autoscaling
    def test_no_autoscaling(self):
        if False:
            print('Hello World!')
        autoscaling_client = client('autoscaling', region_name=AWS_REGION)
        autoscaling_client.launch_configurations = []
        from prowler.providers.aws.services.autoscaling.autoscaling_service import AutoScaling
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.autoscaling.autoscaling_find_secrets_ec2_launch_configuration.autoscaling_find_secrets_ec2_launch_configuration.autoscaling_client', new=AutoScaling(current_audit_info)):
            from prowler.providers.aws.services.autoscaling.autoscaling_find_secrets_ec2_launch_configuration.autoscaling_find_secrets_ec2_launch_configuration import autoscaling_find_secrets_ec2_launch_configuration
            check = autoscaling_find_secrets_ec2_launch_configuration()
            result = check.execute()
            assert len(result) == 0

    @mock_autoscaling
    def test_one_autoscaling_with_no_secrets(self):
        if False:
            return 10
        launch_configuration_name = 'tester'
        autoscaling_client = client('autoscaling', region_name=AWS_REGION)
        autoscaling_client.create_launch_configuration(LaunchConfigurationName=launch_configuration_name, ImageId='ami-12c6146b', InstanceType='t1.micro', KeyName='the_keys', SecurityGroups=['default', 'default2'], UserData='This is some user_data')
        launch_configuration_arn = autoscaling_client.describe_launch_configurations(LaunchConfigurationNames=[launch_configuration_name])['LaunchConfigurations'][0]['LaunchConfigurationARN']
        from prowler.providers.aws.services.autoscaling.autoscaling_service import AutoScaling
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.autoscaling.autoscaling_find_secrets_ec2_launch_configuration.autoscaling_find_secrets_ec2_launch_configuration.autoscaling_client', new=AutoScaling(current_audit_info)):
            from prowler.providers.aws.services.autoscaling.autoscaling_find_secrets_ec2_launch_configuration.autoscaling_find_secrets_ec2_launch_configuration import autoscaling_find_secrets_ec2_launch_configuration
            check = autoscaling_find_secrets_ec2_launch_configuration()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert result[0].status_extended == f'No secrets found in autoscaling {launch_configuration_name} User Data.'
            assert result[0].resource_id == launch_configuration_name
            assert result[0].resource_arn == launch_configuration_arn
            assert result[0].region == AWS_REGION

    @mock_autoscaling
    def test_one_autoscaling_with_secrets(self):
        if False:
            while True:
                i = 10
        launch_configuration_name = 'tester'
        autoscaling_client = client('autoscaling', region_name=AWS_REGION)
        autoscaling_client.create_launch_configuration(LaunchConfigurationName=launch_configuration_name, ImageId='ami-12c6146b', InstanceType='t1.micro', KeyName='the_keys', SecurityGroups=['default', 'default2'], UserData='DB_PASSWORD=foobar123')
        launch_configuration_arn = autoscaling_client.describe_launch_configurations(LaunchConfigurationNames=[launch_configuration_name])['LaunchConfigurations'][0]['LaunchConfigurationARN']
        from prowler.providers.aws.services.autoscaling.autoscaling_service import AutoScaling
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.autoscaling.autoscaling_find_secrets_ec2_launch_configuration.autoscaling_find_secrets_ec2_launch_configuration.autoscaling_client', new=AutoScaling(current_audit_info)):
            from prowler.providers.aws.services.autoscaling.autoscaling_find_secrets_ec2_launch_configuration.autoscaling_find_secrets_ec2_launch_configuration import autoscaling_find_secrets_ec2_launch_configuration
            check = autoscaling_find_secrets_ec2_launch_configuration()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert result[0].status_extended == f'Potential secret found in autoscaling {launch_configuration_name} User Data.'
            assert result[0].resource_id == launch_configuration_name
            assert result[0].resource_arn == launch_configuration_arn
            assert result[0].region == AWS_REGION

    @mock_autoscaling
    def test_one_autoscaling_file_with_secrets(self):
        if False:
            print('Hello World!')
        f = open(f'{ACTUAL_DIRECTORY}/{FIXTURES_DIR_NAME}/fixture', 'r')
        secrets = f.read()
        launch_configuration_name = 'tester'
        autoscaling_client = client('autoscaling', region_name=AWS_REGION)
        autoscaling_client.create_launch_configuration(LaunchConfigurationName='tester', ImageId='ami-12c6146b', InstanceType='t1.micro', KeyName='the_keys', SecurityGroups=['default', 'default2'], UserData=secrets)
        launch_configuration_arn = autoscaling_client.describe_launch_configurations(LaunchConfigurationNames=[launch_configuration_name])['LaunchConfigurations'][0]['LaunchConfigurationARN']
        from prowler.providers.aws.services.autoscaling.autoscaling_service import AutoScaling
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.autoscaling.autoscaling_find_secrets_ec2_launch_configuration.autoscaling_find_secrets_ec2_launch_configuration.autoscaling_client', new=AutoScaling(current_audit_info)):
            from prowler.providers.aws.services.autoscaling.autoscaling_find_secrets_ec2_launch_configuration.autoscaling_find_secrets_ec2_launch_configuration import autoscaling_find_secrets_ec2_launch_configuration
            check = autoscaling_find_secrets_ec2_launch_configuration()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert result[0].status_extended == f'Potential secret found in autoscaling {launch_configuration_name} User Data.'
            assert result[0].resource_id == launch_configuration_name
            assert result[0].resource_arn == launch_configuration_arn
            assert result[0].region == AWS_REGION

    @mock_autoscaling
    def test_one_launch_configurations_without_user_data(self):
        if False:
            for i in range(10):
                print('nop')
        launch_configuration_name = 'tester'
        autoscaling_client = client('autoscaling', region_name=AWS_REGION)
        autoscaling_client.create_launch_configuration(LaunchConfigurationName=launch_configuration_name, ImageId='ami-12c6146b', InstanceType='t1.micro', KeyName='the_keys', SecurityGroups=['default', 'default2'])
        launch_configuration_arn = autoscaling_client.describe_launch_configurations(LaunchConfigurationNames=[launch_configuration_name])['LaunchConfigurations'][0]['LaunchConfigurationARN']
        from prowler.providers.aws.services.autoscaling.autoscaling_service import AutoScaling
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.autoscaling.autoscaling_find_secrets_ec2_launch_configuration.autoscaling_find_secrets_ec2_launch_configuration.autoscaling_client', new=AutoScaling(current_audit_info)):
            from prowler.providers.aws.services.autoscaling.autoscaling_find_secrets_ec2_launch_configuration.autoscaling_find_secrets_ec2_launch_configuration import autoscaling_find_secrets_ec2_launch_configuration
            check = autoscaling_find_secrets_ec2_launch_configuration()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert result[0].status_extended == f'No secrets found in autoscaling {launch_configuration_name} since User Data is empty.'
            assert result[0].resource_id == launch_configuration_name
            assert result[0].resource_arn == launch_configuration_arn
            assert result[0].region == AWS_REGION

    @mock_autoscaling
    def test_one_autoscaling_file_with_secrets_gzip(self):
        if False:
            return 10
        f = open(f'{ACTUAL_DIRECTORY}/{FIXTURES_DIR_NAME}/fixture.gz', 'rb')
        secrets = f.read()
        launch_configuration_name = 'tester'
        autoscaling_client = client('autoscaling', region_name=AWS_REGION)
        autoscaling_client.create_launch_configuration(LaunchConfigurationName='tester', ImageId='ami-12c6146b', InstanceType='t1.micro', KeyName='the_keys', SecurityGroups=['default', 'default2'], UserData=secrets)
        launch_configuration_arn = autoscaling_client.describe_launch_configurations(LaunchConfigurationNames=[launch_configuration_name])['LaunchConfigurations'][0]['LaunchConfigurationARN']
        from prowler.providers.aws.services.autoscaling.autoscaling_service import AutoScaling
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.autoscaling.autoscaling_find_secrets_ec2_launch_configuration.autoscaling_find_secrets_ec2_launch_configuration.autoscaling_client', new=AutoScaling(current_audit_info)):
            from prowler.providers.aws.services.autoscaling.autoscaling_find_secrets_ec2_launch_configuration.autoscaling_find_secrets_ec2_launch_configuration import autoscaling_find_secrets_ec2_launch_configuration
            check = autoscaling_find_secrets_ec2_launch_configuration()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert result[0].status_extended == f'Potential secret found in autoscaling {launch_configuration_name} User Data.'
            assert result[0].resource_id == launch_configuration_name
            assert result[0].resource_arn == launch_configuration_arn
            assert result[0].region == AWS_REGION