from unittest import mock
from boto3 import client, resource, session
from moto import mock_ec2
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.common.models import Audit_Metadata
AWS_REGION = 'us-east-1'
AWS_ACCOUNT_NUMBER = '123456789012'

class Test_vpc_flow_logs_enabled:

    def set_mocked_audit_info(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=['us-east-1', 'eu-west-1'], organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_ec2
    def test_vpc_only_default_vpcs(self):
        if False:
            i = 10
            return i + 15
        from prowler.providers.aws.services.vpc.vpc_service import VPC
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.vpc.vpc_flow_logs_enabled.vpc_flow_logs_enabled.vpc_client', new=VPC(current_audit_info)):
            from prowler.providers.aws.services.vpc.vpc_flow_logs_enabled.vpc_flow_logs_enabled import vpc_flow_logs_enabled
            check = vpc_flow_logs_enabled()
            result = check.execute()
            assert len(result) == 2

    @mock_ec2
    def test_vpc_with_flow_logs(self):
        if False:
            print('Hello World!')
        from prowler.providers.aws.services.vpc.vpc_service import VPC
        ec2_client = client('ec2', region_name=AWS_REGION)
        vpc = ec2_client.create_vpc(CidrBlock='10.0.0.0/16', TagSpecifications=[{'ResourceType': 'vpc', 'Tags': [{'Key': 'Name', 'Value': 'vpc_name'}]}])['Vpc']
        ec2_client.create_flow_logs(ResourceType='VPC', ResourceIds=[vpc['VpcId']], TrafficType='ALL', LogDestinationType='cloud-watch-logs', LogGroupName='test_logs', DeliverLogsPermissionArn='arn:aws:iam::' + AWS_ACCOUNT_NUMBER + ':role/test-role')
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.vpc.vpc_flow_logs_enabled.vpc_flow_logs_enabled.vpc_client', new=VPC(current_audit_info)):
            from prowler.providers.aws.services.vpc.vpc_flow_logs_enabled.vpc_flow_logs_enabled import vpc_flow_logs_enabled
            check = vpc_flow_logs_enabled()
            result = check.execute()
            for result in result:
                if result.resource_id == vpc['VpcId']:
                    assert result.status == 'PASS'
                    assert result.status_extended == 'VPC vpc_name Flow logs are enabled.'
                    assert result.resource_id == vpc['VpcId']

    @mock_ec2
    def test_vpc_without_flow_logs(self):
        if False:
            while True:
                i = 10
        from prowler.providers.aws.services.vpc.vpc_service import VPC
        ec2_client = client('ec2', region_name=AWS_REGION)
        vpc = ec2_client.create_vpc(CidrBlock='10.0.0.0/16')['Vpc']
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.vpc.vpc_flow_logs_enabled.vpc_flow_logs_enabled.vpc_client', new=VPC(current_audit_info)):
            from prowler.providers.aws.services.vpc.vpc_flow_logs_enabled.vpc_flow_logs_enabled import vpc_flow_logs_enabled
            check = vpc_flow_logs_enabled()
            result = check.execute()
            for result in result:
                if result.resource_id == vpc['VpcId']:
                    assert result.status == 'FAIL'
                    assert result.status_extended == f"VPC {vpc['VpcId']} Flow logs are disabled."
                    assert result.resource_id == vpc['VpcId']

    @mock_ec2
    def test_vpc_without_flow_logs_ignoring(self):
        if False:
            print('Hello World!')
        from prowler.providers.aws.services.vpc.vpc_service import VPC
        ec2_client = client('ec2', region_name=AWS_REGION)
        ec2_client.create_vpc(CidrBlock='10.0.0.0/16')['Vpc']
        current_audit_info = self.set_mocked_audit_info()
        current_audit_info.ignore_unused_services = True
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.vpc.vpc_flow_logs_enabled.vpc_flow_logs_enabled.vpc_client', new=VPC(current_audit_info)):
            from prowler.providers.aws.services.vpc.vpc_flow_logs_enabled.vpc_flow_logs_enabled import vpc_flow_logs_enabled
            check = vpc_flow_logs_enabled()
            result = check.execute()
            assert len(result) == 0

    @mock_ec2
    def test_vpc_without_flow_logs_ignoring_in_use(self):
        if False:
            for i in range(10):
                print('nop')
        from prowler.providers.aws.services.vpc.vpc_service import VPC
        ec2 = resource('ec2', region_name=AWS_REGION)
        vpc = ec2.create_vpc(CidrBlock='10.0.0.0/16')
        subnet = ec2.create_subnet(VpcId=vpc.id, CidrBlock='10.0.0.0/18')
        ec2.create_network_interface(SubnetId=subnet.id)
        current_audit_info = self.set_mocked_audit_info()
        current_audit_info.ignore_unused_services = True
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.vpc.vpc_flow_logs_enabled.vpc_flow_logs_enabled.vpc_client', new=VPC(current_audit_info)):
            from prowler.providers.aws.services.vpc.vpc_flow_logs_enabled.vpc_flow_logs_enabled import vpc_flow_logs_enabled
            check = vpc_flow_logs_enabled()
            result = check.execute()
            for result in result:
                if result.resource_id == vpc.id:
                    assert result.status == 'FAIL'
                    assert result.status_extended == f'VPC {vpc.id} Flow logs are disabled.'
                    assert result.resource_id == vpc.id