from unittest import mock
from boto3 import client, session
from moto import mock_ec2
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.common.models import Audit_Metadata
AWS_REGION = 'us-east-1'
AWS_ACCOUNT_NUMBER = '123456789012'

class Test_vpc_different_regions:

    def set_mocked_audit_info(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region='us-east-1', credentials=None, assumed_role_info=None, audited_regions=['us-east-1', 'eu-west-1'], organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_ec2
    def test_vpc_different_regions(self):
        if False:
            while True:
                i = 10
        ec2_client = client('ec2', region_name=AWS_REGION)
        ec2_client.create_vpc(CidrBlock='172.28.7.0/24', InstanceTenancy='default')
        ec2_client_eu = client('ec2', region_name='eu-west-1')
        ec2_client_eu.create_vpc(CidrBlock='172.28.7.0/24', InstanceTenancy='default')
        from prowler.providers.aws.services.vpc.vpc_service import VPC
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info):
            with mock.patch('prowler.providers.aws.services.vpc.vpc_different_regions.vpc_different_regions.vpc_client', new=VPC(current_audit_info)):
                from prowler.providers.aws.services.vpc.vpc_different_regions.vpc_different_regions import vpc_different_regions
                check = vpc_different_regions()
                result = check.execute()
                assert len(result) == 1
                assert result[0].status == 'PASS'
                assert result[0].region == 'us-east-1'
                assert result[0].status_extended == 'VPCs found in more than one region.'
                assert result[0].resource_id == AWS_ACCOUNT_NUMBER
                assert result[0].resource_tags == []

    @mock_ec2
    def test_vpc_only_one_regions(self):
        if False:
            i = 10
            return i + 15
        ec2_client = client('ec2', region_name=AWS_REGION)
        ec2_client.create_vpc(CidrBlock='172.28.6.0/24', InstanceTenancy='default')
        from prowler.providers.aws.services.vpc.vpc_service import VPC
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info):
            with mock.patch('prowler.providers.aws.services.vpc.vpc_different_regions.vpc_different_regions.vpc_client', new=VPC(current_audit_info)):
                from prowler.providers.aws.services.vpc.vpc_different_regions.vpc_different_regions import vpc_different_regions
                check = vpc_different_regions()
                result = check.execute()
                assert len(result) == 1
                assert result[0].status == 'FAIL'
                assert result[0].region == 'us-east-1'
                assert result[0].status_extended == 'VPCs found only in one region.'
                assert result[0].resource_id == AWS_ACCOUNT_NUMBER
                assert result[0].resource_tags == []