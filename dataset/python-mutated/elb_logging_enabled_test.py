from re import search
from unittest import mock
from boto3 import client, resource, session
from moto import mock_ec2, mock_elb
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.common.models import Audit_Metadata
AWS_REGION = 'eu-west-1'
AWS_ACCOUNT_NUMBER = '123456789012'
elb_arn = f'arn:aws:elasticloadbalancing:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:loadbalancer/my-lb'

class Test_elb_logging_enabled:

    def set_mocked_audit_info(self):
        if False:
            i = 10
            return i + 15
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=['us-east-1', 'eu-west-1'], organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_elb
    def test_elb_no_balancers(self):
        if False:
            while True:
                i = 10
        from prowler.providers.aws.services.elb.elb_service import ELB
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=self.set_mocked_audit_info()), mock.patch('prowler.providers.aws.services.elb.elb_logging_enabled.elb_logging_enabled.elb_client', new=ELB(self.set_mocked_audit_info())):
            from prowler.providers.aws.services.elb.elb_logging_enabled.elb_logging_enabled import elb_logging_enabled
            check = elb_logging_enabled()
            result = check.execute()
            assert len(result) == 0

    @mock_ec2
    @mock_elb
    def test_elb_without_access_log(self):
        if False:
            i = 10
            return i + 15
        elb = client('elb', region_name=AWS_REGION)
        ec2 = resource('ec2', region_name=AWS_REGION)
        security_group = ec2.create_security_group(GroupName='sg01', Description='Test security group sg01')
        elb.create_load_balancer(LoadBalancerName='my-lb', Listeners=[{'Protocol': 'tcp', 'LoadBalancerPort': 80, 'InstancePort': 8080}, {'Protocol': 'http', 'LoadBalancerPort': 81, 'InstancePort': 9000}], AvailabilityZones=[f'{AWS_REGION}a'], Scheme='internal', SecurityGroups=[security_group.id])
        from prowler.providers.aws.services.elb.elb_service import ELB
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=self.set_mocked_audit_info()), mock.patch('prowler.providers.aws.services.elb.elb_logging_enabled.elb_logging_enabled.elb_client', new=ELB(self.set_mocked_audit_info())):
            from prowler.providers.aws.services.elb.elb_logging_enabled.elb_logging_enabled import elb_logging_enabled
            check = elb_logging_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert search('does not have access logs configured', result[0].status_extended)
            assert result[0].resource_id == 'my-lb'
            assert result[0].resource_arn == elb_arn

    @mock_ec2
    @mock_elb
    def test_elb_with_deletion_protection(self):
        if False:
            i = 10
            return i + 15
        elb = client('elb', region_name=AWS_REGION)
        ec2 = resource('ec2', region_name=AWS_REGION)
        security_group = ec2.create_security_group(GroupName='sg01', Description='Test security group sg01')
        elb.create_load_balancer(LoadBalancerName='my-lb', Listeners=[{'Protocol': 'tcp', 'LoadBalancerPort': 80, 'InstancePort': 8080}, {'Protocol': 'http', 'LoadBalancerPort': 81, 'InstancePort': 9000}], AvailabilityZones=[f'{AWS_REGION}a'], Scheme='internal', SecurityGroups=[security_group.id])
        elb.modify_load_balancer_attributes(LoadBalancerName='my-lb', LoadBalancerAttributes={'AccessLog': {'Enabled': True, 'S3BucketName': 'mb', 'EmitInterval': 42, 'S3BucketPrefix': 's3bf'}})
        from prowler.providers.aws.services.elb.elb_service import ELB
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=self.set_mocked_audit_info()), mock.patch('prowler.providers.aws.services.elb.elb_logging_enabled.elb_logging_enabled.elb_client', new=ELB(self.set_mocked_audit_info())):
            from prowler.providers.aws.services.elb.elb_logging_enabled.elb_logging_enabled import elb_logging_enabled
            check = elb_logging_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert search('has access logs to S3 configured', result[0].status_extended)
            assert result[0].resource_id == 'my-lb'
            assert result[0].resource_arn == elb_arn