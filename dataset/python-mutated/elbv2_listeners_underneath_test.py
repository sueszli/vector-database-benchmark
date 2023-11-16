from re import search
from unittest import mock
from boto3 import client, resource, session
from moto import mock_ec2, mock_elbv2
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.common.models import Audit_Metadata
AWS_REGION = 'eu-west-1'
AWS_ACCOUNT_NUMBER = '123456789012'

class Test_elbv2_listeners_underneath:

    def set_mocked_audit_info(self):
        if False:
            return 10
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=['us-east-1', 'eu-west-1'], organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_elbv2
    def test_elb_no_balancers(self):
        if False:
            while True:
                i = 10
        from prowler.providers.aws.services.elbv2.elbv2_service import ELBv2
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=self.set_mocked_audit_info()), mock.patch('prowler.providers.aws.services.elbv2.elbv2_listeners_underneath.elbv2_listeners_underneath.elbv2_client', new=ELBv2(self.set_mocked_audit_info())):
            from prowler.providers.aws.services.elbv2.elbv2_listeners_underneath.elbv2_listeners_underneath import elbv2_listeners_underneath
            check = elbv2_listeners_underneath()
            result = check.execute()
            assert len(result) == 0

    @mock_ec2
    @mock_elbv2
    def test_elbv2_without_listeners(self):
        if False:
            while True:
                i = 10
        conn = client('elbv2', region_name=AWS_REGION)
        ec2 = resource('ec2', region_name=AWS_REGION)
        security_group = ec2.create_security_group(GroupName='a-security-group', Description='First One')
        vpc = ec2.create_vpc(CidrBlock='172.28.7.0/24', InstanceTenancy='default')
        subnet1 = ec2.create_subnet(VpcId=vpc.id, CidrBlock='172.28.7.192/26', AvailabilityZone=f'{AWS_REGION}a')
        subnet2 = ec2.create_subnet(VpcId=vpc.id, CidrBlock='172.28.7.0/26', AvailabilityZone=f'{AWS_REGION}b')
        lb = conn.create_load_balancer(Name='my-lb', Subnets=[subnet1.id, subnet2.id], SecurityGroups=[security_group.id], Scheme='internal', Type='application')['LoadBalancers'][0]
        from prowler.providers.aws.services.elbv2.elbv2_service import ELBv2
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=self.set_mocked_audit_info()), mock.patch('prowler.providers.aws.services.elbv2.elbv2_listeners_underneath.elbv2_listeners_underneath.elbv2_client', new=ELBv2(self.set_mocked_audit_info())):
            from prowler.providers.aws.services.elbv2.elbv2_listeners_underneath.elbv2_listeners_underneath import elbv2_listeners_underneath
            check = elbv2_listeners_underneath()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert search('has no listeners underneath', result[0].status_extended)
            assert result[0].resource_id == 'my-lb'
            assert result[0].resource_arn == lb['LoadBalancerArn']

    @mock_ec2
    @mock_elbv2
    def test_elbv2_with_listeners(self):
        if False:
            return 10
        conn = client('elbv2', region_name=AWS_REGION)
        ec2 = resource('ec2', region_name=AWS_REGION)
        security_group = ec2.create_security_group(GroupName='a-security-group', Description='First One')
        vpc = ec2.create_vpc(CidrBlock='172.28.7.0/24', InstanceTenancy='default')
        subnet1 = ec2.create_subnet(VpcId=vpc.id, CidrBlock='172.28.7.192/26', AvailabilityZone=f'{AWS_REGION}a')
        subnet2 = ec2.create_subnet(VpcId=vpc.id, CidrBlock='172.28.7.0/26', AvailabilityZone=f'{AWS_REGION}b')
        lb = conn.create_load_balancer(Name='my-lb', Subnets=[subnet1.id, subnet2.id], SecurityGroups=[security_group.id], Scheme='internal')['LoadBalancers'][0]
        response = conn.create_target_group(Name='a-target', Protocol='HTTP', Port=8080, VpcId=vpc.id, HealthCheckProtocol='HTTP', HealthCheckPort='8080', HealthCheckPath='/', HealthCheckIntervalSeconds=5, HealthCheckTimeoutSeconds=3, HealthyThresholdCount=5, UnhealthyThresholdCount=2, Matcher={'HttpCode': '200'})
        target_group = response.get('TargetGroups')[0]
        target_group_arn = target_group['TargetGroupArn']
        response = conn.create_listener(LoadBalancerArn=lb['LoadBalancerArn'], Protocol='HTTP', DefaultActions=[{'Type': 'forward', 'TargetGroupArn': target_group_arn}])
        from prowler.providers.aws.services.elbv2.elbv2_service import ELBv2
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=self.set_mocked_audit_info()), mock.patch('prowler.providers.aws.services.elbv2.elbv2_listeners_underneath.elbv2_listeners_underneath.elbv2_client', new=ELBv2(self.set_mocked_audit_info())):
            from prowler.providers.aws.services.elbv2.elbv2_listeners_underneath.elbv2_listeners_underneath import elbv2_listeners_underneath
            check = elbv2_listeners_underneath()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert search('has listeners underneath', result[0].status_extended)
            assert result[0].resource_id == 'my-lb'
            assert result[0].resource_arn == lb['LoadBalancerArn']