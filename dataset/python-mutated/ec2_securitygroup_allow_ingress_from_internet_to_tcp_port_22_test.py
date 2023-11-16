from re import search
from unittest import mock
from boto3 import client, resource, session
from moto import mock_ec2
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.vpc.vpc_service import VPC
from prowler.providers.common.models import Audit_Metadata
AWS_REGION = 'us-east-1'
AWS_ACCOUNT_NUMBER = '123456789012'

class Test_ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22:

    def set_mocked_audit_info(self):
        if False:
            print('Hello World!')
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=['us-east-1', 'eu-west-1'], organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_ec2
    def test_ec2_default_sgs(self):
        if False:
            i = 10
            return i + 15
        ec2_client = client('ec2', region_name=AWS_REGION)
        ec2_client.create_vpc(CidrBlock='10.0.0.0/16')
        from prowler.providers.aws.services.ec2.ec2_service import EC2
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.ec2.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.ec2_client', new=EC2(current_audit_info)), mock.patch('prowler.providers.aws.services.ec2.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.vpc_client', new=VPC(current_audit_info)):
            from prowler.providers.aws.services.ec2.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22 import ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22
            check = ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22()
            result = check.execute()
            assert len(result) == 3
            assert result[0].status == 'PASS'
            assert result[1].status == 'PASS'
            assert result[2].status == 'PASS'

    @mock_ec2
    def test_ec2_non_compliant_default_sg(self):
        if False:
            return 10
        ec2_client = client('ec2', region_name=AWS_REGION)
        ec2_client.create_vpc(CidrBlock='10.0.0.0/16')
        default_sg = ec2_client.describe_security_groups(GroupNames=['default'])['SecurityGroups'][0]
        default_sg_id = default_sg['GroupId']
        default_sg_name = default_sg['GroupName']
        ec2_client.authorize_security_group_ingress(GroupId=default_sg_id, IpPermissions=[{'IpProtocol': 'tcp', 'FromPort': 22, 'ToPort': 22, 'IpRanges': [{'CidrIp': '0.0.0.0/0'}]}])
        from prowler.providers.aws.services.ec2.ec2_service import EC2
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.ec2.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.ec2_client', new=EC2(current_audit_info)), mock.patch('prowler.providers.aws.services.ec2.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.vpc_client', new=VPC(current_audit_info)):
            from prowler.providers.aws.services.ec2.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22 import ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22
            check = ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22()
            result = check.execute()
            assert len(result) == 3
            for sg in result:
                if sg.resource_id == default_sg_id:
                    assert sg.status == 'FAIL'
                    assert sg.region == AWS_REGION
                    assert sg.status_extended == f'Security group {default_sg_name} ({default_sg_id}) has SSH port 22 open to the Internet.'
                    assert search('has SSH port 22 open to the Internet', sg.status_extended)
                    assert sg.resource_arn == f'arn:{current_audit_info.audited_partition}:ec2:{AWS_REGION}:{current_audit_info.audited_account}:security-group/{default_sg_id}'
                    assert sg.resource_details == default_sg_name
                    assert sg.resource_tags == []

    @mock_ec2
    def test_ec2_compliant_default_sg(self):
        if False:
            while True:
                i = 10
        ec2_client = client('ec2', region_name=AWS_REGION)
        ec2_client.create_vpc(CidrBlock='10.0.0.0/16')
        default_sg = ec2_client.describe_security_groups(GroupNames=['default'])['SecurityGroups'][0]
        default_sg_id = default_sg['GroupId']
        default_sg_name = default_sg['GroupName']
        ec2_client.authorize_security_group_ingress(GroupId=default_sg_id, IpPermissions=[{'IpProtocol': 'tcp', 'FromPort': 22, 'ToPort': 22, 'IpRanges': [{'CidrIp': '123.123.123.123/32'}]}])
        from prowler.providers.aws.services.ec2.ec2_service import EC2
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.ec2.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.ec2_client', new=EC2(current_audit_info)), mock.patch('prowler.providers.aws.services.ec2.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.vpc_client', new=VPC(current_audit_info)):
            from prowler.providers.aws.services.ec2.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22 import ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22
            check = ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22()
            result = check.execute()
            assert len(result) == 3
            for sg in result:
                if sg.resource_id == default_sg_id:
                    assert sg.status == 'PASS'
                    assert sg.region == AWS_REGION
                    assert sg.status_extended == f'Security group {default_sg_name} ({default_sg_id}) does not have SSH port 22 open to the Internet.'
                    assert sg.resource_arn == f'arn:{current_audit_info.audited_partition}:ec2:{AWS_REGION}:{current_audit_info.audited_account}:security-group/{default_sg_id}'
                    assert sg.resource_details == default_sg_name
                    assert sg.resource_tags == []

    @mock_ec2
    def test_ec2_default_sgs_ignoring(self):
        if False:
            i = 10
            return i + 15
        ec2_client = client('ec2', region_name=AWS_REGION)
        ec2_client.create_vpc(CidrBlock='10.0.0.0/16')
        from prowler.providers.aws.services.ec2.ec2_service import EC2
        current_audit_info = self.set_mocked_audit_info()
        current_audit_info.ignore_unused_services = True
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.ec2.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.ec2_client', new=EC2(current_audit_info)), mock.patch('prowler.providers.aws.services.ec2.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.vpc_client', new=VPC(current_audit_info)):
            from prowler.providers.aws.services.ec2.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22 import ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22
            check = ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22()
            result = check.execute()
            assert len(result) == 0

    @mock_ec2
    def test_ec2_default_sgs_ignoring_vpc_in_use(self):
        if False:
            return 10
        ec2 = resource('ec2', region_name=AWS_REGION)
        vpc = ec2.create_vpc(CidrBlock='10.0.0.0/16')
        subnet = ec2.create_subnet(VpcId=vpc.id, CidrBlock='10.0.0.0/18')
        ec2.create_network_interface(SubnetId=subnet.id)
        ec2_client = client('ec2', region_name=AWS_REGION)
        default_sg = ec2_client.describe_security_groups(GroupNames=['default'])['SecurityGroups'][0]
        default_sg['GroupId']
        default_sg['GroupName']
        from prowler.providers.aws.services.ec2.ec2_service import EC2
        current_audit_info = self.set_mocked_audit_info()
        current_audit_info.ignore_unused_services = True
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.ec2.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.ec2_client', new=EC2(current_audit_info)), mock.patch('prowler.providers.aws.services.ec2.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.vpc_client', new=VPC(current_audit_info)):
            from prowler.providers.aws.services.ec2.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22.ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22 import ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22
            check = ec2_securitygroup_allow_ingress_from_internet_to_tcp_port_22()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert result[0].region == AWS_REGION