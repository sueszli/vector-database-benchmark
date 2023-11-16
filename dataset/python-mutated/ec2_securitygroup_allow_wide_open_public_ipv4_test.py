from unittest import mock
from boto3 import client, session
from moto import mock_ec2
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.common.models import Audit_Metadata
AWS_REGION = 'us-east-1'
AWS_ACCOUNT_NUMBER = '123456789012'

class Test_ec2_securitygroup_allow_wide_open_public_ipv4:

    def set_mocked_audit_info(self):
        if False:
            while True:
                i = 10
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=['us-east-1', 'eu-west-1'], organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_ec2
    def test_ec2_default_sgs(self):
        if False:
            while True:
                i = 10
        ec2_client = client('ec2', region_name=AWS_REGION)
        ec2_client.create_vpc(CidrBlock='10.0.0.0/16')
        from prowler.providers.aws.services.ec2.ec2_service import EC2
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.ec2.ec2_securitygroup_allow_wide_open_public_ipv4.ec2_securitygroup_allow_wide_open_public_ipv4.ec2_client', new=EC2(current_audit_info)):
            from prowler.providers.aws.services.ec2.ec2_securitygroup_allow_wide_open_public_ipv4.ec2_securitygroup_allow_wide_open_public_ipv4 import ec2_securitygroup_allow_wide_open_public_ipv4
            check = ec2_securitygroup_allow_wide_open_public_ipv4()
            result = check.execute()
            assert len(result) == 3
            assert result[0].status == 'PASS'
            assert result[1].status == 'PASS'
            assert result[2].status == 'PASS'

    @mock_ec2
    def test_ec2_default_sg_with_RFC1918_address(self):
        if False:
            return 10
        ec2_client = client('ec2', region_name=AWS_REGION)
        ec2_client.create_vpc(CidrBlock='10.0.0.0/16')
        default_sg = ec2_client.describe_security_groups(GroupNames=['default'])['SecurityGroups'][0]
        default_sg_id = default_sg['GroupId']
        default_sg_name = default_sg['GroupName']
        ec2_client.authorize_security_group_ingress(GroupId=default_sg_id, IpPermissions=[{'IpProtocol': '-1', 'IpRanges': [{'CidrIp': '192.0.2.15/32'}]}])
        from prowler.providers.aws.services.ec2.ec2_service import EC2
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.ec2.ec2_securitygroup_allow_wide_open_public_ipv4.ec2_securitygroup_allow_wide_open_public_ipv4.ec2_client', new=EC2(current_audit_info)):
            from prowler.providers.aws.services.ec2.ec2_securitygroup_allow_wide_open_public_ipv4.ec2_securitygroup_allow_wide_open_public_ipv4 import ec2_securitygroup_allow_wide_open_public_ipv4
            check = ec2_securitygroup_allow_wide_open_public_ipv4()
            result = check.execute()
            assert len(result) == 3
            for sg in result:
                if sg.resource_id == default_sg_id:
                    assert sg.status == 'PASS'
                    assert sg.region == AWS_REGION
                    assert sg.status_extended == f'Security group {default_sg_name} ({default_sg_id}) has no potential wide-open non-RFC1918 address.'
                    assert sg.resource_arn == f'arn:{current_audit_info.audited_partition}:ec2:{AWS_REGION}:{current_audit_info.audited_account}:security-group/{default_sg_id}'
                    assert sg.resource_details == default_sg_name
                    assert sg.resource_tags == []

    @mock_ec2
    def test_ec2_default_sg_with_non_RFC1918_address(self):
        if False:
            i = 10
            return i + 15
        ec2_client = client('ec2', region_name=AWS_REGION)
        ec2_client.create_vpc(CidrBlock='10.0.0.0/16')
        default_sg = ec2_client.describe_security_groups(GroupNames=['default'])['SecurityGroups'][0]
        default_sg_id = default_sg['GroupId']
        default_sg_name = default_sg['GroupName']
        ec2_client.authorize_security_group_ingress(GroupId=default_sg_id, IpPermissions=[{'IpProtocol': '-1', 'IpRanges': [{'CidrIp': '82.122.0.0/16'}]}])
        from prowler.providers.aws.services.ec2.ec2_service import EC2
        current_audit_info = self.set_mocked_audit_info()
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.ec2.ec2_securitygroup_allow_wide_open_public_ipv4.ec2_securitygroup_allow_wide_open_public_ipv4.ec2_client', new=EC2(current_audit_info)):
            from prowler.providers.aws.services.ec2.ec2_securitygroup_allow_wide_open_public_ipv4.ec2_securitygroup_allow_wide_open_public_ipv4 import ec2_securitygroup_allow_wide_open_public_ipv4
            check = ec2_securitygroup_allow_wide_open_public_ipv4()
            result = check.execute()
            assert len(result) == 3
            for sg in result:
                if sg.resource_id == default_sg_id:
                    assert sg.status == 'FAIL'
                    assert sg.region == AWS_REGION
                    assert sg.status_extended == f'Security group {default_sg_name} ({default_sg_id}) has potential wide-open non-RFC1918 address 82.122.0.0/16 in ingress rule.'
                    assert sg.resource_arn == f'arn:{current_audit_info.audited_partition}:ec2:{AWS_REGION}:{current_audit_info.audited_account}:security-group/{default_sg_id}'
                    assert sg.resource_details == default_sg_name
                    assert sg.resource_tags == []