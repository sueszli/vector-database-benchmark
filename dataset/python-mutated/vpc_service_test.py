import json
from boto3 import client, resource, session
from moto import mock_ec2, mock_elbv2
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.vpc.vpc_service import VPC, Route
from prowler.providers.common.models import Audit_Metadata
AWS_ACCOUNT_NUMBER = '123456789012'
AWS_REGION = 'us-east-1'

class Test_VPC_Service:

    def set_mocked_audit_info(self):
        if False:
            print('Hello World!')
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=['eu-west-1', 'us-east-1'], organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_ec2
    def test_service(self):
        if False:
            while True:
                i = 10
        audit_info = self.set_mocked_audit_info()
        vpc = VPC(audit_info)
        assert vpc.service == 'ec2'

    @mock_ec2
    def test_client(self):
        if False:
            i = 10
            return i + 15
        audit_info = self.set_mocked_audit_info()
        vpc = VPC(audit_info)
        for regional_client in vpc.regional_clients.values():
            assert regional_client.__class__.__name__ == 'EC2'

    @mock_ec2
    def test__get_session__(self):
        if False:
            while True:
                i = 10
        audit_info = self.set_mocked_audit_info()
        vpc = VPC(audit_info)
        assert vpc.session.__class__.__name__ == 'Session'

    @mock_ec2
    def test_audited_account(self):
        if False:
            return 10
        audit_info = self.set_mocked_audit_info()
        vpc = VPC(audit_info)
        assert vpc.audited_account == AWS_ACCOUNT_NUMBER

    @mock_ec2
    def test__describe_vpcs__(self):
        if False:
            return 10
        ec2_client = client('ec2', region_name=AWS_REGION)
        vpc = ec2_client.create_vpc(CidrBlock='10.0.0.0/16', TagSpecifications=[{'ResourceType': 'vpc', 'Tags': [{'Key': 'test', 'Value': 'test'}]}])['Vpc']
        audit_info = self.set_mocked_audit_info()
        vpc = VPC(audit_info)
        assert len(vpc.vpcs) == 3
        for vpc in vpc.vpcs.values():
            if vpc.cidr_block == '10.0.0.0/16':
                assert vpc.tags == [{'Key': 'test', 'Value': 'test'}]

    @mock_ec2
    def test__describe_flow_logs__(self):
        if False:
            for i in range(10):
                print('nop')
        ec2_client = client('ec2', region_name=AWS_REGION)
        new_vpc = ec2_client.create_vpc(CidrBlock='10.0.0.0/16')['Vpc']
        ec2_client.create_flow_logs(ResourceType='VPC', ResourceIds=[new_vpc['VpcId']], TrafficType='ALL', LogDestinationType='cloud-watch-logs', LogGroupName='test_logs', DeliverLogsPermissionArn='arn:aws:iam::' + str(AWS_ACCOUNT_NUMBER) + ':role/test-role')
        audit_info = self.set_mocked_audit_info()
        vpc = VPC(audit_info)
        for vpc_iter in vpc.vpcs.values():
            if vpc_iter.id == new_vpc['VpcId']:
                assert vpc_iter.flow_log is True

    @mock_ec2
    def test__describe_vpc_peering_connections__(self):
        if False:
            while True:
                i = 10
        ec2_client = client('ec2', region_name=AWS_REGION)
        vpc = ec2_client.create_vpc(CidrBlock='10.0.0.0/16')
        peer_vpc = ec2_client.create_vpc(CidrBlock='11.0.0.0/16')
        vpc_pcx = ec2_client.create_vpc_peering_connection(VpcId=vpc['Vpc']['VpcId'], PeerVpcId=peer_vpc['Vpc']['VpcId'], TagSpecifications=[{'ResourceType': 'vpc-peering-connection', 'Tags': [{'Key': 'test', 'Value': 'test'}]}])
        vpc_pcx_id = vpc_pcx['VpcPeeringConnection']['VpcPeeringConnectionId']
        vpc_pcx = ec2_client.accept_vpc_peering_connection(VpcPeeringConnectionId=vpc_pcx_id)
        audit_info = self.set_mocked_audit_info()
        vpc = VPC(audit_info)
        assert len(vpc.vpc_peering_connections) == 1
        assert vpc.vpc_peering_connections[0].id == vpc_pcx_id
        assert vpc.vpc_peering_connections[0].tags == [{'Key': 'test', 'Value': 'test'}]

    @mock_ec2
    def test__describe_route_tables__(self):
        if False:
            print('Hello World!')
        ec2_client = client('ec2', region_name=AWS_REGION)
        _ = resource('ec2', region_name=AWS_REGION)
        vpc = ec2_client.create_vpc(CidrBlock='10.0.0.0/16')
        peer_vpc = ec2_client.create_vpc(CidrBlock='11.0.0.0/16')
        vpc_pcx = ec2_client.create_vpc_peering_connection(VpcId=vpc['Vpc']['VpcId'], PeerVpcId=peer_vpc['Vpc']['VpcId'])
        vpc_pcx_id = vpc_pcx['VpcPeeringConnection']['VpcPeeringConnectionId']
        vpc_pcx = ec2_client.accept_vpc_peering_connection(VpcPeeringConnectionId=vpc_pcx_id)
        main_route_table_id = ec2_client.describe_route_tables(Filters=[{'Name': 'vpc-id', 'Values': [vpc['Vpc']['VpcId']]}, {'Name': 'association.main', 'Values': ['true']}])['RouteTables'][0]['RouteTableId']
        audit_info = self.set_mocked_audit_info()
        vpc = VPC(audit_info)
        vpc.vpc_peering_connections[0].route_tables = [Route(id=main_route_table_id, destination_cidrs=['10.0.0.4/24'])]
        assert len(vpc.vpc_peering_connections[0].route_tables) == 1
        assert vpc.vpc_peering_connections[0].id == vpc_pcx_id

    @mock_ec2
    def test__describe_vpc_endpoints__(self):
        if False:
            return 10
        ec2_client = client('ec2', region_name=AWS_REGION)
        vpc = ec2_client.create_vpc(CidrBlock='10.0.0.0/16')['Vpc']
        route_table = ec2_client.create_route_table(VpcId=vpc['VpcId'])['RouteTable']
        endpoint = ec2_client.create_vpc_endpoint(VpcId=vpc['VpcId'], ServiceName='com.amazonaws.us-east-1.s3', RouteTableIds=[route_table['RouteTableId']], VpcEndpointType='Gateway', PolicyDocument=json.dumps({'Statement': [{'Action': '*', 'Effect': 'Allow', 'Principal': '*', 'Resource': '*'}]}), TagSpecifications=[{'ResourceType': 'vpc-endpoint', 'Tags': [{'Key': 'test', 'Value': 'test'}]}])['VpcEndpoint']['VpcEndpointId']
        audit_info = self.set_mocked_audit_info()
        vpc = VPC(audit_info)
        assert len(vpc.vpc_endpoints) == 1
        assert vpc.vpc_endpoints[0].id == endpoint
        assert vpc.vpc_endpoints[0].tags == [{'Key': 'test', 'Value': 'test'}]

    @mock_ec2
    @mock_elbv2
    def test__describe_vpc_endpoint_services__(self):
        if False:
            i = 10
            return i + 15
        ec2_client = client('ec2', region_name=AWS_REGION)
        elbv2_client = client('elbv2', region_name=AWS_REGION)
        vpc = ec2_client.create_vpc(CidrBlock='172.28.7.0/24', InstanceTenancy='default')
        subnet = ec2_client.create_subnet(VpcId=vpc['Vpc']['VpcId'], CidrBlock='172.28.7.192/26', AvailabilityZone=f'{AWS_REGION}a')
        lb_name = 'lb_vpce-test'
        lb_arn = elbv2_client.create_load_balancer(Name=lb_name, Subnets=[subnet['Subnet']['SubnetId']], Scheme='internal', Type='network')['LoadBalancers'][0]['LoadBalancerArn']
        endpoint = ec2_client.create_vpc_endpoint_service_configuration(NetworkLoadBalancerArns=[lb_arn], TagSpecifications=[{'ResourceType': 'vpc-endpoint-service-configuration', 'Tags': [{'Key': 'test', 'Value': 'test'}]}])
        endpoint_id = endpoint['ServiceConfiguration']['ServiceId']
        endpoint_arn = f'arn:aws:ec2:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:vpc-endpoint-service/{endpoint_id}'
        endpoint_service = endpoint['ServiceConfiguration']['ServiceName']
        audit_info = self.set_mocked_audit_info()
        vpc = VPC(audit_info)
        for vpce in vpc.vpc_endpoint_services:
            assert vpce.arn == endpoint_arn
            assert vpce.id == endpoint_id
            assert vpce.service == endpoint_service
            assert vpce.owner_id == AWS_ACCOUNT_NUMBER
            assert vpce.allowed_principals == []
            assert vpce.region == AWS_REGION
            assert vpce.tags == []

    @mock_ec2
    def test__describe_vpc_subnets__(self):
        if False:
            for i in range(10):
                print('nop')
        ec2_client = client('ec2', region_name=AWS_REGION)
        vpc = ec2_client.create_vpc(CidrBlock='172.28.7.0/24', InstanceTenancy='default')
        subnet = ec2_client.create_subnet(VpcId=vpc['Vpc']['VpcId'], CidrBlock='172.28.7.192/26', AvailabilityZone=f'{AWS_REGION}a')
        audit_info = self.set_mocked_audit_info()
        vpc = VPC(audit_info)
        assert len(vpc.vpcs) == 3
        for vpc in vpc.vpcs.values():
            if vpc.cidr_block == '172.28.7.0/24':
                assert vpc.subnets[0].id == subnet['Subnet']['SubnetId']
                assert vpc.subnets[0].default is False
                assert vpc.subnets[0].vpc_id == vpc.id
                assert vpc.subnets[0].cidr_block == '172.28.7.192/26'
                assert vpc.subnets[0].availability_zone == f'{AWS_REGION}a'
                assert vpc.subnets[0].public is False
                assert vpc.subnets[0].nat_gateway is False
                assert vpc.subnets[0].region == AWS_REGION
                assert vpc.subnets[0].tags is None