from __future__ import annotations
from ipaddress import IPv4Network
import boto3
from airflow.decorators import task
from airflow.utils.trigger_rule import TriggerRule

def _get_next_available_cidr(vpc_id: str) -> str:
    if False:
        i = 10
        return i + 15
    'Checks the CIDR blocks of existing subnets and attempts to extrapolate the next available block.'
    error_msg_template = 'Can not calculate the next available CIDR block: {}'
    vpc_filter = {'Name': 'vpc-id', 'Values': [vpc_id]}
    existing_subnets = boto3.client('ec2').describe_subnets(Filters=[vpc_filter])['Subnets']
    if not existing_subnets:
        raise ValueError(error_msg_template.format('No subnets are found on the provided VPC.'))
    existing_cidr_blocks = [IPv4Network(subnet['CidrBlock']) for subnet in existing_subnets]
    if len({block.prefixlen for block in existing_cidr_blocks}) > 1:
        raise ValueError(error_msg_template.format('Subnets do not all use the same CIDR block size.'))
    last_used_block = max(existing_cidr_blocks)
    (*_, last_reserved_ip) = last_used_block
    return f'{last_reserved_ip + 1}/{last_used_block.prefixlen}'

@task
def get_default_vpc_id() -> str:
    if False:
        return 10
    "Returns the VPC ID of the account's default VPC."
    filters = [{'Name': 'is-default', 'Values': ['true']}]
    return boto3.client('ec2').describe_vpcs(Filters=filters)['Vpcs'][0]['VpcId']

@task
def create_address_allocation():
    if False:
        print('Hello World!')
    'Allocate a new IP address'
    return boto3.client('ec2').allocate_address()['AllocationId']

@task
def create_nat_gateway(allocation_id: str, subnet_id: str):
    if False:
        i = 10
        return i + 15
    'Create a NAT gateway'
    client = boto3.client('ec2')
    nat_gateway_id = client.create_nat_gateway(AllocationId=allocation_id, SubnetId=subnet_id, ConnectivityType='public')['NatGateway']['NatGatewayId']
    waiter = client.get_waiter('nat_gateway_available')
    waiter.wait(NatGatewayIds=[nat_gateway_id])
    return nat_gateway_id

@task
def create_route_table(vpc_id: str, nat_gateway_id: str, test_name: str):
    if False:
        for i in range(10):
            print('nop')
    'Create a route table for private subnets.'
    client = boto3.client('ec2')
    tags = [{'Key': 'Name', 'Value': f'Route table for {test_name}'}]
    route_table_id = client.create_route_table(VpcId=vpc_id, TagSpecifications=[{'ResourceType': 'route-table', 'Tags': tags}])['RouteTable']['RouteTableId']
    client.create_route(RouteTableId=route_table_id, DestinationCidrBlock='0.0.0.0/0', NatGatewayId=nat_gateway_id)
    return route_table_id

@task
def create_private_subnets(vpc_id: str, route_table_id: str, test_name: str, number_to_make: int=1, cidr_block: str | None=None):
    if False:
        i = 10
        return i + 15
    '\n    Fargate Profiles require two private subnets in two different availability zones.\n    These subnets require as well an egress route to the internet, using a NAT gateway to achieve this.\n    '
    client = boto3.client('ec2')
    subnet_ids = []
    tags = [{'Key': 'Name', 'Value': f'Private Subnet for {test_name}'}]
    zone_names = [zone['ZoneName'] for zone in client.describe_availability_zones()['AvailabilityZones']]
    for counter in range(number_to_make):
        new_subnet = client.create_subnet(VpcId=vpc_id, CidrBlock=cidr_block or _get_next_available_cidr(vpc_id), AvailabilityZone=zone_names[counter], TagSpecifications=[{'ResourceType': 'subnet', 'Tags': tags}])['Subnet']['SubnetId']
        subnet_ids.append(new_subnet)
        client.get_waiter('subnet_available').wait(SubnetIds=[new_subnet])
        client.associate_route_table(RouteTableId=route_table_id, SubnetId=new_subnet)
    return subnet_ids

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_subnets(subnets) -> None:
    if False:
        for i in range(10):
            print('nop')
    for subnet in subnets:
        boto3.client('ec2').delete_subnet(SubnetId=subnet)

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_route_table(route_table_id: str) -> None:
    if False:
        i = 10
        return i + 15
    boto3.client('ec2').delete_route_table(RouteTableId=route_table_id)

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_nat_gateway(nat_gateway_id: str) -> None:
    if False:
        while True:
            i = 10
    client = boto3.client('ec2')
    client.delete_nat_gateway(NatGatewayId=nat_gateway_id)
    waiter = client.get_waiter('nat_gateway_deleted')
    waiter.wait(NatGatewayIds=[nat_gateway_id])

@task(trigger_rule=TriggerRule.ALL_DONE)
def remove_address_allocation(allocation_id):
    if False:
        while True:
            i = 10
    boto3.client('ec2').release_address(AllocationId=allocation_id)