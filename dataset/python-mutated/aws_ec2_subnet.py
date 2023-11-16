from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest
from localstack.utils.strings import str_to_bool

class EC2SubnetProperties(TypedDict):
    VpcId: Optional[str]
    AssignIpv6AddressOnCreation: Optional[bool]
    AvailabilityZone: Optional[str]
    AvailabilityZoneId: Optional[str]
    CidrBlock: Optional[str]
    EnableDns64: Optional[bool]
    Ipv6CidrBlock: Optional[str]
    Ipv6CidrBlocks: Optional[list[str]]
    Ipv6Native: Optional[bool]
    MapPublicIpOnLaunch: Optional[bool]
    NetworkAclAssociationId: Optional[str]
    OutpostArn: Optional[str]
    PrivateDnsNameOptionsOnLaunch: Optional[dict]
    SubnetId: Optional[str]
    Tags: Optional[list[Tag]]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class EC2SubnetProvider(ResourceProvider[EC2SubnetProperties]):
    TYPE = 'AWS::EC2::Subnet'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[EC2SubnetProperties]) -> ProgressEvent[EC2SubnetProperties]:
        if False:
            return 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/SubnetId\n\n        Required properties:\n          - VpcId\n\n        Create-only properties:\n          - /properties/VpcId\n          - /properties/AvailabilityZone\n          - /properties/AvailabilityZoneId\n          - /properties/CidrBlock\n          - /properties/OutpostArn\n          - /properties/Ipv6Native\n\n        Read-only properties:\n          - /properties/NetworkAclAssociationId\n          - /properties/SubnetId\n          - /properties/Ipv6CidrBlocks\n\n        IAM permissions required:\n          - ec2:DescribeSubnets\n          - ec2:CreateSubnet\n          - ec2:CreateTags\n          - ec2:ModifySubnetAttribute\n\n        '
        model = request.desired_state
        ec2 = request.aws_client_factory.ec2
        params = util.select_attributes(model, ['AvailabilityZone', 'AvailabilityZoneId', 'CidrBlock', 'Ipv6CidrBlock', 'Ipv6Native', 'OutpostArn', 'VpcId'])
        if model.get('Tags'):
            tags = [{'ResourceType': 'subnet', 'Tags': model.get('Tags')}]
            params['TagSpecifications'] = tags
        response = ec2.create_subnet(**params)
        model['SubnetId'] = response['Subnet']['SubnetId']
        bool_attrs = ['AssignIpv6AddressOnCreation', 'EnableDns64', 'MapPublicIpOnLaunch']
        custom_attrs = bool_attrs + ['PrivateDnsNameOptionsOnLaunch']
        if not any((attr in model for attr in custom_attrs)):
            return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)
        for attr in bool_attrs:
            if attr in model:
                kwargs = {attr: {'Value': str_to_bool(model[attr])}}
                ec2.modify_subnet_attribute(SubnetId=model['SubnetId'], **kwargs)
        dns_options = model.get('PrivateDnsNameOptionsOnLaunch')
        if dns_options:
            if isinstance(dns_options, str):
                dns_options = json.loads(dns_options)
            if dns_options.get('HostnameType'):
                ec2.modify_subnet_attribute(SubnetId=model['SubnetId'], PrivateDnsHostnameTypeOnLaunch=dns_options.get('HostnameType'))
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[EC2SubnetProperties]) -> ProgressEvent[EC2SubnetProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - ec2:DescribeSubnets\n          - ec2:DescribeNetworkAcls\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[EC2SubnetProperties]) -> ProgressEvent[EC2SubnetProperties]:
        if False:
            return 10
        '\n        Delete a resource\n\n        IAM permissions required:\n          - ec2:DescribeSubnets\n          - ec2:DeleteSubnet\n        '
        model = request.desired_state
        ec2 = request.aws_client_factory.ec2
        ec2.delete_subnet(SubnetId=model['SubnetId'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[EC2SubnetProperties]) -> ProgressEvent[EC2SubnetProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Update a resource\n\n        IAM permissions required:\n          - ec2:DescribeSubnets\n          - ec2:ModifySubnetAttribute\n          - ec2:CreateTags\n          - ec2:DeleteTags\n          - ec2:AssociateSubnetCidrBlock\n          - ec2:DisassociateSubnetCidrBlock\n        '
        raise NotImplementedError