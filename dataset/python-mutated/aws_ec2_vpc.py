from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.models.ec2 import _get_default_acl_for_vpc, _get_default_security_group_for_vpc
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class EC2VPCProperties(TypedDict):
    CidrBlock: Optional[str]
    CidrBlockAssociations: Optional[list[str]]
    DefaultNetworkAcl: Optional[str]
    DefaultSecurityGroup: Optional[str]
    EnableDnsHostnames: Optional[bool]
    EnableDnsSupport: Optional[bool]
    InstanceTenancy: Optional[str]
    Ipv4IpamPoolId: Optional[str]
    Ipv4NetmaskLength: Optional[int]
    Ipv6CidrBlocks: Optional[list[str]]
    Tags: Optional[list[Tag]]
    VpcId: Optional[str]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class EC2VPCProvider(ResourceProvider[EC2VPCProperties]):
    TYPE = 'AWS::EC2::VPC'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[EC2VPCProperties]) -> ProgressEvent[EC2VPCProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/VpcId\n\n        Create-only properties:\n          - /properties/CidrBlock\n          - /properties/Ipv4IpamPoolId\n          - /properties/Ipv4NetmaskLength\n\n        Read-only properties:\n          - /properties/CidrBlockAssociations\n          - /properties/DefaultNetworkAcl\n          - /properties/DefaultSecurityGroup\n          - /properties/Ipv6CidrBlocks\n          - /properties/VpcId\n\n        IAM permissions required:\n          - ec2:CreateVpc\n          - ec2:DescribeVpcs\n          - ec2:ModifyVpcAttribute\n\n        '
        model = request.desired_state
        ec2 = request.aws_client_factory.ec2
        if not request.custom_context.get(REPEATED_INVOCATION):
            params = util.select_attributes(model, ['CidrBlock', 'InstanceTenancy'])
            if model.get('Tags'):
                tags = [{'ResourceType': 'vpc', 'Tags': model.get('Tags')}]
                params['TagSpecifications'] = tags
            response = ec2.create_vpc(**params)
            model['VpcId'] = response['Vpc']['VpcId']
            model['CidrBlockAssociations'] = [cba['AssociationId'] for cba in response['Vpc']['CidrBlockAssociationSet']]
            model['DefaultNetworkAcl'] = _get_default_acl_for_vpc(ec2, model['VpcId'])
            model['DefaultSecurityGroup'] = _get_default_security_group_for_vpc(ec2, model['VpcId'])
            request.custom_context[REPEATED_INVOCATION] = True
            return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model=model, custom_context=request.custom_context)
        response = ec2.describe_vpcs(VpcIds=[model['VpcId']])['Vpcs'][0]
        if response['State'] == 'pending':
            return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model=model, custom_context=request.custom_context)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[EC2VPCProperties]) -> ProgressEvent[EC2VPCProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - ec2:DescribeVpcs\n          - ec2:DescribeSecurityGroups\n          - ec2:DescribeNetworkAcls\n          - ec2:DescribeVpcAttribute\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[EC2VPCProperties]) -> ProgressEvent[EC2VPCProperties]:
        if False:
            print('Hello World!')
        '\n        Delete a resource\n\n        IAM permissions required:\n          - ec2:DeleteVpc\n          - ec2:DescribeVpcs\n        '
        model = request.desired_state
        ec2 = request.aws_client_factory.ec2
        resp = ec2.describe_route_tables(Filters=[{'Name': 'vpc-id', 'Values': [model['VpcId']]}, {'Name': 'association.main', 'Values': ['false']}])
        for rt in resp['RouteTables']:
            for assoc in rt.get('Associations', []):
                if assoc.get('Main'):
                    continue
                ec2.disassociate_route_table(AssociationId=assoc['RouteTableAssociationId'])
            ec2.delete_route_table(RouteTableId=rt['RouteTableId'])
        ec2.delete_vpc(VpcId=model['VpcId'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[EC2VPCProperties]) -> ProgressEvent[EC2VPCProperties]:
        if False:
            return 10
        '\n        Update a resource\n\n        IAM permissions required:\n          - ec2:CreateTags\n          - ec2:ModifyVpcAttribute\n          - ec2:DeleteTags\n          - ec2:ModifyVpcTenancy\n        '
        raise NotImplementedError