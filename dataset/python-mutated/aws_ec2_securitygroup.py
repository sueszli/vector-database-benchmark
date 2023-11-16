from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class EC2SecurityGroupProperties(TypedDict):
    GroupDescription: Optional[str]
    GroupId: Optional[str]
    GroupName: Optional[str]
    Id: Optional[str]
    SecurityGroupEgress: Optional[list[Egress]]
    SecurityGroupIngress: Optional[list[Ingress]]
    Tags: Optional[list[Tag]]
    VpcId: Optional[str]

class Ingress(TypedDict):
    IpProtocol: Optional[str]
    CidrIp: Optional[str]
    CidrIpv6: Optional[str]
    Description: Optional[str]
    FromPort: Optional[int]
    SourcePrefixListId: Optional[str]
    SourceSecurityGroupId: Optional[str]
    SourceSecurityGroupName: Optional[str]
    SourceSecurityGroupOwnerId: Optional[str]
    ToPort: Optional[int]

class Egress(TypedDict):
    IpProtocol: Optional[str]
    CidrIp: Optional[str]
    CidrIpv6: Optional[str]
    Description: Optional[str]
    DestinationPrefixListId: Optional[str]
    DestinationSecurityGroupId: Optional[str]
    FromPort: Optional[int]
    ToPort: Optional[int]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class EC2SecurityGroupProvider(ResourceProvider[EC2SecurityGroupProperties]):
    TYPE = 'AWS::EC2::SecurityGroup'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[EC2SecurityGroupProperties]) -> ProgressEvent[EC2SecurityGroupProperties]:
        if False:
            return 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - GroupDescription\n\n        Create-only properties:\n          - /properties/GroupDescription\n          - /properties/GroupName\n          - /properties/VpcId\n\n        Read-only properties:\n          - /properties/Id\n          - /properties/GroupId\n\n\n\n        '
        model = request.desired_state
        ec2 = request.aws_client_factory.ec2
        params = {}
        if not model.get('GroupName'):
            params['GroupName'] = util.generate_default_name(request.stack_name, request.logical_resource_id)
        else:
            params['GroupName'] = model['GroupName']
        if (vpc_id := model.get('VpcId')):
            params['VpcId'] = vpc_id
        params['Description'] = model.get('GroupDescription', '')
        response = ec2.create_security_group(**params)
        model['GroupId'] = response['GroupId']
        model['Id'] = response['GroupId']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[EC2SecurityGroupProperties]) -> ProgressEvent[EC2SecurityGroupProperties]:
        if False:
            while True:
                i = 10
        '\n        Fetch resource information\n\n\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[EC2SecurityGroupProperties]) -> ProgressEvent[EC2SecurityGroupProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Delete a resource\n\n\n        '
        model = request.desired_state
        ec2 = request.aws_client_factory.ec2
        ec2.delete_security_group(GroupId=model['GroupId'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[EC2SecurityGroupProperties]) -> ProgressEvent[EC2SecurityGroupProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update a resource\n\n\n        '
        raise NotImplementedError