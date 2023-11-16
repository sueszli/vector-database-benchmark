from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class EC2NatGatewayProperties(TypedDict):
    SubnetId: Optional[str]
    AllocationId: Optional[str]
    ConnectivityType: Optional[str]
    MaxDrainDurationSeconds: Optional[int]
    NatGatewayId: Optional[str]
    PrivateIpAddress: Optional[str]
    SecondaryAllocationIds: Optional[list[str]]
    SecondaryPrivateIpAddressCount: Optional[int]
    SecondaryPrivateIpAddresses: Optional[list[str]]
    Tags: Optional[list[Tag]]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class EC2NatGatewayProvider(ResourceProvider[EC2NatGatewayProperties]):
    TYPE = 'AWS::EC2::NatGateway'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[EC2NatGatewayProperties]) -> ProgressEvent[EC2NatGatewayProperties]:
        if False:
            print('Hello World!')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/NatGatewayId\n\n        Required properties:\n          - SubnetId\n\n        Create-only properties:\n          - /properties/SubnetId\n          - /properties/ConnectivityType\n          - /properties/AllocationId\n          - /properties/PrivateIpAddress\n\n        Read-only properties:\n          - /properties/NatGatewayId\n\n        IAM permissions required:\n          - ec2:CreateNatGateway\n          - ec2:DescribeNatGateways\n          - ec2:CreateTags\n\n        '
        model = request.desired_state
        ec2 = request.aws_client_factory.ec2
        if not request.custom_context.get(REPEATED_INVOCATION):
            params = util.select_attributes(model, ['SubnetId', 'AllocationId'])
            if model.get('Tags'):
                tags = [{'ResourceType': 'natgateway', 'Tags': model.get('Tags')}]
                params['TagSpecifications'] = tags
            response = ec2.create_nat_gateway(SubnetId=model['SubnetId'], AllocationId=model['AllocationId'])
            model['NatGatewayId'] = response['NatGateway']['NatGatewayId']
            request.custom_context[REPEATED_INVOCATION] = True
            return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model=model, custom_context=request.custom_context)
        response = ec2.describe_nat_gateways(NatGatewayIds=[model['NatGatewayId']])
        if response['NatGateways'][0]['State'] == 'pending':
            return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model=model, custom_context=request.custom_context)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[EC2NatGatewayProperties]) -> ProgressEvent[EC2NatGatewayProperties]:
        if False:
            return 10
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - ec2:DescribeNatGateways\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[EC2NatGatewayProperties]) -> ProgressEvent[EC2NatGatewayProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete a resource\n\n        IAM permissions required:\n          - ec2:DeleteNatGateway\n          - ec2:DescribeNatGateways\n        '
        model = request.desired_state
        ec2 = request.aws_client_factory.ec2
        if not request.custom_context.get(REPEATED_INVOCATION):
            request.custom_context[REPEATED_INVOCATION] = True
            ec2.delete_nat_gateway(NatGatewayId=model['NatGatewayId'])
            return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model=model, custom_context=request.custom_context)
        is_deleting = False
        try:
            response = ec2.describe_nat_gateways(NatGatewayIds=[model['NatGatewayId']])
            is_deleting = response['NatGateways'][0]['State'] == 'deleting'
        except ec2.exceptions.ClientError:
            pass
        if is_deleting:
            return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model=model, custom_context=request.custom_context)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[EC2NatGatewayProperties]) -> ProgressEvent[EC2NatGatewayProperties]:
        if False:
            return 10
        '\n        Update a resource\n\n        IAM permissions required:\n          - ec2:DescribeNatGateways\n          - ec2:CreateTags\n          - ec2:DeleteTags\n          - ec2:AssociateNatGatewayAddress\n          - ec2:DisassociateNatGatewayAddress\n          - ec2:AssignPrivateNatGatewayAddress\n          - ec2:UnassignPrivateNatGatewayAddress\n        '
        raise NotImplementedError