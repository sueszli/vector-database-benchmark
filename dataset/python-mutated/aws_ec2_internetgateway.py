from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class EC2InternetGatewayProperties(TypedDict):
    InternetGatewayId: Optional[str]
    Tags: Optional[list[Tag]]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class EC2InternetGatewayProvider(ResourceProvider[EC2InternetGatewayProperties]):
    TYPE = 'AWS::EC2::InternetGateway'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[EC2InternetGatewayProperties]) -> ProgressEvent[EC2InternetGatewayProperties]:
        if False:
            print('Hello World!')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/InternetGatewayId\n\n        Read-only properties:\n          - /properties/InternetGatewayId\n\n        IAM permissions required:\n          - ec2:CreateInternetGateway\n          - ec2:CreateTags\n\n        '
        model = request.desired_state
        ec2 = request.aws_client_factory.ec2
        tags = [{'ResourceType': "'internet-gateway'", 'Tags': model.get('Tags', [])}]
        response = ec2.create_internet_gateway(TagSpecifications=tags)
        model['InternetGatewayId'] = response['InternetGateway']['InternetGatewayId']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[EC2InternetGatewayProperties]) -> ProgressEvent[EC2InternetGatewayProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - ec2:DescribeInternetGateways\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[EC2InternetGatewayProperties]) -> ProgressEvent[EC2InternetGatewayProperties]:
        if False:
            while True:
                i = 10
        '\n        Delete a resource\n\n        IAM permissions required:\n          - ec2:DeleteInternetGateway\n        '
        model = request.desired_state
        ec2 = request.aws_client_factory.ec2
        response = ec2.describe_internet_gateways(InternetGatewayIds=[model['InternetGatewayId']])
        for gateway in response.get('InternetGateways', []):
            for attachment in gateway.get('Attachments', []):
                ec2.detach_internet_gateway(InternetGatewayId=model['InternetGatewayId'], VpcId=attachment['VpcId'])
        ec2.delete_internet_gateway(InternetGatewayId=model['InternetGatewayId'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[EC2InternetGatewayProperties]) -> ProgressEvent[EC2InternetGatewayProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update a resource\n\n        IAM permissions required:\n          - ec2:DeleteTags\n          - ec2:CreateTags\n        '
        raise NotImplementedError