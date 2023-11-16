from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class EC2RouteTableProperties(TypedDict):
    VpcId: Optional[str]
    RouteTableId: Optional[str]
    Tags: Optional[list[Tag]]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class EC2RouteTableProvider(ResourceProvider[EC2RouteTableProperties]):
    TYPE = 'AWS::EC2::RouteTable'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[EC2RouteTableProperties]) -> ProgressEvent[EC2RouteTableProperties]:
        if False:
            while True:
                i = 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/RouteTableId\n\n        Required properties:\n          - VpcId\n\n        Create-only properties:\n          - /properties/VpcId\n\n        Read-only properties:\n          - /properties/RouteTableId\n\n        IAM permissions required:\n          - ec2:CreateRouteTable\n          - ec2:CreateTags\n          - ec2:DescribeRouteTables\n\n        '
        model = request.desired_state
        ec2 = request.aws_client_factory.ec2
        params = util.select_attributes(model, ['VpcId', 'Tags'])
        tags = [{'ResourceType': 'route-table', 'Tags': params.get('Tags', [])}]
        response = ec2.create_route_table(VpcId=params['VpcId'], TagSpecifications=tags)
        model['RouteTableId'] = response['RouteTable']['RouteTableId']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[EC2RouteTableProperties]) -> ProgressEvent[EC2RouteTableProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - ec2:DescribeRouteTables\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[EC2RouteTableProperties]) -> ProgressEvent[EC2RouteTableProperties]:
        if False:
            return 10
        '\n        Delete a resource\n\n        IAM permissions required:\n          - ec2:DescribeRouteTables\n          - ec2:DeleteRouteTable\n        '
        model = request.desired_state
        ec2 = request.aws_client_factory.ec2
        try:
            ec2.delete_route_table(RouteTableId=model['RouteTableId'])
        except ec2.exceptions.ClientError:
            pass
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[EC2RouteTableProperties]) -> ProgressEvent[EC2RouteTableProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Update a resource\n\n        IAM permissions required:\n          - ec2:CreateTags\n          - ec2:DeleteTags\n          - ec2:DescribeRouteTables\n        '
        raise NotImplementedError