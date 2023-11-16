from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class EC2SubnetRouteTableAssociationProperties(TypedDict):
    RouteTableId: Optional[str]
    SubnetId: Optional[str]
    Id: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class EC2SubnetRouteTableAssociationProvider(ResourceProvider[EC2SubnetRouteTableAssociationProperties]):
    TYPE = 'AWS::EC2::SubnetRouteTableAssociation'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[EC2SubnetRouteTableAssociationProperties]) -> ProgressEvent[EC2SubnetRouteTableAssociationProperties]:
        if False:
            return 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - RouteTableId\n          - SubnetId\n\n        Create-only properties:\n          - /properties/SubnetId\n          - /properties/RouteTableId\n\n        Read-only properties:\n          - /properties/Id\n\n        IAM permissions required:\n          - ec2:AssociateRouteTable\n          - ec2:ReplaceRouteTableAssociation\n          - ec2:DescribeSubnets\n          - ec2:DescribeRouteTables\n\n        '
        model = request.desired_state
        ec2 = request.aws_client_factory.ec2
        if not request.custom_context.get(REPEATED_INVOCATION):
            model['Id'] = ec2.associate_route_table(RouteTableId=model['RouteTableId'], SubnetId=model['SubnetId'])['AssociationId']
            request.custom_context[REPEATED_INVOCATION] = True
            return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model=model, custom_context=request.custom_context)
        route_table = ec2.describe_route_tables(RouteTableIds=[model['RouteTableId']])['RouteTables'][0]
        for association in route_table['Associations']:
            if association['RouteTableAssociationId'] == model['Id']:
                return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)
        return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[EC2SubnetRouteTableAssociationProperties]) -> ProgressEvent[EC2SubnetRouteTableAssociationProperties]:
        if False:
            print('Hello World!')
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - ec2:DescribeRouteTables\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[EC2SubnetRouteTableAssociationProperties]) -> ProgressEvent[EC2SubnetRouteTableAssociationProperties]:
        if False:
            return 10
        '\n        Delete a resource\n\n        IAM permissions required:\n          - ec2:DisassociateRouteTable\n          - ec2:DescribeSubnets\n          - ec2:DescribeRouteTables\n        '
        model = request.desired_state
        ec2 = request.aws_client_factory.ec2
        try:
            ec2.disassociate_route_table(AssociationId=model['Id'])
        except ec2.exceptions.ClientError:
            pass
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[EC2SubnetRouteTableAssociationProperties]) -> ProgressEvent[EC2SubnetRouteTableAssociationProperties]:
        if False:
            return 10
        '\n        Update a resource\n\n\n        '
        raise NotImplementedError