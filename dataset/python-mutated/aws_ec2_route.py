from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
from moto.ec2.utils import generate_route_id
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class EC2RouteProperties(TypedDict):
    RouteTableId: Optional[str]
    CarrierGatewayId: Optional[str]
    DestinationCidrBlock: Optional[str]
    DestinationIpv6CidrBlock: Optional[str]
    EgressOnlyInternetGatewayId: Optional[str]
    GatewayId: Optional[str]
    Id: Optional[str]
    InstanceId: Optional[str]
    LocalGatewayId: Optional[str]
    NatGatewayId: Optional[str]
    NetworkInterfaceId: Optional[str]
    TransitGatewayId: Optional[str]
    VpcEndpointId: Optional[str]
    VpcPeeringConnectionId: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class EC2RouteProvider(ResourceProvider[EC2RouteProperties]):
    TYPE = 'AWS::EC2::Route'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[EC2RouteProperties]) -> ProgressEvent[EC2RouteProperties]:
        if False:
            print('Hello World!')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - RouteTableId\n\n        Create-only properties:\n          - /properties/RouteTableId\n          - /properties/DestinationCidrBlock\n\n        Read-only properties:\n          - /properties/Id\n\n\n\n        '
        model = request.desired_state
        ec2 = request.aws_client_factory.ec2
        cidr_block = model.get('DestinationCidrBlock')
        ipv6_cidr_block = model.get('DestinationIpv6CidrBlock', '')
        ec2.create_route(DestinationCidrBlock=cidr_block, DestinationIpv6CidrBlock=ipv6_cidr_block, RouteTableId=model['RouteTableId'])
        model['Id'] = generate_route_id(model['RouteTableId'], cidr_block, ipv6_cidr_block)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[EC2RouteProperties]) -> ProgressEvent[EC2RouteProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Fetch resource information\n\n\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[EC2RouteProperties]) -> ProgressEvent[EC2RouteProperties]:
        if False:
            while True:
                i = 10
        '\n        Delete a resource\n\n\n        '
        model = request.desired_state
        ec2 = request.aws_client_factory.ec2
        cidr_block = model.get('DestinationCidrBlock')
        ipv6_cidr_block = model.get('DestinationIpv6CidrBlock', '')
        try:
            ec2.delete_route(DestinationCidrBlock=cidr_block, DestinationIpv6CidrBlock=ipv6_cidr_block, RouteTableId=model['RouteTableId'])
        except ec2.exceptions.ClientError:
            pass
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[EC2RouteProperties]) -> ProgressEvent[EC2RouteProperties]:
        if False:
            print('Hello World!')
        '\n        Update a resource\n\n\n        '
        raise NotImplementedError