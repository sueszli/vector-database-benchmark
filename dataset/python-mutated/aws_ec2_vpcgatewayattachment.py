from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class EC2VPCGatewayAttachmentProperties(TypedDict):
    VpcId: Optional[str]
    Id: Optional[str]
    InternetGatewayId: Optional[str]
    VpnGatewayId: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class EC2VPCGatewayAttachmentProvider(ResourceProvider[EC2VPCGatewayAttachmentProperties]):
    TYPE = 'AWS::EC2::VPCGatewayAttachment'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[EC2VPCGatewayAttachmentProperties]) -> ProgressEvent[EC2VPCGatewayAttachmentProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - VpcId\n\n\n\n        Read-only properties:\n          - /properties/Id\n\n\n\n        '
        model = request.desired_state
        ec2 = request.aws_client_factory.ec2
        if model.get('InternetGatewayId'):
            ec2.attach_internet_gateway(InternetGatewayId=model['InternetGatewayId'], VpcId=model['VpcId'])
        else:
            ec2.attach_vpn_gateway(VpnGatewayId=model['VpnGatewayId'], VpcId=model['VpcId'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[EC2VPCGatewayAttachmentProperties]) -> ProgressEvent[EC2VPCGatewayAttachmentProperties]:
        if False:
            print('Hello World!')
        '\n        Fetch resource information\n\n\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[EC2VPCGatewayAttachmentProperties]) -> ProgressEvent[EC2VPCGatewayAttachmentProperties]:
        if False:
            return 10
        '\n        Delete a resource\n\n\n        '
        model = request.desired_state
        ec2 = request.aws_client_factory.ec2
        try:
            if model.get('InternetGatewayId'):
                ec2.detach_internet_gateway(InternetGatewayId=model['InternetGatewayId'], VpcId=model['VpcId'])
            else:
                ec2.detach_vpn_gateway(VpnGatewayId=model['VpnGatewayId'], VpcId=model['VpcId'])
        except ec2.exceptions.ClientError:
            pass
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[EC2VPCGatewayAttachmentProperties]) -> ProgressEvent[EC2VPCGatewayAttachmentProperties]:
        if False:
            print('Hello World!')
        '\n        Update a resource\n\n\n        '
        raise NotImplementedError