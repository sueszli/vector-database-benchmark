from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class EC2VPCGatewayAttachmentProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::EC2::VPCGatewayAttachment'

    def __init__(self):
        if False:
            while True:
                i = 10
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            return 10
        from localstack.services.ec2.resource_providers.aws_ec2_vpcgatewayattachment import EC2VPCGatewayAttachmentProvider
        self.factory = EC2VPCGatewayAttachmentProvider