from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class EC2NatGatewayProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::EC2::NatGateway'

    def __init__(self):
        if False:
            return 10
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            print('Hello World!')
        from localstack.services.ec2.resource_providers.aws_ec2_natgateway import EC2NatGatewayProvider
        self.factory = EC2NatGatewayProvider