from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class EC2RouteProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::EC2::Route'

    def __init__(self):
        if False:
            return 10
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            return 10
        from localstack.services.ec2.resource_providers.aws_ec2_route import EC2RouteProvider
        self.factory = EC2RouteProvider