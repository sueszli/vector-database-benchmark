from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class EC2NetworkAclProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::EC2::NetworkAcl'

    def __init__(self):
        if False:
            return 10
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            i = 10
            return i + 15
        from localstack.services.ec2.resource_providers.aws_ec2_networkacl import EC2NetworkAclProvider
        self.factory = EC2NetworkAclProvider