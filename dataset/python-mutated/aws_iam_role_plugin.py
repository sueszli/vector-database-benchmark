from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class IAMRoleProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::IAM::Role'

    def __init__(self):
        if False:
            print('Hello World!')
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            for i in range(10):
                print('nop')
        from localstack.services.iam.resource_providers.aws_iam_role import IAMRoleProvider
        self.factory = IAMRoleProvider