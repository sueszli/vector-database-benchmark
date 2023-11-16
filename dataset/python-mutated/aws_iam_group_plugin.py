from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class IAMGroupProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::IAM::Group'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            return 10
        from localstack.services.iam.resource_providers.aws_iam_group import IAMGroupProvider
        self.factory = IAMGroupProvider