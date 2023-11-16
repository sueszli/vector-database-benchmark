from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class IAMManagedPolicyProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::IAM::ManagedPolicy'

    def __init__(self):
        if False:
            print('Hello World!')
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            while True:
                i = 10
        from localstack.services.iam.resource_providers.aws_iam_managedpolicy import IAMManagedPolicyProvider
        self.factory = IAMManagedPolicyProvider