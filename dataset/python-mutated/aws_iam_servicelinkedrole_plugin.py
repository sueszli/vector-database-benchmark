from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class IAMServiceLinkedRoleProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::IAM::ServiceLinkedRole'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            i = 10
            return i + 15
        from localstack.services.iam.resource_providers.aws_iam_servicelinkedrole import IAMServiceLinkedRoleProvider
        self.factory = IAMServiceLinkedRoleProvider