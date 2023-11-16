from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class SecretsManagerSecretProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::SecretsManager::Secret'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            while True:
                i = 10
        from localstack.services.secretsmanager.resource_providers.aws_secretsmanager_secret import SecretsManagerSecretProvider
        self.factory = SecretsManagerSecretProvider