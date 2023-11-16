from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class SecretsManagerSecretTargetAttachmentProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::SecretsManager::SecretTargetAttachment'

    def __init__(self):
        if False:
            while True:
                i = 10
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            return 10
        from localstack.services.secretsmanager.resource_providers.aws_secretsmanager_secrettargetattachment import SecretsManagerSecretTargetAttachmentProvider
        self.factory = SecretsManagerSecretTargetAttachmentProvider