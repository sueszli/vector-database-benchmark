from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class CertificateManagerCertificateProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::CertificateManager::Certificate'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            while True:
                i = 10
        from localstack.services.certificatemanager.resource_providers.aws_certificatemanager_certificate import CertificateManagerCertificateProvider
        self.factory = CertificateManagerCertificateProvider