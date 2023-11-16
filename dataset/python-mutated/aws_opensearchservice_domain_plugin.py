from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class OpenSearchServiceDomainProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::OpenSearchService::Domain'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            while True:
                i = 10
        from localstack.services.opensearch.resource_providers.aws_opensearchservice_domain import OpenSearchServiceDomainProvider
        self.factory = OpenSearchServiceDomainProvider