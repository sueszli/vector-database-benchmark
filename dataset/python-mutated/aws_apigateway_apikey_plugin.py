from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class ApiGatewayApiKeyProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::ApiGateway::ApiKey'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            i = 10
            return i + 15
        from localstack.services.apigateway.resource_providers.aws_apigateway_apikey import ApiGatewayApiKeyProvider
        self.factory = ApiGatewayApiKeyProvider