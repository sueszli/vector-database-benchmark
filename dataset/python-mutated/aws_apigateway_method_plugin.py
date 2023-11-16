from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class ApiGatewayMethodProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::ApiGateway::Method'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            return 10
        from localstack.services.apigateway.resource_providers.aws_apigateway_method import ApiGatewayMethodProvider
        self.factory = ApiGatewayMethodProvider