from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class ApiGatewayRequestValidatorProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::ApiGateway::RequestValidator'

    def __init__(self):
        if False:
            return 10
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            return 10
        from localstack.services.apigateway.resource_providers.aws_apigateway_requestvalidator import ApiGatewayRequestValidatorProvider
        self.factory = ApiGatewayRequestValidatorProvider