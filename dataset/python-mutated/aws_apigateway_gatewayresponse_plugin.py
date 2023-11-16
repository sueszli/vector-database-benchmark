from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class ApiGatewayGatewayResponseProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::ApiGateway::GatewayResponse'

    def __init__(self):
        if False:
            return 10
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            print('Hello World!')
        from localstack.services.apigateway.resource_providers.aws_apigateway_gatewayresponse import ApiGatewayGatewayResponseProvider
        self.factory = ApiGatewayGatewayResponseProvider