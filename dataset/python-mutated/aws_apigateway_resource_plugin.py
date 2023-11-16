from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class ApiGatewayResourceProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::ApiGateway::Resource'

    def __init__(self):
        if False:
            print('Hello World!')
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            for i in range(10):
                print('nop')
        from localstack.services.apigateway.resource_providers.aws_apigateway_resource import ApiGatewayResourceProvider
        self.factory = ApiGatewayResourceProvider