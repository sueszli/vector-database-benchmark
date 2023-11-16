from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class ApiGatewayStageProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::ApiGateway::Stage'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            for i in range(10):
                print('nop')
        from localstack.services.apigateway.resource_providers.aws_apigateway_stage import ApiGatewayStageProvider
        self.factory = ApiGatewayStageProvider