from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class SSMParameterProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::SSM::Parameter'

    def __init__(self):
        if False:
            while True:
                i = 10
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            for i in range(10):
                print('nop')
        from localstack.services.ssm.resource_providers.aws_ssm_parameter import SSMParameterProvider
        self.factory = SSMParameterProvider