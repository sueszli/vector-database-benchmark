from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class LogsLogGroupProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::Logs::LogGroup'

    def __init__(self):
        if False:
            print('Hello World!')
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            i = 10
            return i + 15
        from localstack.services.logs.resource_providers.aws_logs_loggroup import LogsLogGroupProvider
        self.factory = LogsLogGroupProvider