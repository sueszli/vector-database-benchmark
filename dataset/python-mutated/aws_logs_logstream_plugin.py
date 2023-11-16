from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class LogsLogStreamProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::Logs::LogStream'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            print('Hello World!')
        from localstack.services.logs.resource_providers.aws_logs_logstream import LogsLogStreamProvider
        self.factory = LogsLogStreamProvider