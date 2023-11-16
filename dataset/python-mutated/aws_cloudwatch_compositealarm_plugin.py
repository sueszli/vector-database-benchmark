from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class CloudWatchCompositeAlarmProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::CloudWatch::CompositeAlarm'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            print('Hello World!')
        from localstack.services.cloudwatch.resource_providers.aws_cloudwatch_compositealarm import CloudWatchCompositeAlarmProvider
        self.factory = CloudWatchCompositeAlarmProvider