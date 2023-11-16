from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class SNSTopicProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::SNS::Topic'

    def __init__(self):
        if False:
            return 10
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            for i in range(10):
                print('nop')
        from localstack.services.sns.resource_providers.aws_sns_topic import SNSTopicProvider
        self.factory = SNSTopicProvider