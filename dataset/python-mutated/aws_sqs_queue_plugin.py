from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class SQSQueueProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::SQS::Queue'

    def __init__(self):
        if False:
            print('Hello World!')
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            print('Hello World!')
        from localstack.services.sqs.resource_providers.aws_sqs_queue import SQSQueueProvider
        self.factory = SQSQueueProvider