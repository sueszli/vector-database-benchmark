from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class KinesisStreamProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::Kinesis::Stream'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            for i in range(10):
                print('nop')
        from localstack.services.kinesis.resource_providers.aws_kinesis_stream import KinesisStreamProvider
        self.factory = KinesisStreamProvider