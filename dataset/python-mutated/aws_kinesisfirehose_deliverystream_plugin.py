from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class KinesisFirehoseDeliveryStreamProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::KinesisFirehose::DeliveryStream'

    def __init__(self):
        if False:
            print('Hello World!')
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            for i in range(10):
                print('nop')
        from localstack.services.kinesisfirehose.resource_providers.aws_kinesisfirehose_deliverystream import KinesisFirehoseDeliveryStreamProvider
        self.factory = KinesisFirehoseDeliveryStreamProvider