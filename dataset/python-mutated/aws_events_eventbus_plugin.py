from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class EventsEventBusProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::Events::EventBus'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            print('Hello World!')
        from localstack.services.events.resource_providers.aws_events_eventbus import EventsEventBusProvider
        self.factory = EventsEventBusProvider