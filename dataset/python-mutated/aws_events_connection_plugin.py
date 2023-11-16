from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class EventsConnectionProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::Events::Connection'

    def __init__(self):
        if False:
            print('Hello World!')
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            i = 10
            return i + 15
        from localstack.services.events.resource_providers.aws_events_connection import EventsConnectionProvider
        self.factory = EventsConnectionProvider