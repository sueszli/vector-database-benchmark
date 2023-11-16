from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class EventsRuleProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::Events::Rule'

    def __init__(self):
        if False:
            while True:
                i = 10
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            i = 10
            return i + 15
        from localstack.services.events.resource_providers.aws_events_rule import EventsRuleProvider
        self.factory = EventsRuleProvider