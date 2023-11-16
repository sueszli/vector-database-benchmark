from sentry.eventstore.models import GroupEvent
from sentry.rules import EventState
from sentry.rules.conditions.base import EventCondition

class EveryEventCondition(EventCondition):
    id = 'sentry.rules.conditions.every_event.EveryEventCondition'
    label = 'The event occurs'

    def passes(self, event: GroupEvent, state: EventState) -> bool:
        if False:
            while True:
                i = 10
        return True

    def is_enabled(self) -> bool:
        if False:
            print('Hello World!')
        return False