from mage_ai.orchestration.pipeline_scheduler import schedule_with_event

class EventTrigger:

    def run(self, event) -> None:
        if False:
            i = 10
            return i + 15
        print(f'Trigger by event: {event}')
        schedule_with_event(event)