import uuid
from sentry.replays.lib.event_linking import transform_event_for_linking_payload
from sentry.testutils.cases import ReplaysSnubaTestCase

class TestEventLink(ReplaysSnubaTestCase):

    def test_event_link_types(self):
        if False:
            for i in range(10):
                print('nop')
        replay_id = uuid.uuid4().hex
        for level in ['debug', 'info', 'warning', 'error', 'fatal']:
            event = self.store_event(data={'level': level, 'message': 'testing', 'contexts': {'replay': {'replay_id': replay_id}}}, project_id=self.project.id)
            stored = transform_event_for_linking_payload(replay_id, event)
            self.store_replays(stored)