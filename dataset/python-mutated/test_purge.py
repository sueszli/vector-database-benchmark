from twisted.test.proto_helpers import MemoryReactor
from synapse.api.errors import NotFoundError, SynapseError
from synapse.rest.client import room
from synapse.server import HomeServer
from synapse.util import Clock
from tests.unittest import HomeserverTestCase

class PurgeTests(HomeserverTestCase):
    user_id = '@red:server'
    servlets = [room.register_servlets]

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            return 10
        hs = self.setup_test_homeserver('server')
        return hs

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.room_id = self.helper.create_room_as(self.user_id)
        self.store = hs.get_datastores().main
        self._storage_controllers = self.hs.get_storage_controllers()

    def test_purge_history(self) -> None:
        if False:
            return 10
        '\n        Purging a room history will delete everything before the topological point.\n        '
        first = self.helper.send(self.room_id, body='test1')
        second = self.helper.send(self.room_id, body='test2')
        third = self.helper.send(self.room_id, body='test3')
        last = self.helper.send(self.room_id, body='test4')
        token = self.get_success(self.store.get_topological_token_for_event(last['event_id']))
        token_str = self.get_success(token.to_string(self.hs.get_datastores().main))
        self.get_success(self._storage_controllers.purge_events.purge_history(self.room_id, token_str, True))
        self.get_failure(self.store.get_event(first['event_id']), NotFoundError)
        self.get_failure(self.store.get_event(second['event_id']), NotFoundError)
        self.get_failure(self.store.get_event(third['event_id']), NotFoundError)
        self.get_success(self.store.get_event(last['event_id']))

    def test_purge_history_wont_delete_extrems(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Purging a room history will delete everything before the topological point.\n        '
        first = self.helper.send(self.room_id, body='test1')
        second = self.helper.send(self.room_id, body='test2')
        third = self.helper.send(self.room_id, body='test3')
        last = self.helper.send(self.room_id, body='test4')
        token = self.get_success(self.store.get_topological_token_for_event(last['event_id']))
        assert token.topological is not None
        event = f't{token.topological + 1}-{token.stream + 1}'
        f = self.get_failure(self._storage_controllers.purge_events.purge_history(self.room_id, event, True), SynapseError)
        self.assertIn('greater than forward', f.value.args[0])
        self.get_success(self.store.get_event(first['event_id']))
        self.get_success(self.store.get_event(second['event_id']))
        self.get_success(self.store.get_event(third['event_id']))
        self.get_success(self.store.get_event(last['event_id']))

    def test_purge_room(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Purging a room will delete everything about it.\n        '
        first = self.helper.send(self.room_id, body='test1')
        create_event = self.get_success(self._storage_controllers.state.get_current_state_event(self.room_id, 'm.room.create', ''))
        assert create_event is not None
        self.get_success(self._storage_controllers.purge_events.purge_room(self.room_id))
        self.store._invalidate_local_get_event_cache(create_event.event_id)
        self.get_failure(self.store.get_event(create_event.event_id), NotFoundError)
        self.get_failure(self.store.get_event(first['event_id']), NotFoundError)