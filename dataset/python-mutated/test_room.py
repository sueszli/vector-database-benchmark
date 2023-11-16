import json
from twisted.test.proto_helpers import MemoryReactor
from synapse.api.constants import RoomTypes
from synapse.rest import admin
from synapse.rest.client import login, room
from synapse.server import HomeServer
from synapse.storage.databases.main.room import _BackgroundUpdates
from synapse.util import Clock
from tests.unittest import HomeserverTestCase

class RoomBackgroundUpdateStoreTestCase(HomeserverTestCase):
    servlets = [admin.register_servlets, room.register_servlets, login.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            print('Hello World!')
        self.store = hs.get_datastores().main
        self.user_id = self.register_user('foo', 'pass')
        self.token = self.login('foo', 'pass')

    def _generate_room(self) -> str:
        if False:
            while True:
                i = 10
        'Create a room and return the room ID.'
        return self.helper.create_room_as(self.user_id, tok=self.token)

    def run_background_updates(self, update_name: str) -> None:
        if False:
            while True:
                i = 10
        'Insert and run the background update.'
        self.get_success(self.store.db_pool.simple_insert('background_updates', {'update_name': update_name, 'progress_json': '{}'}))
        self.store.db_pool.updates._all_done = False
        self.wait_for_background_updates()

    def test_background_populate_rooms_creator_column(self) -> None:
        if False:
            return 10
        'Test that the background update to populate the rooms creator column\n        works properly.\n        '
        room_id = self._generate_room()
        self.get_success(self.store.db_pool.simple_update(table='rooms', keyvalues={'room_id': room_id}, updatevalues={'creator': None}, desc='test'))
        room_creator_before = self.get_success(self.store.db_pool.simple_select_one_onecol(table='rooms', keyvalues={'room_id': room_id}, retcol='creator', allow_none=True))
        self.assertEqual(room_creator_before, None)
        self.run_background_updates(_BackgroundUpdates.POPULATE_ROOMS_CREATOR_COLUMN)
        room_creator_after = self.get_success(self.store.db_pool.simple_select_one_onecol(table='rooms', keyvalues={'room_id': room_id}, retcol='creator', allow_none=True))
        self.assertEqual(room_creator_after, self.user_id)

    def test_background_add_room_type_column(self) -> None:
        if False:
            print('Hello World!')
        'Test that the background update to populate the `room_type` column in\n        `room_stats_state` works properly.\n        '
        room_id = self._generate_room()
        event_id = self.get_success(self.store.db_pool.simple_select_one_onecol(table='current_state_events', keyvalues={'room_id': room_id, 'type': 'm.room.create'}, retcol='event_id'))
        event = {'content': {'creator': '@user:server.org', 'room_version': '9', 'type': RoomTypes.SPACE}, 'type': 'm.room.create'}
        self.get_success(self.store.db_pool.simple_update(table='event_json', keyvalues={'event_id': event_id}, updatevalues={'json': json.dumps(event)}, desc='test'))
        self.run_background_updates(_BackgroundUpdates.ADD_ROOM_TYPE_COLUMN)
        room_type_after = self.get_success(self.store.db_pool.simple_select_one_onecol(table='room_stats_state', keyvalues={'room_id': room_id}, retcol='room_type', allow_none=True))
        self.assertEqual(room_type_after, RoomTypes.SPACE)

    def test_populate_stats_broken_rooms(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Ensure that re-populating room stats skips broken rooms.'
        good_room_id = self._generate_room()
        room_id = self._generate_room()
        self.get_success(self.store.db_pool.simple_update(table='rooms', keyvalues={'room_id': room_id}, updatevalues={'room_version': None}, desc='test'))
        self.get_success(self.store.db_pool.simple_delete(table='room_stats_state', keyvalues={'1': 1}, desc='test'))
        self.run_background_updates('populate_stats_process_rooms')
        results = self.get_success(self.store.db_pool.simple_select_onecol(table='room_stats_state', keyvalues={}, retcol='room_id'))
        self.assertEqual(results, [good_room_id])