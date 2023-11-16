from twisted.test.proto_helpers import MemoryReactor
from synapse.events import EventBase
from synapse.rest import admin, login, room
from synapse.server import HomeServer
from synapse.types import JsonDict
from synapse.util import Clock
from tests.unittest import HomeserverTestCase

class EventUnsignedAdditionTestCase(HomeserverTestCase):
    servlets = [room.register_servlets, admin.register_servlets, login.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, homeserver: HomeServer) -> None:
        if False:
            print('Hello World!')
        self._store = homeserver.get_datastores().main
        self._module_api = homeserver.get_module_api()
        self._account_data_mgr = self._module_api.account_data_manager

    def test_annotate_event(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that we can annotate an event when we request it from the\n        server.\n        '

        async def add_unsigned_event(event: EventBase) -> JsonDict:
            return {'test_key': event.event_id}
        self._module_api.register_add_extra_fields_to_unsigned_client_event_callbacks(add_field_to_unsigned_callback=add_unsigned_event)
        user_id = self.register_user('user', 'password')
        token = self.login('user', 'password')
        room_id = self.helper.create_room_as(user_id, tok=token)
        result = self.helper.send(room_id, 'Hello!', tok=token)
        event_id = result['event_id']
        event_json = self.helper.get_event(room_id, event_id, tok=token)
        self.assertEqual(event_json['unsigned'].get('test_key'), event_id)