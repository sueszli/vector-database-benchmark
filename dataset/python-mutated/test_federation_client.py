from unittest import mock
import twisted.web.client
from twisted.internet import defer
from twisted.test.proto_helpers import MemoryReactor
from synapse.api.room_versions import RoomVersions
from synapse.events import EventBase
from synapse.rest import admin
from synapse.rest.client import login, room
from synapse.server import HomeServer
from synapse.util import Clock
from tests.test_utils import FakeResponse, event_injection
from tests.unittest import FederatingHomeserverTestCase

class FederationClientTest(FederatingHomeserverTestCase):
    servlets = [admin.register_servlets, room.register_servlets, login.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, homeserver: HomeServer) -> None:
        if False:
            while True:
                i = 10
        super().prepare(reactor, clock, homeserver)
        self._mock_agent = mock.create_autospec(twisted.web.client.Agent, spec_set=True)
        homeserver.get_federation_http_client().agent = self._mock_agent
        self.reactor.advance(1000000000)
        self.creator = f'@creator:{self.OTHER_SERVER_NAME}'
        self.test_room_id = '!room_id'

    def test_get_room_state(self) -> None:
        if False:
            return 10
        create_event_dict = self.add_hashes_and_signatures_from_other_server({'room_id': self.test_room_id, 'type': 'm.room.create', 'state_key': '', 'sender': self.creator, 'content': {'creator': self.creator}, 'prev_events': [], 'auth_events': [], 'origin_server_ts': 500})
        member_event_dict = self.add_hashes_and_signatures_from_other_server({'room_id': self.test_room_id, 'type': 'm.room.member', 'sender': self.creator, 'state_key': self.creator, 'content': {'membership': 'join'}, 'prev_events': [], 'auth_events': [], 'origin_server_ts': 600})
        pl_event_dict = self.add_hashes_and_signatures_from_other_server({'room_id': self.test_room_id, 'type': 'm.room.power_levels', 'sender': self.creator, 'state_key': '', 'content': {}, 'prev_events': [], 'auth_events': [], 'origin_server_ts': 700})
        self._mock_agent.request.side_effect = lambda *args, **kwargs: defer.succeed(FakeResponse.json(payload={'pdus': [create_event_dict, member_event_dict, pl_event_dict], 'auth_chain': [create_event_dict, member_event_dict]}))
        (state_resp, auth_resp) = self.get_success(self.hs.get_federation_client().get_room_state('yet.another.server', self.test_room_id, 'event_id', RoomVersions.V9))
        self._mock_agent.request.assert_called_once_with(b'GET', b'matrix-federation://yet.another.server/_matrix/federation/v1/state/%21room_id?event_id=event_id', headers=mock.ANY, bodyProducer=None)
        self.assertEqual(auth_resp, [])
        self.assertCountEqual([e.type for e in state_resp], ['m.room.create', 'm.room.member', 'm.room.power_levels'])

    def test_get_pdu_returns_nothing_when_event_does_not_exist(self) -> None:
        if False:
            print('Hello World!')
        'No event should be returned when the event does not exist'
        pulled_pdu_info = self.get_success(self.hs.get_federation_client().get_pdu(['yet.another.server'], 'event_should_not_exist', RoomVersions.V9))
        self.assertEqual(pulled_pdu_info, None)

    def test_get_pdu(self) -> None:
        if False:
            return 10
        'Test to make sure an event is returned by `get_pdu()`'
        self._get_pdu_once()

    def test_get_pdu_event_from_cache_is_pristine(self) -> None:
        if False:
            return 10
        'Test that modifications made to events returned by `get_pdu()`\n        do not propagate back to to the internal cache (events returned should\n        be a copy).\n        '
        remote_pdu = self._get_pdu_once()
        remote_pdu.internal_metadata.outlier = True
        pulled_pdu_info2 = self.get_success(self.hs.get_federation_client().get_pdu(['yet.another.server'], remote_pdu.event_id, RoomVersions.V9))
        assert pulled_pdu_info2 is not None
        remote_pdu2 = pulled_pdu_info2.pdu
        self.assertEqual(remote_pdu.event_id, remote_pdu2.event_id)
        self.assertIsNotNone(remote_pdu2)
        self.assertEqual(remote_pdu2.internal_metadata.outlier, False)

    def _get_pdu_once(self) -> EventBase:
        if False:
            i = 10
            return i + 15
        'Retrieve an event via `get_pdu()` and assert that an event was returned.\n        Also used to prime the cache for subsequent test logic.\n        '
        message_event_dict = self.add_hashes_and_signatures_from_other_server({'room_id': self.test_room_id, 'type': 'm.room.message', 'sender': self.creator, 'state_key': '', 'content': {}, 'prev_events': [], 'auth_events': [], 'origin_server_ts': 700, 'depth': 10})
        self._mock_agent.request.side_effect = lambda *args, **kwargs: defer.succeed(FakeResponse.json(payload={'origin': 'yet.another.server', 'origin_server_ts': 900, 'pdus': [message_event_dict]}))
        pulled_pdu_info = self.get_success(self.hs.get_federation_client().get_pdu(['yet.another.server'], 'event_id', RoomVersions.V9))
        assert pulled_pdu_info is not None
        remote_pdu = pulled_pdu_info.pdu
        self._mock_agent.request.assert_called_once_with(b'GET', b'matrix-federation://yet.another.server/_matrix/federation/v1/event/event_id', headers=mock.ANY, bodyProducer=None)
        self.assertIsNotNone(remote_pdu)
        self.assertEqual(remote_pdu.internal_metadata.outlier, False)
        return remote_pdu

    def test_backfill_invalid_signature_records_failed_pull_attempts(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test to make sure that events from /backfill with invalid signatures get\n        recorded as failed pull attempts.\n        '
        OTHER_USER = f'@user:{self.OTHER_SERVER_NAME}'
        main_store = self.hs.get_datastores().main
        user_id = self.register_user('kermit', 'test')
        tok = self.login('kermit', 'test')
        room_id = self.helper.create_room_as(room_creator=user_id, tok=tok)
        (pulled_event, _) = self.get_success(event_injection.create_event(self.hs, room_id=room_id, sender=OTHER_USER, type='test_event_type', content={'body': 'garply'}))
        self._mock_agent.request.side_effect = lambda *args, **kwargs: defer.succeed(FakeResponse.json(payload={'origin': 'yet.another.server', 'origin_server_ts': 900, 'pdus': [pulled_event.get_pdu_json()]}))
        self.get_success(self.hs.get_federation_client().backfill(dest='yet.another.server', room_id=room_id, limit=1, extremities=[pulled_event.event_id]))
        backfill_num_attempts = self.get_success(main_store.db_pool.simple_select_one_onecol(table='event_failed_pull_attempts', keyvalues={'event_id': pulled_event.event_id}, retcol='num_attempts'))
        self.assertEqual(backfill_num_attempts, 2)