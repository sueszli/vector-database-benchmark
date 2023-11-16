import logging
from typing import Optional
from unittest.mock import patch
from synapse.api.room_versions import RoomVersions
from synapse.events import EventBase, make_event_from_dict
from synapse.events.snapshot import EventContext
from synapse.types import JsonDict, create_requester
from synapse.visibility import filter_events_for_client, filter_events_for_server
from tests import unittest
from tests.utils import create_room
logger = logging.getLogger(__name__)
TEST_ROOM_ID = '!TEST:ROOM'

class FilterEventsForServerTestCase(unittest.HomeserverTestCase):

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        super().setUp()
        self.event_creation_handler = self.hs.get_event_creation_handler()
        self.event_builder_factory = self.hs.get_event_builder_factory()
        self._storage_controllers = self.hs.get_storage_controllers()
        assert self._storage_controllers.persistence is not None
        self._persistence = self._storage_controllers.persistence
        self.get_success(create_room(self.hs, TEST_ROOM_ID, '@someone:ROOM'))

    def test_filtering(self) -> None:
        if False:
            while True:
                i = 10
        self._inject_visibility('@admin:hs', 'joined')
        for i in range(10):
            self._inject_room_member('@resident%i:hs' % i)
        events_to_filter = []
        for i in range(10):
            user = '@user%i:%s' % (i, 'test_server' if i == 5 else 'other_server')
            evt = self._inject_room_member(user, extra_content={'a': 'b'})
            events_to_filter.append(evt)
        filtered = self.get_success(filter_events_for_server(self._storage_controllers, 'test_server', 'hs', events_to_filter, redact=True, filter_out_erased_senders=True, filter_out_remote_partial_state_events=True))
        for i in range(5):
            self.assertEqual(events_to_filter[i].event_id, filtered[i].event_id)
            self.assertNotIn('a', filtered[i].content)
        for i in range(5, 10):
            self.assertEqual(events_to_filter[i].event_id, filtered[i].event_id)
            self.assertEqual(filtered[i].content['a'], 'b')

    def test_filter_outlier(self) -> None:
        if False:
            i = 10
            return i + 15
        self._inject_room_member('@resident:remote_hs')
        self._inject_visibility('@resident:remote_hs', 'joined')
        outlier = self._inject_outlier()
        self.assertEqual(self.get_success(filter_events_for_server(self._storage_controllers, 'remote_hs', 'hs', [outlier], redact=True, filter_out_erased_senders=True, filter_out_remote_partial_state_events=True)), [outlier])
        evt = self._inject_message('@unerased:local_hs')
        filtered = self.get_success(filter_events_for_server(self._storage_controllers, 'remote_hs', 'local_hs', [outlier, evt], redact=True, filter_out_erased_senders=True, filter_out_remote_partial_state_events=True))
        self.assertEqual(len(filtered), 2, f'expected 2 results, got: {filtered}')
        self.assertEqual(filtered[0], outlier)
        self.assertEqual(filtered[1].event_id, evt.event_id)
        self.assertEqual(filtered[1].content, evt.content)
        filtered = self.get_success(filter_events_for_server(self._storage_controllers, 'other_server', 'local_hs', [outlier, evt], redact=True, filter_out_erased_senders=True, filter_out_remote_partial_state_events=True))
        self.assertEqual(filtered[0], outlier)
        self.assertEqual(filtered[1].event_id, evt.event_id)
        self.assertNotIn('body', filtered[1].content)

    def test_erased_user(self) -> None:
        if False:
            i = 10
            return i + 15
        events_to_filter = []
        evt = self._inject_message('@unerased:local_hs')
        events_to_filter.append(evt)
        evt = self._inject_message('@erased:local_hs')
        events_to_filter.append(evt)
        evt = self._inject_room_member('@joiner:remote_hs')
        events_to_filter.append(evt)
        evt = self._inject_message('@unerased:local_hs')
        events_to_filter.append(evt)
        evt = self._inject_message('@erased:local_hs')
        events_to_filter.append(evt)
        self.get_success(self.hs.get_datastores().main.mark_user_erased('@erased:local_hs'))
        filtered = self.get_success(filter_events_for_server(self._storage_controllers, 'test_server', 'local_hs', events_to_filter, redact=True, filter_out_erased_senders=True, filter_out_remote_partial_state_events=True))
        for i in range(len(events_to_filter)):
            self.assertEqual(events_to_filter[i].event_id, filtered[i].event_id, 'Unexpected event at result position %i' % (i,))
        for i in (0, 3):
            self.assertEqual(events_to_filter[i].content['body'], filtered[i].content['body'], 'Unexpected event content at result position %i' % (i,))
        for i in (1, 4):
            self.assertNotIn('body', filtered[i].content)

    def _inject_visibility(self, user_id: str, visibility: str) -> EventBase:
        if False:
            return 10
        content = {'history_visibility': visibility}
        builder = self.event_builder_factory.for_room_version(RoomVersions.V1, {'type': 'm.room.history_visibility', 'sender': user_id, 'state_key': '', 'room_id': TEST_ROOM_ID, 'content': content})
        (event, unpersisted_context) = self.get_success(self.event_creation_handler.create_new_client_event(builder))
        context = self.get_success(unpersisted_context.persist(event))
        self.get_success(self._persistence.persist_event(event, context))
        return event

    def _inject_room_member(self, user_id: str, membership: str='join', extra_content: Optional[JsonDict]=None) -> EventBase:
        if False:
            while True:
                i = 10
        content = {'membership': membership}
        content.update(extra_content or {})
        builder = self.event_builder_factory.for_room_version(RoomVersions.V1, {'type': 'm.room.member', 'sender': user_id, 'state_key': user_id, 'room_id': TEST_ROOM_ID, 'content': content})
        (event, unpersisted_context) = self.get_success(self.event_creation_handler.create_new_client_event(builder))
        context = self.get_success(unpersisted_context.persist(event))
        self.get_success(self._persistence.persist_event(event, context))
        return event

    def _inject_message(self, user_id: str, content: Optional[JsonDict]=None) -> EventBase:
        if False:
            for i in range(10):
                print('nop')
        if content is None:
            content = {'body': 'testytest', 'msgtype': 'm.text'}
        builder = self.event_builder_factory.for_room_version(RoomVersions.V1, {'type': 'm.room.message', 'sender': user_id, 'room_id': TEST_ROOM_ID, 'content': content})
        (event, unpersisted_context) = self.get_success(self.event_creation_handler.create_new_client_event(builder))
        context = self.get_success(unpersisted_context.persist(event))
        self.get_success(self._persistence.persist_event(event, context))
        return event

    def _inject_outlier(self) -> EventBase:
        if False:
            while True:
                i = 10
        builder = self.event_builder_factory.for_room_version(RoomVersions.V1, {'type': 'm.room.member', 'sender': '@test:user', 'state_key': '@test:user', 'room_id': TEST_ROOM_ID, 'content': {'membership': 'join'}})
        event = self.get_success(builder.build(prev_event_ids=[], auth_event_ids=[]))
        event.internal_metadata.outlier = True
        self.get_success(self._persistence.persist_event(event, EventContext.for_outlier(self._storage_controllers)))
        return event

class FilterEventsForClientTestCase(unittest.FederatingHomeserverTestCase):

    def test_out_of_band_invite_rejection(self) -> None:
        if False:
            print('Hello World!')
        invite_pdu = {'room_id': '!room:id', 'depth': 1, 'auth_events': [], 'prev_events': [], 'origin_server_ts': 1, 'sender': '@someone:' + self.OTHER_SERVER_NAME, 'type': 'm.room.member', 'state_key': '@user:test', 'content': {'membership': 'invite'}}
        self.add_hashes_and_signatures_from_other_server(invite_pdu)
        invite_event_id = make_event_from_dict(invite_pdu, RoomVersions.V9).event_id
        self.get_success(self.hs.get_federation_server().on_invite_request(self.OTHER_SERVER_NAME, invite_pdu, '9'))
        with patch.object(self.hs.get_federation_handler(), 'do_remotely_reject_invite', side_effect=Exception()):
            (reject_event_id, _) = self.get_success(self.hs.get_room_member_handler().remote_reject_invite(invite_event_id, txn_id=None, requester=create_requester('@user:test'), content={}))
        (invite_event, reject_event) = self.get_success(self.hs.get_datastores().main.get_events_as_list([invite_event_id, reject_event_id]))
        self.assertEqual(self.get_success(filter_events_for_client(self.hs.get_storage_controllers(), '@user:test', [invite_event, reject_event])), [invite_event, reject_event])
        self.assertEqual(self.get_success(filter_events_for_client(self.hs.get_storage_controllers(), '@other:test', [invite_event, reject_event])), [])