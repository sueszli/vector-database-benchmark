from typing import Optional
from unittest import mock
from twisted.test.proto_helpers import MemoryReactor
from synapse.api.errors import AuthError, StoreError
from synapse.api.room_versions import RoomVersion
from synapse.event_auth import check_state_dependent_auth_rules, check_state_independent_auth_rules
from synapse.events import make_event_from_dict
from synapse.events.snapshot import EventContext
from synapse.federation.transport.client import StateRequestResponse
from synapse.logging.context import LoggingContext
from synapse.rest import admin
from synapse.rest.client import login, room
from synapse.server import HomeServer
from synapse.state import StateResolutionStore
from synapse.state.v2 import _mainline_sort, _reverse_topological_power_sort
from synapse.types import JsonDict
from synapse.util import Clock
from tests import unittest
from tests.test_utils import event_injection

class FederationEventHandlerTests(unittest.FederatingHomeserverTestCase):
    servlets = [admin.register_servlets, login.register_servlets, room.register_servlets]

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            while True:
                i = 10
        self.mock_federation_transport_client = mock.Mock(spec=['get_room_state_ids', 'get_room_state', 'get_event', 'backfill'])
        self.mock_federation_transport_client.get_room_state_ids = mock.AsyncMock()
        self.mock_federation_transport_client.get_room_state = mock.AsyncMock()
        self.mock_federation_transport_client.get_event = mock.AsyncMock()
        self.mock_federation_transport_client.backfill = mock.AsyncMock()
        return super().setup_test_homeserver(federation_transport_client=self.mock_federation_transport_client)

    def test_process_pulled_event_with_missing_state(self) -> None:
        if False:
            print('Hello World!')
        'Ensure that we correctly handle pulled events with lots of missing state\n\n        In this test, we pretend we are processing a "pulled" event (eg, via backfill\n        or get_missing_events). The pulled event has a prev_event we haven\'t previously\n        seen, so the server requests the state at that prev_event. There is a lot\n        of state we don\'t have, so we expect the server to make a /state request.\n\n        We check that the pulled event is correctly persisted, and that the state is\n        as we expect.\n        '
        return self._test_process_pulled_event_with_missing_state(False)

    def test_process_pulled_event_with_missing_state_where_prev_is_outlier(self) -> None:
        if False:
            while True:
                i = 10
        'Ensure that we correctly handle pulled events with lots of missing state\n\n        A slight modification to test_process_pulled_event_with_missing_state. Again\n        we have a "pulled" event which refers to a prev_event with lots of state,\n        but in this case we already have the prev_event (as an outlier, obviously -\n        if it were a regular event, we wouldn\'t need to request the state).\n        '
        return self._test_process_pulled_event_with_missing_state(True)

    def _test_process_pulled_event_with_missing_state(self, prev_exists_as_outlier: bool) -> None:
        if False:
            while True:
                i = 10
        OTHER_USER = f'@user:{self.OTHER_SERVER_NAME}'
        main_store = self.hs.get_datastores().main
        state_storage_controller = self.hs.get_storage_controllers().state
        user_id = self.register_user('kermit', 'test')
        tok = self.login('kermit', 'test')
        room_id = self.helper.create_room_as(room_creator=user_id, tok=tok)
        room_version = self.get_success(main_store.get_room_version(room_id))
        self.helper.send_state(room_id, 'm.room.power_levels', {'events_default': 0, 'state_default': 0}, tok=tok)
        member_event = self.get_success(event_injection.inject_member_event(self.hs, room_id, OTHER_USER, 'join'))
        initial_state_map = self.get_success(main_store.get_partial_current_state_ids(room_id))
        auth_event_ids = [initial_state_map['m.room.create', ''], initial_state_map['m.room.power_levels', ''], member_event.event_id]
        state_events = [make_event_from_dict(self.add_hashes_and_signatures_from_other_server({'type': 'test_state_type', 'state_key': f'state_{i}', 'room_id': room_id, 'sender': OTHER_USER, 'prev_events': [member_event.event_id], 'auth_events': auth_event_ids, 'origin_server_ts': 1, 'depth': 10, 'content': {'body': f'state_{i}'}}), room_version) for i in range(1, 10)]
        state_at_prev_event = state_events + self.get_success(main_store.get_events_as_list(initial_state_map.values()))
        prev_event = make_event_from_dict(self.add_hashes_and_signatures_from_other_server({'type': 'test_regular_type', 'room_id': room_id, 'sender': OTHER_USER, 'prev_events': [], 'auth_events': auth_event_ids, 'origin_server_ts': 1, 'depth': 11, 'content': {'body': 'missing_prev'}}), room_version)
        if prev_exists_as_outlier:
            prev_event.internal_metadata.outlier = True
            persistence = self.hs.get_storage_controllers().persistence
            assert persistence is not None
            self.get_success(persistence.persist_event(prev_event, EventContext.for_outlier(self.hs.get_storage_controllers())))
        else:

            async def get_event(destination: str, event_id: str, timeout: Optional[int]=None) -> JsonDict:
                self.assertEqual(destination, self.OTHER_SERVER_NAME)
                self.assertEqual(event_id, prev_event.event_id)
                return {'pdus': [prev_event.get_pdu_json()]}
            self.mock_federation_transport_client.get_event.side_effect = get_event
        pulled_event = make_event_from_dict(self.add_hashes_and_signatures_from_other_server({'type': 'test_regular_type', 'room_id': room_id, 'sender': OTHER_USER, 'prev_events': [prev_event.event_id], 'auth_events': auth_event_ids, 'origin_server_ts': 1, 'depth': 12, 'content': {'body': 'pulled'}}), room_version)
        self.mock_federation_transport_client.get_room_state_ids.return_value = {'pdu_ids': [e.event_id for e in state_at_prev_event], 'auth_chain_ids': []}
        self.mock_federation_transport_client.get_room_state.return_value = StateRequestResponse(auth_events=[], state=state_at_prev_event)
        self.reactor.advance(60000)
        with LoggingContext('test'):
            self.get_success(self.hs.get_federation_event_handler()._process_pulled_event(self.OTHER_SERVER_NAME, pulled_event, backfilled=False))
        persisted = self.get_success(main_store.get_event(pulled_event.event_id))
        self.assertIsNotNone(persisted, 'pulled event was not persisted at all')
        self.assertFalse(persisted.internal_metadata.is_outlier(), 'pulled event was an outlier')
        state = self.get_success(state_storage_controller.get_state_ids_for_event(pulled_event.event_id))
        expected_state = {(e.type, e.state_key): e.event_id for e in state_at_prev_event}
        self.assertEqual(state, expected_state)
        if prev_exists_as_outlier:
            self.mock_federation_transport_client.get_event.assert_not_called()

    def test_process_pulled_event_records_failed_backfill_attempts(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test to make sure that failed backfill attempts for an event are\n        recorded in the `event_failed_pull_attempts` table.\n\n        In this test, we pretend we are processing a "pulled" event via\n        backfill. The pulled event has a fake `prev_event` which our server has\n        obviously never seen before so it attempts to request the state at that\n        `prev_event` which expectedly fails because it\'s a fake event. Because\n        the server can\'t fetch the state at the missing `prev_event`, the\n        "pulled" event fails the history check and is fails to process.\n\n        We check that we correctly record the number of failed pull attempts\n        of the pulled event and as a sanity check, that the "pulled" event isn\'t\n        persisted.\n        '
        OTHER_USER = f'@user:{self.OTHER_SERVER_NAME}'
        main_store = self.hs.get_datastores().main
        user_id = self.register_user('kermit', 'test')
        tok = self.login('kermit', 'test')
        room_id = self.helper.create_room_as(room_creator=user_id, tok=tok)
        room_version = self.get_success(main_store.get_room_version(room_id))
        self.mock_federation_transport_client.get_room_state_ids.return_value = {'pdu_ids': [], 'auth_chain_ids': []}
        self.mock_federation_transport_client.get_room_state.return_value = StateRequestResponse(auth_events=[], state=[])
        pulled_event = make_event_from_dict(self.add_hashes_and_signatures_from_other_server({'type': 'test_regular_type', 'room_id': room_id, 'sender': OTHER_USER, 'prev_events': ['$fake_prev_event'], 'auth_events': [], 'origin_server_ts': 1, 'depth': 12, 'content': {'body': 'pulled'}}), room_version)
        with LoggingContext('test'):
            self.get_success(self.hs.get_federation_event_handler()._process_pulled_event(self.OTHER_SERVER_NAME, pulled_event, backfilled=True))
        backfill_num_attempts = self.get_success(main_store.db_pool.simple_select_one_onecol(table='event_failed_pull_attempts', keyvalues={'event_id': pulled_event.event_id}, retcol='num_attempts'))
        self.assertEqual(backfill_num_attempts, 1)
        with LoggingContext('test'):
            self.get_success(self.hs.get_federation_event_handler()._process_pulled_event(self.OTHER_SERVER_NAME, pulled_event, backfilled=True))
        backfill_num_attempts = self.get_success(main_store.db_pool.simple_select_one_onecol(table='event_failed_pull_attempts', keyvalues={'event_id': pulled_event.event_id}, retcol='num_attempts'))
        self.assertEqual(backfill_num_attempts, 2)
        persisted = self.get_success(main_store.get_event(pulled_event.event_id, allow_none=True))
        self.assertIsNone(persisted, 'pulled event that fails the history check should not be persisted at all')

    def test_process_pulled_event_clears_backfill_attempts_after_being_successfully_persisted(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test to make sure that failed pull attempts\n        (`event_failed_pull_attempts` table) for an event are cleared after the\n        event is successfully persisted.\n\n        In this test, we pretend we are processing a "pulled" event via\n        backfill. The pulled event succesfully processes and the backward\n        extremeties are updated along with clearing out any failed pull attempts\n        for those old extremities.\n\n        We check that we correctly cleared failed pull attempts of the\n        pulled event.\n        '
        OTHER_USER = f'@user:{self.OTHER_SERVER_NAME}'
        main_store = self.hs.get_datastores().main
        user_id = self.register_user('kermit', 'test')
        tok = self.login('kermit', 'test')
        room_id = self.helper.create_room_as(room_creator=user_id, tok=tok)
        room_version = self.get_success(main_store.get_room_version(room_id))
        self.helper.send_state(room_id, 'm.room.power_levels', {'events_default': 0, 'state_default': 0}, tok=tok)
        member_event = self.get_success(event_injection.inject_member_event(self.hs, room_id, OTHER_USER, 'join'))
        initial_state_map = self.get_success(main_store.get_partial_current_state_ids(room_id))
        auth_event_ids = [initial_state_map['m.room.create', ''], initial_state_map['m.room.power_levels', ''], member_event.event_id]
        pulled_event = make_event_from_dict(self.add_hashes_and_signatures_from_other_server({'type': 'test_regular_type', 'room_id': room_id, 'sender': OTHER_USER, 'prev_events': [member_event.event_id], 'auth_events': auth_event_ids, 'origin_server_ts': 1, 'depth': 12, 'content': {'body': 'pulled'}}), room_version)
        self.get_success(main_store.record_event_failed_pull_attempt(pulled_event.room_id, pulled_event.event_id, 'fake cause'))
        backfill_num_attempts = self.get_success(main_store.db_pool.simple_select_one_onecol(table='event_failed_pull_attempts', keyvalues={'event_id': pulled_event.event_id}, retcol='num_attempts'))
        self.assertEqual(backfill_num_attempts, 1)
        with LoggingContext('test'):
            self.get_success(self.hs.get_federation_event_handler()._process_pulled_event(self.OTHER_SERVER_NAME, pulled_event, backfilled=True))
        backfill_num_attempts = self.get_success(main_store.db_pool.simple_select_one_onecol(table='event_failed_pull_attempts', keyvalues={'event_id': pulled_event.event_id}, retcol='num_attempts', allow_none=True))
        self.assertIsNone(backfill_num_attempts)
        persisted = self.get_success(main_store.get_event(pulled_event.event_id, allow_none=True))
        self.assertIsNotNone(persisted, 'pulled event was not persisted at all')

    def test_backfill_signature_failure_does_not_fetch_same_prev_event_later(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Test to make sure we backoff and don't try to fetch a missing prev_event when we\n        already know it has a invalid signature from checking the signatures of all of\n        the events in the backfill response.\n        "
        OTHER_USER = f'@user:{self.OTHER_SERVER_NAME}'
        main_store = self.hs.get_datastores().main
        user_id = self.register_user('kermit', 'test')
        tok = self.login('kermit', 'test')
        room_id = self.helper.create_room_as(room_creator=user_id, tok=tok)
        room_version = self.get_success(main_store.get_room_version(room_id))
        self.helper.send_state(room_id, 'm.room.power_levels', {'events_default': 0, 'state_default': 0}, tok=tok)
        member_event = self.get_success(event_injection.inject_member_event(self.hs, room_id, OTHER_USER, 'join'))
        initial_state_map = self.get_success(main_store.get_partial_current_state_ids(room_id))
        auth_event_ids = [initial_state_map['m.room.create', ''], initial_state_map['m.room.power_levels', ''], member_event.event_id]
        pulled_event_without_signatures = make_event_from_dict({'type': 'test_regular_type', 'room_id': room_id, 'sender': OTHER_USER, 'prev_events': [member_event.event_id], 'auth_events': auth_event_ids, 'origin_server_ts': 1, 'depth': 12, 'content': {'body': 'pulled_event_without_signatures'}}, room_version)
        pulled_event = make_event_from_dict(self.add_hashes_and_signatures_from_other_server({'type': 'test_regular_type', 'room_id': room_id, 'sender': OTHER_USER, 'prev_events': [member_event.event_id, pulled_event_without_signatures.event_id], 'auth_events': auth_event_ids, 'origin_server_ts': 1, 'depth': 12, 'content': {'body': 'pulled_event'}}), room_version)
        self.mock_federation_transport_client.backfill.return_value = {'origin': self.OTHER_SERVER_NAME, 'origin_server_ts': 123, 'pdus': [pulled_event_without_signatures.get_pdu_json(), pulled_event.get_pdu_json()]}
        event_endpoint_requested_count = 0
        room_state_ids_endpoint_requested_count = 0
        room_state_endpoint_requested_count = 0

        async def get_event(destination: str, event_id: str, timeout: Optional[int]=None) -> None:
            nonlocal event_endpoint_requested_count
            event_endpoint_requested_count += 1

        async def get_room_state_ids(destination: str, room_id: str, event_id: str) -> None:
            nonlocal room_state_ids_endpoint_requested_count
            room_state_ids_endpoint_requested_count += 1

        async def get_room_state(room_version: RoomVersion, destination: str, room_id: str, event_id: str) -> None:
            nonlocal room_state_endpoint_requested_count
            room_state_endpoint_requested_count += 1
        self.mock_federation_transport_client.get_event.side_effect = get_event
        self.mock_federation_transport_client.get_room_state_ids.side_effect = get_room_state_ids
        self.mock_federation_transport_client.get_room_state.side_effect = get_room_state
        with LoggingContext('test'):
            self.get_success(self.hs.get_federation_event_handler().backfill(self.OTHER_SERVER_NAME, room_id, limit=1, extremities=['$some_extremity']))
        if event_endpoint_requested_count > 0:
            self.fail(f"We don't expect an outbound request to /event in the happy path but if the logic is sneaking around what we expect, make sure to fail the test. We don't expect it because the signature failure should cause us to backoff and not asking about pulled_event_without_signatures={pulled_event_without_signatures.event_id} again")
        if room_state_ids_endpoint_requested_count > 0:
            self.fail(f"We don't expect an outbound request to /state_ids in the happy path but if the logic is sneaking around what we expect, make sure to fail the test. We don't expect it because the signature failure should cause us to backoff and not asking about pulled_event_without_signatures={pulled_event_without_signatures.event_id} again")
        if room_state_endpoint_requested_count > 0:
            self.fail(f"We don't expect an outbound request to /state in the happy path but if the logic is sneaking around what we expect, make sure to fail the test. We don't expect it because the signature failure should cause us to backoff and not asking about pulled_event_without_signatures={pulled_event_without_signatures.event_id} again")
        backfill_num_attempts_for_event_without_signatures = self.get_success(main_store.db_pool.simple_select_one_onecol(table='event_failed_pull_attempts', keyvalues={'event_id': pulled_event_without_signatures.event_id}, retcol='num_attempts'))
        self.assertEqual(backfill_num_attempts_for_event_without_signatures, 1)
        self.get_failure(main_store.db_pool.simple_select_one_onecol(table='event_failed_pull_attempts', keyvalues={'event_id': pulled_event.event_id}, retcol='num_attempts'), StoreError)

    def test_backfill_process_previously_failed_pull_attempt_event_in_the_background(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Sanity check that events are still processed even if it is in the background\n        for events that already have failed pull attempts.\n        '
        OTHER_USER = f'@user:{self.OTHER_SERVER_NAME}'
        main_store = self.hs.get_datastores().main
        user_id = self.register_user('kermit', 'test')
        tok = self.login('kermit', 'test')
        room_id = self.helper.create_room_as(room_creator=user_id, tok=tok)
        room_version = self.get_success(main_store.get_room_version(room_id))
        self.helper.send_state(room_id, 'm.room.power_levels', {'events_default': 0, 'state_default': 0}, tok=tok)
        member_event = self.get_success(event_injection.inject_member_event(self.hs, room_id, OTHER_USER, 'join'))
        initial_state_map = self.get_success(main_store.get_partial_current_state_ids(room_id))
        auth_event_ids = [initial_state_map['m.room.create', ''], initial_state_map['m.room.power_levels', ''], member_event.event_id]
        pulled_event = make_event_from_dict(self.add_hashes_and_signatures_from_other_server({'type': 'test_regular_type', 'room_id': room_id, 'sender': OTHER_USER, 'prev_events': [member_event.event_id], 'auth_events': auth_event_ids, 'origin_server_ts': 1, 'depth': 12, 'content': {'body': 'pulled_event'}}), room_version)
        self.get_success(main_store.record_event_failed_pull_attempt(room_id, pulled_event.event_id, 'fake cause'))
        self.mock_federation_transport_client.backfill.return_value = {'origin': self.OTHER_SERVER_NAME, 'origin_server_ts': 123, 'pdus': [pulled_event.get_pdu_json()]}
        with LoggingContext('test'):
            self.get_success(self.hs.get_federation_event_handler().backfill(self.OTHER_SERVER_NAME, room_id, limit=1, extremities=['$some_extremity']))
        self.reactor.pump((0.1,))
        self.get_success(main_store.get_event(pulled_event.event_id, allow_none=False))

    def test_process_pulled_event_with_rejected_missing_state(self) -> None:
        if False:
            print('Hello World!')
        'Ensure that we correctly handle pulled events with missing state containing a\n        rejected state event\n\n        In this test, we pretend we are processing a "pulled" event (eg, via backfill\n        or get_missing_events). The pulled event has a prev_event we haven\'t previously\n        seen, so the server requests the state at that prev_event. We expect the server\n        to make a /state request.\n\n        We simulate a remote server whose /state includes a rejected kick event for a\n        local user. Notably, the kick event is rejected only because it cites a rejected\n        auth event and would otherwise be accepted based on the room state. During state\n        resolution, we re-run auth and can potentially introduce such rejected events\n        into the state if we are not careful.\n\n        We check that the pulled event is correctly persisted, and that the state\n        afterwards does not include the rejected kick.\n        '
        OTHER_USER = f'@user:{self.OTHER_SERVER_NAME}'
        main_store = self.hs.get_datastores().main
        kermit_user_id = self.register_user('kermit', 'test')
        kermit_tok = self.login('kermit', 'test')
        room_id = self.helper.create_room_as(room_creator=kermit_user_id, tok=kermit_tok)
        room_version = self.get_success(main_store.get_room_version(room_id))
        bert_user_id = self.register_user('bert', 'test')
        bert_tok = self.login('bert', 'test')
        self.helper.join(room_id, user=bert_user_id, tok=bert_tok)
        self.helper.send_state(room_id, 'm.room.power_levels', {'users': {kermit_user_id: 100, OTHER_USER: 100}}, tok=kermit_tok)
        other_member_event = self.get_success(event_injection.inject_member_event(self.hs, room_id, OTHER_USER, 'join'))
        initial_state_map = self.get_success(main_store.get_partial_current_state_ids(room_id))
        create_event = self.get_success(main_store.get_event(initial_state_map['m.room.create', '']))
        bert_member_event = self.get_success(main_store.get_event(initial_state_map['m.room.member', bert_user_id]))
        power_levels_event = self.get_success(main_store.get_event(initial_state_map['m.room.power_levels', '']))
        next_depth = 100
        next_timestamp = other_member_event.origin_server_ts + 100
        rejected_power_levels_event = make_event_from_dict(self.add_hashes_and_signatures_from_other_server({'type': 'm.room.power_levels', 'state_key': '', 'room_id': room_id, 'sender': OTHER_USER, 'prev_events': [other_member_event.event_id], 'auth_events': [initial_state_map['m.room.create', ''], initial_state_map['m.room.power_levels', ''], other_member_event.event_id, other_member_event.event_id], 'origin_server_ts': next_timestamp, 'depth': next_depth, 'content': power_levels_event.content}), room_version)
        next_depth += 1
        next_timestamp += 100
        with LoggingContext('send_rejected_power_levels_event'):
            self.get_success(self.hs.get_federation_event_handler()._process_pulled_event(self.OTHER_SERVER_NAME, rejected_power_levels_event, backfilled=False))
            self.assertEqual(self.get_success(main_store.get_rejection_reason(rejected_power_levels_event.event_id)), 'auth_error')
        rejected_kick_event = make_event_from_dict(self.add_hashes_and_signatures_from_other_server({'type': 'm.room.member', 'state_key': bert_user_id, 'room_id': room_id, 'sender': OTHER_USER, 'prev_events': [rejected_power_levels_event.event_id], 'auth_events': [initial_state_map['m.room.create', ''], rejected_power_levels_event.event_id, initial_state_map['m.room.member', bert_user_id], initial_state_map['m.room.member', OTHER_USER]], 'origin_server_ts': next_timestamp, 'depth': next_depth, 'content': {'membership': 'leave'}}), room_version)
        next_depth += 1
        next_timestamp += 100
        self.get_failure(check_state_independent_auth_rules(main_store, rejected_kick_event), AuthError)
        check_state_dependent_auth_rules(rejected_kick_event, [create_event, power_levels_event, other_member_event, bert_member_event])
        self.assertEqual(self.get_success(_mainline_sort(self.clock, room_id, event_ids=[bert_member_event.event_id, rejected_kick_event.event_id], resolved_power_event_id=power_levels_event.event_id, event_map={bert_member_event.event_id: bert_member_event, rejected_kick_event.event_id: rejected_kick_event}, state_res_store=StateResolutionStore(main_store))), [bert_member_event.event_id, rejected_kick_event.event_id], "The rejected kick event will not be applied after bert's join event during state resolution. The test setup is incorrect.")
        with LoggingContext('send_rejected_kick_event'):
            self.get_success(self.hs.get_federation_event_handler()._process_pulled_event(self.OTHER_SERVER_NAME, rejected_kick_event, backfilled=False))
            self.assertEqual(self.get_success(main_store.get_rejection_reason(rejected_kick_event.event_id)), 'auth_error')
        self.reactor.advance(100)
        new_power_levels_event = self.get_success(main_store.get_event(self.helper.send_state(room_id, 'm.room.power_levels', {'users': {kermit_user_id: 100, OTHER_USER: 100, bert_user_id: 1}}, tok=kermit_tok)['event_id']))
        self.assertEqual(self.get_success(_reverse_topological_power_sort(self.clock, room_id, event_ids=[new_power_levels_event.event_id, rejected_power_levels_event.event_id], event_map={}, state_res_store=StateResolutionStore(main_store), full_conflicted_set=set())), [rejected_power_levels_event.event_id, new_power_levels_event.event_id], 'The power levels events will not have the desired ordering during state resolution. The test setup is incorrect.')
        missing_event = make_event_from_dict(self.add_hashes_and_signatures_from_other_server({'type': 'm.room.message', 'room_id': room_id, 'sender': OTHER_USER, 'prev_events': [rejected_kick_event.event_id], 'auth_events': [initial_state_map['m.room.create', ''], initial_state_map['m.room.power_levels', ''], initial_state_map['m.room.member', OTHER_USER]], 'origin_server_ts': next_timestamp, 'depth': next_depth, 'content': {'msgtype': 'm.text', 'body': 'foo'}}), room_version)
        next_depth += 1
        next_timestamp += 100
        pulled_event = make_event_from_dict(self.add_hashes_and_signatures_from_other_server({'type': 'm.room.message', 'room_id': room_id, 'sender': OTHER_USER, 'prev_events': [new_power_levels_event.event_id, missing_event.event_id], 'auth_events': [initial_state_map['m.room.create', ''], new_power_levels_event.event_id, initial_state_map['m.room.member', OTHER_USER]], 'origin_server_ts': next_timestamp, 'depth': next_depth, 'content': {'msgtype': 'm.text', 'body': 'bar'}}), room_version)
        next_depth += 1
        next_timestamp += 100
        state_before_missing_event = self.get_success(main_store.get_events_as_list(initial_state_map.values()))
        state_before_missing_event = [event for event in state_before_missing_event if event.event_id != bert_member_event.event_id]
        state_before_missing_event.append(rejected_kick_event)
        self.reactor.advance(60000)
        with LoggingContext('send_pulled_event'):

            async def get_event(destination: str, event_id: str, timeout: Optional[int]=None) -> JsonDict:
                self.assertEqual(destination, self.OTHER_SERVER_NAME)
                self.assertEqual(event_id, missing_event.event_id)
                return {'pdus': [missing_event.get_pdu_json()]}

            async def get_room_state_ids(destination: str, room_id: str, event_id: str) -> JsonDict:
                self.assertEqual(destination, self.OTHER_SERVER_NAME)
                self.assertEqual(event_id, missing_event.event_id)
                return {'pdu_ids': [event.event_id for event in state_before_missing_event], 'auth_chain_ids': []}

            async def get_room_state(room_version: RoomVersion, destination: str, room_id: str, event_id: str) -> StateRequestResponse:
                self.assertEqual(destination, self.OTHER_SERVER_NAME)
                self.assertEqual(event_id, missing_event.event_id)
                return StateRequestResponse(state=state_before_missing_event, auth_events=[])
            self.mock_federation_transport_client.get_event.side_effect = get_event
            self.mock_federation_transport_client.get_room_state_ids.side_effect = get_room_state_ids
            self.mock_federation_transport_client.get_room_state.side_effect = get_room_state
            self.get_success(self.hs.get_federation_event_handler()._process_pulled_event(self.OTHER_SERVER_NAME, pulled_event, backfilled=False))
            self.assertIsNone(self.get_success(main_store.get_rejection_reason(pulled_event.event_id)), 'Pulled event was unexpectedly rejected, likely due to a problem with the test setup.')
            self.assertEqual({pulled_event.event_id}, self.get_success(main_store.have_events_in_timeline([pulled_event.event_id])), 'Pulled event was not persisted, likely due to a problem with the test setup.')
            new_state_map = self.get_success(main_store.get_partial_current_state_ids(room_id))
            self.assertEqual(new_state_map['m.room.member', bert_user_id], bert_member_event.event_id, 'Rejected kick event unexpectedly became part of room state.')