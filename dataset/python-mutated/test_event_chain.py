from typing import Dict, List, Set, Tuple, cast
from twisted.test.proto_helpers import MemoryReactor
from twisted.trial import unittest
from synapse.api.constants import EventTypes
from synapse.api.room_versions import RoomVersions
from synapse.events import EventBase
from synapse.events.snapshot import EventContext
from synapse.rest import admin
from synapse.rest.client import login, room
from synapse.server import HomeServer
from synapse.storage.database import LoggingTransaction
from synapse.storage.databases.main.events import _LinkMap
from synapse.storage.types import Cursor
from synapse.types import create_requester
from synapse.util import Clock
from tests.unittest import HomeserverTestCase

class EventChainStoreTestCase(HomeserverTestCase):

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            return 10
        self.store = hs.get_datastores().main
        self._next_stream_ordering = 1

    def test_simple(self) -> None:
        if False:
            return 10
        'Test that the example in `docs/auth_chain_difference_algorithm.md`\n        works.\n        '
        event_factory = self.hs.get_event_builder_factory()
        bob = '@creator:test'
        alice = '@alice:test'
        room_id = '!room:test'
        self.get_success(self.store.store_room(room_id=room_id, room_creator_user_id='', is_public=True, room_version=RoomVersions.V6))
        create = self.get_success(event_factory.for_room_version(RoomVersions.V6, {'type': EventTypes.Create, 'state_key': '', 'sender': bob, 'room_id': room_id, 'content': {'tag': 'create'}}).build(prev_event_ids=[], auth_event_ids=[]))
        bob_join = self.get_success(event_factory.for_room_version(RoomVersions.V6, {'type': EventTypes.Member, 'state_key': bob, 'sender': bob, 'room_id': room_id, 'content': {'tag': 'bob_join'}}).build(prev_event_ids=[], auth_event_ids=[create.event_id]))
        power = self.get_success(event_factory.for_room_version(RoomVersions.V6, {'type': EventTypes.PowerLevels, 'state_key': '', 'sender': bob, 'room_id': room_id, 'content': {'tag': 'power'}}).build(prev_event_ids=[], auth_event_ids=[create.event_id, bob_join.event_id]))
        alice_invite = self.get_success(event_factory.for_room_version(RoomVersions.V6, {'type': EventTypes.Member, 'state_key': alice, 'sender': bob, 'room_id': room_id, 'content': {'tag': 'alice_invite'}}).build(prev_event_ids=[], auth_event_ids=[create.event_id, bob_join.event_id, power.event_id]))
        alice_join = self.get_success(event_factory.for_room_version(RoomVersions.V6, {'type': EventTypes.Member, 'state_key': alice, 'sender': alice, 'room_id': room_id, 'content': {'tag': 'alice_join'}}).build(prev_event_ids=[], auth_event_ids=[create.event_id, alice_invite.event_id, power.event_id]))
        power_2 = self.get_success(event_factory.for_room_version(RoomVersions.V6, {'type': EventTypes.PowerLevels, 'state_key': '', 'sender': bob, 'room_id': room_id, 'content': {'tag': 'power_2'}}).build(prev_event_ids=[], auth_event_ids=[create.event_id, bob_join.event_id, power.event_id]))
        bob_join_2 = self.get_success(event_factory.for_room_version(RoomVersions.V6, {'type': EventTypes.Member, 'state_key': bob, 'sender': bob, 'room_id': room_id, 'content': {'tag': 'bob_join_2'}}).build(prev_event_ids=[], auth_event_ids=[create.event_id, bob_join.event_id, power.event_id]))
        alice_join2 = self.get_success(event_factory.for_room_version(RoomVersions.V6, {'type': EventTypes.Member, 'state_key': alice, 'sender': alice, 'room_id': room_id, 'content': {'tag': 'alice_join2'}}).build(prev_event_ids=[], auth_event_ids=[create.event_id, alice_join.event_id, power_2.event_id]))
        events = [create, bob_join, power, alice_invite, alice_join, bob_join_2, power_2, alice_join2]
        expected_links = [(bob_join, create), (power, create), (power, bob_join), (alice_invite, create), (alice_invite, power), (alice_invite, bob_join), (bob_join_2, power), (alice_join2, power_2)]
        self.persist(events)
        (chain_map, link_map) = self.fetch_chains(events)
        self.assertEqual(len(expected_links), len(list(link_map.get_additions())))
        for (start, end) in expected_links:
            (start_id, start_seq) = chain_map[start.event_id]
            (end_id, end_seq) = chain_map[end.event_id]
            self.assertIn((start_seq, end_seq), list(link_map.get_links_between(start_id, end_id)))
        for event in events[1:]:
            self.assertTrue(link_map.exists_path_from(chain_map[event.event_id], chain_map[create.event_id]))
            self.assertFalse(link_map.exists_path_from(chain_map[create.event_id], chain_map[event.event_id]))

    def test_out_of_order_events(self) -> None:
        if False:
            print('Hello World!')
        "Test that we handle persisting events that we don't have the full\n        auth chain for yet (which should only happen for out of band memberships).\n        "
        event_factory = self.hs.get_event_builder_factory()
        bob = '@creator:test'
        alice = '@alice:test'
        room_id = '!room:test'
        self.get_success(self.store.store_room(room_id=room_id, room_creator_user_id='', is_public=True, room_version=RoomVersions.V6))
        create = self.get_success(event_factory.for_room_version(RoomVersions.V6, {'type': EventTypes.Create, 'state_key': '', 'sender': bob, 'room_id': room_id, 'content': {'tag': 'create'}}).build(prev_event_ids=[], auth_event_ids=[]))
        bob_join = self.get_success(event_factory.for_room_version(RoomVersions.V6, {'type': EventTypes.Member, 'state_key': bob, 'sender': bob, 'room_id': room_id, 'content': {'tag': 'bob_join'}}).build(prev_event_ids=[], auth_event_ids=[create.event_id]))
        power = self.get_success(event_factory.for_room_version(RoomVersions.V6, {'type': EventTypes.PowerLevels, 'state_key': '', 'sender': bob, 'room_id': room_id, 'content': {'tag': 'power'}}).build(prev_event_ids=[], auth_event_ids=[create.event_id, bob_join.event_id]))
        self.persist([create, bob_join, power])
        alice_invite = self.get_success(event_factory.for_room_version(RoomVersions.V6, {'type': EventTypes.Member, 'state_key': alice, 'sender': bob, 'room_id': room_id, 'content': {'tag': 'alice_invite'}}).build(prev_event_ids=[], auth_event_ids=[create.event_id, bob_join.event_id, power.event_id]))
        alice_join = self.get_success(event_factory.for_room_version(RoomVersions.V6, {'type': EventTypes.Member, 'state_key': alice, 'sender': alice, 'room_id': room_id, 'content': {'tag': 'alice_join'}}).build(prev_event_ids=[], auth_event_ids=[create.event_id, alice_invite.event_id, power.event_id]))
        alice_join2 = self.get_success(event_factory.for_room_version(RoomVersions.V6, {'type': EventTypes.Member, 'state_key': alice, 'sender': alice, 'room_id': room_id, 'content': {'tag': 'alice_join2'}}).build(prev_event_ids=[], auth_event_ids=[create.event_id, alice_join.event_id, power.event_id]))
        self.persist([alice_join])
        self.persist([alice_join2])
        self.persist([alice_invite])
        events = [create, bob_join, power, alice_invite, alice_join]
        (chain_map, link_map) = self.fetch_chains(events)
        expected_links = [(bob_join, create), (power, create), (power, bob_join), (alice_invite, create), (alice_invite, power), (alice_invite, bob_join)]
        self.assertEqual(len(expected_links), len(list(link_map.get_additions())))
        for (start, end) in expected_links:
            (start_id, start_seq) = chain_map[start.event_id]
            (end_id, end_seq) = chain_map[end.event_id]
            self.assertIn((start_seq, end_seq), list(link_map.get_links_between(start_id, end_id)))

    def persist(self, events: List[EventBase]) -> None:
        if False:
            while True:
                i = 10
        'Persist the given events and check that the links generated match\n        those given.\n        '
        persist_events_store = self.hs.get_datastores().persist_events
        assert persist_events_store is not None
        for e in events:
            e.internal_metadata.stream_ordering = self._next_stream_ordering
            self._next_stream_ordering += 1

        def _persist(txn: LoggingTransaction) -> None:
            if False:
                return 10
            assert persist_events_store is not None
            persist_events_store._store_event_txn(txn, [(e, EventContext(self.hs.get_storage_controllers(), {})) for e in events])
            persist_events_store._persist_event_auth_chain_txn(txn, events)
        self.get_success(persist_events_store.db_pool.runInteraction('_persist', _persist))

    def fetch_chains(self, events: List[EventBase]) -> Tuple[Dict[str, Tuple[int, int]], _LinkMap]:
        if False:
            for i in range(10):
                print('nop')
        rows = cast(List[Tuple[str, int, int]], self.get_success(self.store.db_pool.simple_select_many_batch(table='event_auth_chains', column='event_id', iterable=[e.event_id for e in events], retcols=('event_id', 'chain_id', 'sequence_number'), keyvalues={})))
        chain_map = {event_id: (chain_id, sequence_number) for (event_id, chain_id, sequence_number) in rows}
        auth_chain_rows = cast(List[Tuple[int, int, int, int]], self.get_success(self.store.db_pool.simple_select_many_batch(table='event_auth_chain_links', column='origin_chain_id', iterable=[chain_id for (chain_id, _) in chain_map.values()], retcols=('origin_chain_id', 'origin_sequence_number', 'target_chain_id', 'target_sequence_number'), keyvalues={})))
        link_map = _LinkMap()
        for (origin_chain_id, origin_sequence_number, target_chain_id, target_sequence_number) in auth_chain_rows:
            added = link_map.add_link((origin_chain_id, origin_sequence_number), (target_chain_id, target_sequence_number))
            self.assertTrue(added)
        return (chain_map, link_map)

class LinkMapTestCase(unittest.TestCase):

    def test_simple(self) -> None:
        if False:
            return 10
        'Basic tests for the LinkMap.'
        link_map = _LinkMap()
        link_map.add_link((1, 1), (2, 1), new=False)
        self.assertCountEqual(link_map.get_links_between(1, 2), [(1, 1)])
        self.assertCountEqual(link_map.get_links_from((1, 1)), [(2, 1)])
        self.assertCountEqual(link_map.get_additions(), [])
        self.assertTrue(link_map.exists_path_from((1, 5), (2, 1)))
        self.assertFalse(link_map.exists_path_from((1, 5), (2, 2)))
        self.assertTrue(link_map.exists_path_from((1, 5), (1, 1)))
        self.assertFalse(link_map.exists_path_from((1, 1), (1, 5)))
        self.assertFalse(link_map.add_link((1, 4), (2, 1)))
        self.assertCountEqual(link_map.get_links_between(1, 2), [(1, 1)])
        self.assertTrue(link_map.add_link((1, 3), (2, 3)))
        self.assertCountEqual(link_map.get_links_between(1, 2), [(1, 1), (3, 3)])
        self.assertTrue(link_map.add_link((2, 5), (1, 3)))
        self.assertCountEqual(link_map.get_links_between(2, 1), [(5, 3)])
        self.assertCountEqual(link_map.get_links_between(1, 2), [(1, 1), (3, 3)])
        self.assertCountEqual(link_map.get_additions(), [(1, 3, 2, 3), (2, 5, 1, 3)])

class EventChainBackgroundUpdateTestCase(HomeserverTestCase):
    servlets = [admin.register_servlets, room.register_servlets, login.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.store = hs.get_datastores().main
        self.user_id = self.register_user('foo', 'pass')
        self.token = self.login('foo', 'pass')
        self.requester = create_requester(self.user_id)

    def _generate_room(self) -> Tuple[str, List[Set[str]]]:
        if False:
            for i in range(10):
                print('nop')
        'Insert a room without a chain cover index.'
        room_id = self.helper.create_room_as(self.user_id, tok=self.token)
        self.get_success(self.store.db_pool.simple_update(table='rooms', keyvalues={'room_id': room_id}, updatevalues={'has_auth_chain_index': False}, desc='test'))
        event_handler = self.hs.get_event_creation_handler()
        latest_event_ids = self.get_success(self.store.get_prev_events_for_room(room_id))
        (event, unpersisted_context) = self.get_success(event_handler.create_event(self.requester, {'type': 'some_state_type', 'state_key': '', 'content': {}, 'room_id': room_id, 'sender': self.user_id}, prev_event_ids=latest_event_ids))
        context = self.get_success(unpersisted_context.persist(event))
        self.get_success(event_handler.handle_new_client_event(self.requester, events_and_context=[(event, context)]))
        state_ids1 = self.get_success(context.get_current_state_ids())
        assert state_ids1 is not None
        state1 = set(state_ids1.values())
        (event, unpersisted_context) = self.get_success(event_handler.create_event(self.requester, {'type': 'some_state_type', 'state_key': '', 'content': {}, 'room_id': room_id, 'sender': self.user_id}, prev_event_ids=latest_event_ids))
        context = self.get_success(unpersisted_context.persist(event))
        self.get_success(event_handler.handle_new_client_event(self.requester, events_and_context=[(event, context)]))
        state_ids2 = self.get_success(context.get_current_state_ids())
        assert state_ids2 is not None
        state2 = set(state_ids2.values())

        def _delete_tables(txn: Cursor) -> None:
            if False:
                i = 10
                return i + 15
            txn.execute('DELETE FROM event_auth_chains')
            txn.execute('DELETE FROM event_auth_chain_links')
        self.get_success(self.store.db_pool.runInteraction('test', _delete_tables))
        return (room_id, [state1, state2])

    def test_background_update_single_room(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that the background update to calculate auth chains for historic\n        rooms works correctly.\n        '
        (room_id, states) = self._generate_room()
        self.get_success(self.store.db_pool.simple_insert('background_updates', {'update_name': 'chain_cover', 'progress_json': '{}'}))
        self.store.db_pool.updates._all_done = False
        self.wait_for_background_updates()
        self.assertTrue(self.get_success(self.store.has_auth_chain_index(room_id)))
        self.get_success(self.store.db_pool.runInteraction('test', self.store._get_auth_chain_difference_using_cover_index_txn, room_id, states))

    def test_background_update_multiple_rooms(self) -> None:
        if False:
            while True:
                i = 10
        'Test that the background update to calculate auth chains for historic\n        rooms works correctly.\n        '
        (room_id1, states1) = self._generate_room()
        (room_id2, states2) = self._generate_room()
        (room_id3, states2) = self._generate_room()
        self.get_success(self.store.db_pool.simple_insert('background_updates', {'update_name': 'chain_cover', 'progress_json': '{}'}))
        self.store.db_pool.updates._all_done = False
        self.wait_for_background_updates()
        self.assertTrue(self.get_success(self.store.has_auth_chain_index(room_id1)))
        self.assertTrue(self.get_success(self.store.has_auth_chain_index(room_id2)))
        self.assertTrue(self.get_success(self.store.has_auth_chain_index(room_id3)))
        self.get_success(self.store.db_pool.runInteraction('test', self.store._get_auth_chain_difference_using_cover_index_txn, room_id1, states1))

    def test_background_update_single_large_room(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that the background update to calculate auth chains for historic\n        rooms works correctly.\n        '
        (room_id, states) = self._generate_room()
        for i in range(150):
            self.helper.send_state(room_id, event_type='m.test', body={'index': i}, tok=self.token)
        self.get_success(self.store.db_pool.simple_insert('background_updates', {'update_name': 'chain_cover', 'progress_json': '{}'}))
        self.store.db_pool.updates._all_done = False
        iterations = 0
        while not self.get_success(self.store.db_pool.updates.has_completed_background_updates()):
            iterations += 1
            self.get_success(self.store.db_pool.updates.do_next_background_update(False), by=0.1)
        self.assertGreater(iterations, 1)
        self.assertTrue(self.get_success(self.store.has_auth_chain_index(room_id)))
        self.get_success(self.store.db_pool.runInteraction('test', self.store._get_auth_chain_difference_using_cover_index_txn, room_id, states))

    def test_background_update_multiple_large_room(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that the background update to calculate auth chains for historic\n        rooms works correctly.\n        '
        (room_id1, _) = self._generate_room()
        (room_id2, _) = self._generate_room()
        for i in range(150):
            self.helper.send_state(room_id1, event_type='m.test', body={'index': i}, tok=self.token)
        for i in range(150):
            self.helper.send_state(room_id2, event_type='m.test', body={'index': i}, tok=self.token)
        self.get_success(self.store.db_pool.simple_insert('background_updates', {'update_name': 'chain_cover', 'progress_json': '{}'}))
        self.store.db_pool.updates._all_done = False
        iterations = 0
        while not self.get_success(self.store.db_pool.updates.has_completed_background_updates()):
            iterations += 1
            self.get_success(self.store.db_pool.updates.do_next_background_update(False), by=0.1)
        self.assertGreater(iterations, 1)
        self.assertTrue(self.get_success(self.store.has_auth_chain_index(room_id1)))
        self.assertTrue(self.get_success(self.store.has_auth_chain_index(room_id2)))