import datetime
from typing import Collection, Dict, FrozenSet, Iterable, List, Mapping, Set, Tuple, TypeVar, Union, cast
import attr
from parameterized import parameterized
from twisted.test.proto_helpers import MemoryReactor
from synapse.api.constants import EventTypes
from synapse.api.room_versions import KNOWN_ROOM_VERSIONS, EventFormatVersions, RoomVersion
from synapse.events import EventBase, _EventInternalMetadata
from synapse.rest import admin
from synapse.rest.client import login, room
from synapse.server import HomeServer
from synapse.storage.database import LoggingTransaction
from synapse.storage.types import Cursor
from synapse.types import JsonDict
from synapse.util import Clock, json_encoder
import tests.unittest
import tests.utils
AUTH_GRAPH: Dict[str, List[str]] = {'a': ['e'], 'b': ['e'], 'c': ['g', 'i'], 'd': ['f'], 'e': ['f'], 'f': ['g'], 'g': ['h', 'i'], 'h': ['k'], 'i': ['j'], 'k': [], 'j': []}
DEPTH_GRAPH = {'a': 7, 'b': 7, 'c': 4, 'd': 6, 'e': 6, 'f': 5, 'g': 3, 'h': 2, 'i': 2, 'k': 1, 'j': 1}
T = TypeVar('T')

def get_all_topologically_sorted_orders(nodes: Iterable[T], graph: Mapping[T, Collection[T]]) -> List[List[T]]:
    if False:
        print('Hello World!')
    'Given a set of nodes and a graph, return all possible topological\n    orderings.\n    '
    degree_map = {node: 0 for node in nodes}
    reverse_graph: Dict[T, Set[T]] = {}
    for (node, edges) in graph.items():
        if node not in degree_map:
            continue
        for edge in set(edges):
            if edge in degree_map:
                degree_map[node] += 1
            reverse_graph.setdefault(edge, set()).add(node)
        reverse_graph.setdefault(node, set())
    zero_degree = [node for (node, degree) in degree_map.items() if degree == 0]
    return _get_all_topologically_sorted_orders_inner(reverse_graph, zero_degree, degree_map)

def _get_all_topologically_sorted_orders_inner(reverse_graph: Dict[T, Set[T]], zero_degree: List[T], degree_map: Dict[T, int]) -> List[List[T]]:
    if False:
        print('Hello World!')
    new_paths = []
    for node in zero_degree:
        new_degree_map = degree_map.copy()
        new_zero_degree = zero_degree.copy()
        new_zero_degree.remove(node)
        for edge in reverse_graph.get(node, []):
            if edge in new_degree_map:
                new_degree_map[edge] -= 1
                if new_degree_map[edge] == 0:
                    new_zero_degree.append(edge)
        paths = _get_all_topologically_sorted_orders_inner(reverse_graph, new_zero_degree, new_degree_map)
        for path in paths:
            path.insert(0, node)
        new_paths.extend(paths)
    if not new_paths:
        return [[]]
    return new_paths

def get_all_topologically_consistent_subsets(nodes: Iterable[T], graph: Mapping[T, Collection[T]]) -> Set[FrozenSet[T]]:
    if False:
        return 10
    'Get all subsets of the graph where if node N is in the subgraph, then all\n    nodes that can reach that node (i.e. for all X there exists a path X -> N)\n    are in the subgraph.\n    '
    all_topological_orderings = get_all_topologically_sorted_orders(nodes, graph)
    graph_subsets = set()
    for ordering in all_topological_orderings:
        ordering.reverse()
        for idx in range(len(ordering)):
            graph_subsets.add(frozenset(ordering[:idx]))
    return graph_subsets

@attr.s(auto_attribs=True, frozen=True, slots=True)
class _BackfillSetupInfo:
    room_id: str
    depth_map: Dict[str, int]

class EventFederationWorkerStoreTestCase(tests.unittest.HomeserverTestCase):
    servlets = [admin.register_servlets, room.register_servlets, login.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            i = 10
            return i + 15
        self.store = hs.get_datastores().main
        persist_events = hs.get_datastores().persist_events
        assert persist_events is not None
        self.persist_events = persist_events

    def test_get_prev_events_for_room(self) -> None:
        if False:
            i = 10
            return i + 15
        room_id = '@ROOM:local'

        def insert_event(txn: Cursor, i: int) -> None:
            if False:
                print('Hello World!')
            event_id = '$event_%i:local' % i
            txn.execute("INSERT INTO events (   room_id, event_id, type, depth, topological_ordering,   content, processed, outlier, stream_ordering) VALUES (?, ?, 'm.test', ?, ?, 'test', ?, ?, ?)", (room_id, event_id, i, i, True, False, i))
            txn.execute('INSERT INTO event_forward_extremities (room_id, event_id) VALUES (?, ?)', (room_id, event_id))
        for i in range(20):
            self.get_success(self.store.db_pool.runInteraction('insert', insert_event, i))
        r = self.get_success(self.store.get_prev_events_for_room(room_id))
        self.assertEqual(10, len(r))
        for i in range(10):
            self.assertEqual('$event_%i:local' % (19 - i), r[i])

    def test_get_rooms_with_many_extremities(self) -> None:
        if False:
            i = 10
            return i + 15
        room1 = '#room1'
        room2 = '#room2'
        room3 = '#room3'

        def insert_event(txn: LoggingTransaction, i: int, room_id: str) -> None:
            if False:
                print('Hello World!')
            event_id = '$event_%i:local' % i
            self.store.db_pool.simple_insert_txn(txn, table='events', values={'instance_name': 'master', 'stream_ordering': self.store._stream_id_gen.get_next_txn(txn), 'topological_ordering': 1, 'depth': 1, 'event_id': event_id, 'room_id': room_id, 'type': EventTypes.Message, 'processed': True, 'outlier': False, 'origin_server_ts': 0, 'received_ts': 0, 'sender': '@user:local', 'contains_url': False, 'state_key': None, 'rejection_reason': None})
            txn.execute('INSERT INTO event_forward_extremities (room_id, event_id) VALUES (?, ?)', (room_id, event_id))
        for i in range(20):
            self.get_success(self.store.db_pool.runInteraction('insert', insert_event, i, room1))
            self.get_success(self.store.db_pool.runInteraction('insert', insert_event, i + 100, room2))
            self.get_success(self.store.db_pool.runInteraction('insert', insert_event, i + 200, room3))
        r = self.get_success(self.store.get_rooms_with_many_extremities(5, 5, []))
        self.assertEqual(len(r), 3)
        r = self.get_success(self.store.get_rooms_with_many_extremities(5, 5, [room1]))
        self.assertTrue(room2 in r)
        self.assertTrue(room3 in r)
        self.assertEqual(len(r), 2)
        r = self.get_success(self.store.get_rooms_with_many_extremities(5, 5, [room1, room2]))
        self.assertEqual(r, [room3])
        r = self.get_success(self.store.get_rooms_with_many_extremities(5, 1, [room1]))
        self.assertTrue(r == [room2] or r == [room3])

    def _setup_auth_chain(self, use_chain_cover_index: bool) -> str:
        if False:
            for i in range(10):
                print('nop')
        room_id = '@ROOM:local'

        def store_room(txn: LoggingTransaction) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.store.db_pool.simple_insert_txn(txn, 'rooms', {'room_id': room_id, 'creator': 'room_creator_user_id', 'is_public': True, 'room_version': '6', 'has_auth_chain_index': use_chain_cover_index})
        self.get_success(self.store.db_pool.runInteraction('store_room', store_room))

        def insert_event(txn: LoggingTransaction) -> None:
            if False:
                while True:
                    i = 10
            stream_ordering = 0
            for event_id in AUTH_GRAPH:
                stream_ordering += 1
                depth = DEPTH_GRAPH[event_id]
                self.store.db_pool.simple_insert_txn(txn, table='events', values={'event_id': event_id, 'room_id': room_id, 'depth': depth, 'topological_ordering': depth, 'type': 'm.test', 'processed': True, 'outlier': False, 'stream_ordering': stream_ordering})
            self.persist_events._persist_event_auth_chain_txn(txn, [cast(EventBase, FakeEvent(event_id, room_id, AUTH_GRAPH[event_id])) for event_id in AUTH_GRAPH])
        self.get_success(self.store.db_pool.runInteraction('insert', insert_event))
        return room_id

    @parameterized.expand([(True,), (False,)])
    def test_auth_chain_ids(self, use_chain_cover_index: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        room_id = self._setup_auth_chain(use_chain_cover_index)
        auth_chain_ids = self.get_success(self.store.get_auth_chain_ids(room_id, ['a']))
        self.assertCountEqual(auth_chain_ids, ['e', 'f', 'g', 'h', 'i', 'j', 'k'])
        auth_chain_ids = self.get_success(self.store.get_auth_chain_ids(room_id, ['b']))
        self.assertCountEqual(auth_chain_ids, ['e', 'f', 'g', 'h', 'i', 'j', 'k'])
        auth_chain_ids = self.get_success(self.store.get_auth_chain_ids(room_id, ['a', 'b']))
        self.assertCountEqual(auth_chain_ids, ['e', 'f', 'g', 'h', 'i', 'j', 'k'])
        auth_chain_ids = self.get_success(self.store.get_auth_chain_ids(room_id, ['c']))
        self.assertCountEqual(auth_chain_ids, ['g', 'h', 'i', 'j', 'k'])
        auth_chain_ids = self.get_success(self.store.get_auth_chain_ids(room_id, ['d']))
        self.assertCountEqual(auth_chain_ids, ['f', 'g', 'h', 'i', 'j', 'k'])
        auth_chain_ids = self.get_success(self.store.get_auth_chain_ids(room_id, ['e']))
        self.assertCountEqual(auth_chain_ids, ['f', 'g', 'h', 'i', 'j', 'k'])
        auth_chain_ids = self.get_success(self.store.get_auth_chain_ids(room_id, ['f']))
        self.assertCountEqual(auth_chain_ids, ['g', 'h', 'i', 'j', 'k'])
        auth_chain_ids = self.get_success(self.store.get_auth_chain_ids(room_id, ['g']))
        self.assertCountEqual(auth_chain_ids, ['h', 'i', 'j', 'k'])
        auth_chain_ids = self.get_success(self.store.get_auth_chain_ids(room_id, ['h']))
        self.assertEqual(auth_chain_ids, {'k'})
        auth_chain_ids = self.get_success(self.store.get_auth_chain_ids(room_id, ['i']))
        self.assertEqual(auth_chain_ids, {'j'})
        auth_chain_ids = self.get_success(self.store.get_auth_chain_ids(room_id, ['j']))
        self.assertEqual(auth_chain_ids, set())
        auth_chain_ids = self.get_success(self.store.get_auth_chain_ids(room_id, ['k']))
        self.assertEqual(auth_chain_ids, set())
        auth_chain_ids = self.get_success(self.store.get_auth_chain_ids(room_id, ['b', 'c', 'd']))
        self.assertCountEqual(auth_chain_ids, ['e', 'f', 'g', 'h', 'i', 'j', 'k'])
        auth_chain_ids = self.get_success(self.store.get_auth_chain_ids(room_id, ['h', 'i']))
        self.assertCountEqual(auth_chain_ids, ['k', 'j'])
        auth_chain_ids = self.get_success(self.store.get_auth_chain_ids(room_id, ['b', 'e']))
        self.assertCountEqual(auth_chain_ids, ['e', 'f', 'g', 'h', 'i', 'j', 'k'])
        auth_chain_ids = self.get_success(self.store.get_auth_chain_ids(room_id, ['i'], include_given=True))
        self.assertCountEqual(auth_chain_ids, ['i', 'j'])

    @parameterized.expand([(True,), (False,)])
    def test_auth_difference(self, use_chain_cover_index: bool) -> None:
        if False:
            print('Hello World!')
        room_id = self._setup_auth_chain(use_chain_cover_index)
        self.assert_auth_diff_is_expected(room_id)

    @parameterized.expand([[graph_subset] for graph_subset in get_all_topologically_consistent_subsets(AUTH_GRAPH, AUTH_GRAPH)])
    def test_auth_difference_partial(self, graph_subset: Collection[str]) -> None:
        if False:
            i = 10
            return i + 15
        'Test that if we only have a chain cover index on a partial subset of\n        the room we still get the correct auth chain difference.\n\n        We do this by removing the chain cover index for every valid subset of the\n        graph.\n        '
        room_id = self._setup_auth_chain(True)
        for event_id in graph_subset:
            self.get_success(self.store.db_pool.simple_delete(table='event_auth_chains', keyvalues={'event_id': event_id}, desc='test_auth_difference_partial_remove'))
            self.get_success(self.store.db_pool.simple_insert(table='event_auth_chain_to_calculate', values={'event_id': event_id, 'room_id': room_id, 'type': '', 'state_key': ''}, desc='test_auth_difference_partial_remove'))
        self.assert_auth_diff_is_expected(room_id)

    def assert_auth_diff_is_expected(self, room_id: str) -> None:
        if False:
            print('Hello World!')
        'Assert the auth chain difference returns the correct answers.'
        difference = self.get_success(self.store.get_auth_chain_difference(room_id, [{'a'}, {'b'}]))
        self.assertSetEqual(difference, {'a', 'b'})
        difference = self.get_success(self.store.get_auth_chain_difference(room_id, [{'a'}, {'b'}, {'c'}]))
        self.assertSetEqual(difference, {'a', 'b', 'c', 'e', 'f'})
        difference = self.get_success(self.store.get_auth_chain_difference(room_id, [{'a', 'c'}, {'b'}]))
        self.assertSetEqual(difference, {'a', 'b', 'c'})
        difference = self.get_success(self.store.get_auth_chain_difference(room_id, [{'a', 'c'}, {'b', 'c'}]))
        self.assertSetEqual(difference, {'a', 'b'})
        difference = self.get_success(self.store.get_auth_chain_difference(room_id, [{'a'}, {'b'}, {'d'}]))
        self.assertSetEqual(difference, {'a', 'b', 'd', 'e'})
        difference = self.get_success(self.store.get_auth_chain_difference(room_id, [{'a'}, {'b'}, {'c'}, {'d'}]))
        self.assertSetEqual(difference, {'a', 'b', 'c', 'd', 'e', 'f'})
        difference = self.get_success(self.store.get_auth_chain_difference(room_id, [{'a'}, {'b'}, {'e'}]))
        self.assertSetEqual(difference, {'a', 'b'})
        difference = self.get_success(self.store.get_auth_chain_difference(room_id, [{'a'}]))
        self.assertSetEqual(difference, set())

    def test_auth_difference_partial_cover(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that we correctly handle rooms where not all events have a chain\n        cover calculated. This can happen in some obscure edge cases, including\n        during the background update that calculates the chain cover for old\n        rooms.\n        '
        room_id = '@ROOM:local'
        auth_graph: Dict[str, List[str]] = {'a': ['e'], 'b': ['e'], 'c': ['g', 'i'], 'd': ['f'], 'e': ['f'], 'f': ['g'], 'g': ['h', 'i'], 'h': ['k'], 'i': ['j'], 'k': [], 'j': []}
        depth_map = {'a': 7, 'b': 7, 'c': 4, 'd': 6, 'e': 6, 'f': 5, 'g': 3, 'h': 2, 'i': 2, 'k': 1, 'j': 1}

        def insert_event(txn: LoggingTransaction) -> None:
            if False:
                return 10
            self.store.db_pool.simple_insert_txn(txn, 'rooms', {'room_id': room_id, 'creator': 'room_creator_user_id', 'is_public': True, 'room_version': '6', 'has_auth_chain_index': True})
            stream_ordering = 0
            for event_id in auth_graph:
                stream_ordering += 1
                depth = depth_map[event_id]
                self.store.db_pool.simple_insert_txn(txn, table='events', values={'event_id': event_id, 'room_id': room_id, 'depth': depth, 'topological_ordering': depth, 'type': 'm.test', 'processed': True, 'outlier': False, 'stream_ordering': stream_ordering})
            self.persist_events._persist_event_auth_chain_txn(txn, [cast(EventBase, FakeEvent(event_id, room_id, auth_graph[event_id])) for event_id in auth_graph if event_id != 'b'])
            self.store.db_pool.simple_update_txn(txn, table='rooms', keyvalues={'room_id': room_id}, updatevalues={'has_auth_chain_index': False})
            self.persist_events._persist_event_auth_chain_txn(txn, [cast(EventBase, FakeEvent('b', room_id, auth_graph['b']))])
            self.store.db_pool.simple_update_txn(txn, table='rooms', keyvalues={'room_id': room_id}, updatevalues={'has_auth_chain_index': True})
        self.get_success(self.store.db_pool.runInteraction('insert', insert_event))
        difference = self.get_success(self.store.get_auth_chain_difference(room_id, [{'a'}, {'b'}]))
        self.assertSetEqual(difference, {'a', 'b'})
        difference = self.get_success(self.store.get_auth_chain_difference(room_id, [{'a'}, {'b'}, {'c'}]))
        self.assertSetEqual(difference, {'a', 'b', 'c', 'e', 'f'})
        difference = self.get_success(self.store.get_auth_chain_difference(room_id, [{'a', 'c'}, {'b'}]))
        self.assertSetEqual(difference, {'a', 'b', 'c'})
        difference = self.get_success(self.store.get_auth_chain_difference(room_id, [{'a', 'c'}, {'b', 'c'}]))
        self.assertSetEqual(difference, {'a', 'b'})
        difference = self.get_success(self.store.get_auth_chain_difference(room_id, [{'a'}, {'b'}, {'d'}]))
        self.assertSetEqual(difference, {'a', 'b', 'd', 'e'})
        difference = self.get_success(self.store.get_auth_chain_difference(room_id, [{'a'}, {'b'}, {'c'}, {'d'}]))
        self.assertSetEqual(difference, {'a', 'b', 'c', 'd', 'e', 'f'})
        difference = self.get_success(self.store.get_auth_chain_difference(room_id, [{'a'}, {'b'}, {'e'}]))
        self.assertSetEqual(difference, {'a', 'b'})
        difference = self.get_success(self.store.get_auth_chain_difference(room_id, [{'a'}]))
        self.assertSetEqual(difference, set())

    @parameterized.expand([(room_version,) for room_version in KNOWN_ROOM_VERSIONS.values()])
    def test_prune_inbound_federation_queue(self, room_version: RoomVersion) -> None:
        if False:
            i = 10
            return i + 15
        'Test that pruning of inbound federation queues work'
        room_id = 'some_room_id'

        def prev_event_format(prev_event_id: str) -> Union[Tuple[str, dict], str]:
            if False:
                print('Hello World!')
            'Account for differences in prev_events format across room versions'
            if room_version.event_format == EventFormatVersions.ROOM_V1_V2:
                return (prev_event_id, {})
            return prev_event_id
        self.get_success(self.store.db_pool.simple_insert_many(table='federation_inbound_events_staging', keys=('origin', 'room_id', 'received_ts', 'event_id', 'event_json', 'internal_metadata'), values=[('some_origin', room_id, 0, f'$fake_event_id_{i + 1}', json_encoder.encode({'prev_events': [prev_event_format(f'$fake_event_id_{i}')]}), '{}') for i in range(500)], desc='test_prune_inbound_federation_queue'))
        pruned = self.get_success(self.store.prune_staged_events_in_room(room_id, room_version))
        self.assertTrue(pruned)
        pruned = self.get_success(self.store.prune_staged_events_in_room(room_id, room_version))
        self.assertFalse(pruned)
        count = self.get_success(self.store.db_pool.simple_select_one_onecol(table='federation_inbound_events_staging', keyvalues={'room_id': room_id}, retcol='COUNT(*)', desc='test_prune_inbound_federation_queue'))
        self.assertEqual(count, 1)
        next_staged_event_info = self.get_success(self.store.get_next_staged_event_id_for_room(room_id))
        assert next_staged_event_info
        (_, event_id) = next_staged_event_info
        self.assertEqual(event_id, '$fake_event_id_500')

    def _setup_room_for_backfill_tests(self) -> _BackfillSetupInfo:
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets up a room with various events and backward extremities to test\n        backfill functions against.\n\n        Returns:\n            _BackfillSetupInfo including the `room_id` to test against and\n            `depth_map` of events in the room\n        '
        room_id = '!backfill-room-test:some-host'
        event_graph: Dict[str, List[str]] = {'1': [], '2': ['1'], '3': ['2', 'A'], '4': ['3', 'B'], '5': ['4'], 'A': ['b1', 'b2', 'b3'], 'b1': ['2'], 'b2': ['2'], 'b3': ['2'], 'B': ['b4', 'b5', 'b6'], 'b4': ['3'], 'b5': ['3'], 'b6': ['3']}
        depth_map: Dict[str, int] = {'1': 1, '2': 2, 'b1': 3, 'b2': 3, 'b3': 3, 'A': 4, '3': 5, 'b4': 6, 'b5': 6, 'b6': 6, 'B': 7, '4': 8, '5': 9}
        our_server_events = {'5', '4', 'B', '3', 'A'}
        complete_event_dict_map: Dict[str, JsonDict] = {}
        stream_ordering = 0
        for (event_id, prev_event_ids) in event_graph.items():
            depth = depth_map[event_id]
            complete_event_dict_map[event_id] = {'event_id': event_id, 'type': 'test_regular_type', 'room_id': room_id, 'sender': '@sender', 'prev_event_ids': prev_event_ids, 'auth_event_ids': [], 'origin_server_ts': stream_ordering, 'depth': depth, 'stream_ordering': stream_ordering, 'content': {'body': 'event' + event_id}}
            stream_ordering += 1

        def populate_db(txn: LoggingTransaction) -> None:
            if False:
                return 10
            self.store.db_pool.simple_insert_txn(txn, 'rooms', {'room_id': room_id, 'creator': 'room_creator_user_id', 'is_public': True, 'room_version': '6'})
            for event_id in our_server_events:
                event_dict = complete_event_dict_map[event_id]
                self.store.db_pool.simple_insert_txn(txn, table='events', values={'event_id': event_dict.get('event_id'), 'type': event_dict.get('type'), 'room_id': event_dict.get('room_id'), 'depth': event_dict.get('depth'), 'topological_ordering': event_dict.get('depth'), 'stream_ordering': event_dict.get('stream_ordering'), 'processed': True, 'outlier': False})
            for event_id in our_server_events:
                for prev_event_id in event_graph[event_id]:
                    self.store.db_pool.simple_insert_txn(txn, table='event_edges', values={'event_id': event_id, 'prev_event_id': prev_event_id, 'room_id': room_id})
            prev_events_of_our_events = {prev_event_id for our_server_event in our_server_events for prev_event_id in complete_event_dict_map[our_server_event]['prev_event_ids']}
            backward_extremities = prev_events_of_our_events - our_server_events
            for backward_extremity in backward_extremities:
                self.store.db_pool.simple_insert_txn(txn, table='event_backward_extremities', values={'event_id': backward_extremity, 'room_id': room_id})
        self.get_success(self.store.db_pool.runInteraction('_setup_room_for_backfill_tests_populate_db', populate_db))
        return _BackfillSetupInfo(room_id=room_id, depth_map=depth_map)

    def test_get_backfill_points_in_room(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test to make sure only backfill points that are older and come before\n        the `current_depth` are returned.\n        '
        setup_info = self._setup_room_for_backfill_tests()
        room_id = setup_info.room_id
        depth_map = setup_info.depth_map
        backfill_points = self.get_success(self.store.get_backfill_points_in_room(room_id, depth_map['B'], limit=100))
        backfill_event_ids = [backfill_point[0] for backfill_point in backfill_points]
        self.assertEqual(backfill_event_ids, ['b6', 'b5', 'b4', '2', 'b3', 'b2', 'b1'])
        backfill_points = self.get_success(self.store.get_backfill_points_in_room(room_id, depth_map['A'], limit=100))
        backfill_event_ids = [backfill_point[0] for backfill_point in backfill_points]
        self.assertListEqual(backfill_event_ids, ['b3', 'b2', 'b1'])

    def test_get_backfill_points_in_room_excludes_events_we_have_attempted(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test to make sure that events we have attempted to backfill (and within\n        backoff timeout duration) do not show up as an event to backfill again.\n        '
        setup_info = self._setup_room_for_backfill_tests()
        room_id = setup_info.room_id
        depth_map = setup_info.depth_map
        self.get_success(self.store.record_event_failed_pull_attempt(room_id, 'b5', 'fake cause'))
        self.get_success(self.store.record_event_failed_pull_attempt(room_id, 'b4', 'fake cause'))
        self.get_success(self.store.record_event_failed_pull_attempt(room_id, 'b3', 'fake cause'))
        self.get_success(self.store.record_event_failed_pull_attempt(room_id, 'b2', 'fake cause'))
        backfill_points = self.get_success(self.store.get_backfill_points_in_room(room_id, depth_map['B'], limit=100))
        backfill_event_ids = [backfill_point[0] for backfill_point in backfill_points]
        self.assertEqual(backfill_event_ids, ['b6', '2', 'b1'])

    def test_get_backfill_points_in_room_attempted_event_retry_after_backoff_duration(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test to make sure after we fake attempt to backfill event "b3" many times,\n        we can see retry and see the "b3" again after the backoff timeout duration\n        has exceeded.\n        '
        setup_info = self._setup_room_for_backfill_tests()
        room_id = setup_info.room_id
        depth_map = setup_info.depth_map
        self.get_success(self.store.record_event_failed_pull_attempt(room_id, 'b3', 'fake cause'))
        self.get_success(self.store.record_event_failed_pull_attempt(room_id, 'b1', 'fake cause'))
        self.get_success(self.store.record_event_failed_pull_attempt(room_id, 'b1', 'fake cause'))
        self.get_success(self.store.record_event_failed_pull_attempt(room_id, 'b1', 'fake cause'))
        self.get_success(self.store.record_event_failed_pull_attempt(room_id, 'b1', 'fake cause'))
        self.reactor.advance(datetime.timedelta(hours=2).total_seconds())
        backfill_points = self.get_success(self.store.get_backfill_points_in_room(room_id, depth_map['A'], limit=100))
        backfill_event_ids = [backfill_point[0] for backfill_point in backfill_points]
        self.assertEqual(backfill_event_ids, ['b3', 'b2'])
        self.reactor.advance(datetime.timedelta(hours=20).total_seconds())
        backfill_points = self.get_success(self.store.get_backfill_points_in_room(room_id, depth_map['A'], limit=100))
        backfill_event_ids = [backfill_point[0] for backfill_point in backfill_points]
        self.assertEqual(backfill_event_ids, ['b3', 'b2', 'b1'])

    def test_get_backfill_points_in_room_works_after_many_failed_pull_attempts_that_could_naively_overflow(self) -> None:
        if False:
            return 10
        '\n        A test that reproduces https://github.com/matrix-org/synapse/issues/13929 (Postgres only).\n\n        Test to make sure we can still get backfill points after many failed pull\n        attempts that cause us to backoff to the limit. Even if the backoff formula\n        would tell us to wait for more seconds than can be expressed in a 32 bit\n        signed int.\n        '
        setup_info = self._setup_room_for_backfill_tests()
        room_id = setup_info.room_id
        depth_map = setup_info.depth_map
        for _ in range(10):
            self.get_success(self.store.record_event_failed_pull_attempt(room_id, 'b1', 'fake cause'))
        self.reactor.advance(datetime.timedelta(hours=1100).total_seconds())
        backfill_points = self.get_success(self.store.get_backfill_points_in_room(room_id, depth_map['A'], limit=100))
        backfill_event_ids = [backfill_point[0] for backfill_point in backfill_points]
        self.assertEqual(backfill_event_ids, ['b3', 'b2', 'b1'])

    def test_get_event_ids_with_failed_pull_attempts(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test to make sure we properly get event_ids based on whether they have any\n        failed pull attempts.\n        '
        user_id = self.register_user('alice', 'test')
        tok = self.login('alice', 'test')
        room_id = self.helper.create_room_as(room_creator=user_id, tok=tok)
        self.get_success(self.store.record_event_failed_pull_attempt(room_id, '$failed_event_id1', 'fake cause'))
        self.get_success(self.store.record_event_failed_pull_attempt(room_id, '$failed_event_id2', 'fake cause'))
        event_ids_with_failed_pull_attempts = self.get_success(self.store.get_event_ids_with_failed_pull_attempts(event_ids=['$failed_event_id1', '$fresh_event_id1', '$failed_event_id2', '$fresh_event_id2']))
        self.assertEqual(event_ids_with_failed_pull_attempts, {'$failed_event_id1', '$failed_event_id2'})

    def test_get_event_ids_to_not_pull_from_backoff(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test to make sure only event IDs we should backoff from are returned.\n        '
        user_id = self.register_user('alice', 'test')
        tok = self.login('alice', 'test')
        room_id = self.helper.create_room_as(room_creator=user_id, tok=tok)
        failure_time = self.clock.time_msec()
        self.get_success(self.store.record_event_failed_pull_attempt(room_id, '$failed_event_id', 'fake cause'))
        event_ids_with_backoff = self.get_success(self.store.get_event_ids_to_not_pull_from_backoff(room_id=room_id, event_ids=['$failed_event_id', '$normal_event_id']))
        self.assertEqual(event_ids_with_backoff, {'$failed_event_id': failure_time + 2 * 60 * 60 * 1000})

    def test_get_event_ids_to_not_pull_from_backoff_retry_after_backoff_duration(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test to make sure no event IDs are returned after the backoff duration has\n        elapsed.\n        '
        user_id = self.register_user('alice', 'test')
        tok = self.login('alice', 'test')
        room_id = self.helper.create_room_as(room_creator=user_id, tok=tok)
        self.get_success(self.store.record_event_failed_pull_attempt(room_id, '$failed_event_id', 'fake cause'))
        self.reactor.advance(datetime.timedelta(hours=2).total_seconds())
        event_ids_with_backoff = self.get_success(self.store.get_event_ids_to_not_pull_from_backoff(room_id=room_id, event_ids=['$failed_event_id', '$normal_event_id']))
        self.assertEqual(event_ids_with_backoff, {})

@attr.s(auto_attribs=True)
class FakeEvent:
    event_id: str
    room_id: str
    auth_events: List[str]
    type = 'foo'
    state_key = 'foo'
    internal_metadata = _EventInternalMetadata({})

    def auth_event_ids(self) -> List[str]:
        if False:
            while True:
                i = 10
        return self.auth_events

    def is_state(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True