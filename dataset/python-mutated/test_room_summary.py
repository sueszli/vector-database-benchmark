from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from unittest import mock
from twisted.internet.defer import ensureDeferred
from twisted.test.proto_helpers import MemoryReactor
from synapse.api.constants import EventContentFields, EventTypes, HistoryVisibility, JoinRules, Membership, RestrictedJoinRuleTypes, RoomTypes
from synapse.api.errors import AuthError, NotFoundError, SynapseError
from synapse.api.room_versions import RoomVersions
from synapse.events import make_event_from_dict
from synapse.federation.transport.client import TransportLayerClient
from synapse.handlers.room_summary import _child_events_comparison_key, _RoomEntry
from synapse.rest import admin
from synapse.rest.client import login, room
from synapse.server import HomeServer
from synapse.types import JsonDict, UserID, create_requester
from synapse.util import Clock
from tests import unittest

def _create_event(room_id: str, order: Optional[Any]=None, origin_server_ts: int=0) -> mock.Mock:
    if False:
        for i in range(10):
            print('nop')
    result = mock.Mock(name=room_id)
    result.room_id = room_id
    result.content = {}
    result.origin_server_ts = origin_server_ts
    if order is not None:
        result.content['order'] = order
    return result

def _order(*events: mock.Mock) -> List[mock.Mock]:
    if False:
        return 10
    return sorted(events, key=_child_events_comparison_key)

class TestSpaceSummarySort(unittest.TestCase):

    def test_no_order_last(self) -> None:
        if False:
            i = 10
            return i + 15
        'An event with no ordering is placed behind those with an ordering.'
        ev1 = _create_event('!abc:test')
        ev2 = _create_event('!xyz:test', 'xyz')
        self.assertEqual([ev2, ev1], _order(ev1, ev2))

    def test_order(self) -> None:
        if False:
            while True:
                i = 10
        'The ordering should be used.'
        ev1 = _create_event('!abc:test', 'xyz')
        ev2 = _create_event('!xyz:test', 'abc')
        self.assertEqual([ev2, ev1], _order(ev1, ev2))

    def test_order_origin_server_ts(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Origin server  is a tie-breaker for ordering.'
        ev1 = _create_event('!abc:test', origin_server_ts=10)
        ev2 = _create_event('!xyz:test', origin_server_ts=30)
        self.assertEqual([ev1, ev2], _order(ev1, ev2))

    def test_order_room_id(self) -> None:
        if False:
            print('Hello World!')
        'Room ID is a final tie-breaker for ordering.'
        ev1 = _create_event('!abc:test')
        ev2 = _create_event('!xyz:test')
        self.assertEqual([ev1, ev2], _order(ev1, ev2))

    def test_invalid_ordering_type(self) -> None:
        if False:
            i = 10
            return i + 15
        'Invalid orderings are considered the same as missing.'
        ev1 = _create_event('!abc:test', 1)
        ev2 = _create_event('!xyz:test', 'xyz')
        self.assertEqual([ev2, ev1], _order(ev1, ev2))
        ev1 = _create_event('!abc:test', {})
        self.assertEqual([ev2, ev1], _order(ev1, ev2))
        ev1 = _create_event('!abc:test', [])
        self.assertEqual([ev2, ev1], _order(ev1, ev2))
        ev1 = _create_event('!abc:test', True)
        self.assertEqual([ev2, ev1], _order(ev1, ev2))

    def test_invalid_ordering_value(self) -> None:
        if False:
            while True:
                i = 10
        'Invalid orderings are considered the same as missing.'
        ev1 = _create_event('!abc:test', 'foo\n')
        ev2 = _create_event('!xyz:test', 'xyz')
        self.assertEqual([ev2, ev1], _order(ev1, ev2))
        ev1 = _create_event('!abc:test', 'a' * 51)
        self.assertEqual([ev2, ev1], _order(ev1, ev2))

class SpaceSummaryTestCase(unittest.HomeserverTestCase):
    servlets = [admin.register_servlets_for_client_rest_resource, room.register_servlets, login.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            while True:
                i = 10
        self.hs = hs
        self.handler = self.hs.get_room_summary_handler()
        self.user = self.register_user('user', 'pass')
        self.token = self.login('user', 'pass')
        self.space = self.helper.create_room_as(self.user, tok=self.token, extra_content={'creation_content': {EventContentFields.ROOM_TYPE: RoomTypes.SPACE}})
        self.room = self.helper.create_room_as(self.user, tok=self.token)
        self._add_child(self.space, self.room, self.token)

    def _add_child(self, space_id: str, room_id: str, token: str, order: Optional[str]=None, via: Optional[List[str]]=None) -> None:
        if False:
            print('Hello World!')
        'Add a child room to a space.'
        if via is None:
            via = [self.hs.hostname]
        content: JsonDict = {'via': via}
        if order is not None:
            content['order'] = order
        self.helper.send_state(space_id, event_type=EventTypes.SpaceChild, body=content, tok=token, state_key=room_id)

    def _assert_hierarchy(self, result: JsonDict, rooms_and_children: Iterable[Tuple[str, Iterable[str]]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Assert that the expected room IDs are in the response.\n\n        Args:\n            result: The result from the API call.\n            rooms_and_children: An iterable of tuples where each tuple is:\n                The expected room ID.\n                The expected IDs of any children rooms.\n        '
        result_room_ids = []
        result_children_ids = []
        for result_room in result['rooms']:
            self.assertNotIn('allowed_room_ids', result_room)
            result_room_ids.append(result_room['room_id'])
            result_children_ids.append([(result_room['room_id'], cs['state_key']) for cs in result_room['children_state']])
        room_ids = []
        children_ids = []
        for (room_id, children) in rooms_and_children:
            room_ids.append(room_id)
            children_ids.append([(room_id, child_id) for child_id in children])
        self.assertEqual(result_room_ids, room_ids)
        self.assertEqual(result_children_ids, children_ids)

    def _poke_fed_invite(self, room_id: str, from_user: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Creates a invite (as if received over federation) for the room from the\n        given hostname.\n\n        Args:\n            room_id: The room ID to issue an invite for.\n            fed_hostname: The user to invite from.\n        '
        fed_handler = self.hs.get_federation_handler()
        fed_hostname = UserID.from_string(from_user).domain
        event = make_event_from_dict({'room_id': room_id, 'event_id': '!abcd:' + fed_hostname, 'type': EventTypes.Member, 'sender': from_user, 'state_key': self.user, 'content': {'membership': Membership.INVITE}, 'prev_events': [], 'auth_events': [], 'depth': 1, 'origin_server_ts': 1234})
        self.get_success(fed_handler.on_invite_request(fed_hostname, event, RoomVersions.V6))

    def test_simple_space(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test a simple space with a single room.'
        expected = [(self.space, [self.room]), (self.room, ())]
        result = self.get_success(self.handler.get_room_hierarchy(create_requester(self.user), self.space))
        self._assert_hierarchy(result, expected)

    def test_large_space(self) -> None:
        if False:
            return 10
        'Test a space with a large number of rooms.'
        rooms = [self.room]
        for _ in range(55):
            room = self.helper.create_room_as(self.user, tok=self.token)
            self._add_child(self.space, room, self.token)
            rooms.append(room)
        expected = [(self.space, rooms)] + [(room, []) for room in rooms]
        result = self.get_success(self.handler.get_room_hierarchy(create_requester(self.user), self.space))
        result2 = self.get_success(self.handler.get_room_hierarchy(create_requester(self.user), self.space, from_token=result['next_batch']))
        result['rooms'] += result2['rooms']
        self._assert_hierarchy(result, expected)

    def test_visibility(self) -> None:
        if False:
            return 10
        'A user not in a space cannot inspect it.'
        user2 = self.register_user('user2', 'pass')
        token2 = self.login('user2', 'pass')
        expected = [(self.space, [self.room]), (self.room, ())]
        result = self.get_success(self.handler.get_room_hierarchy(create_requester(user2), self.space))
        self._assert_hierarchy(result, expected)
        self.helper.send_state(self.space, event_type=EventTypes.JoinRules, body={'join_rule': JoinRules.INVITE}, tok=self.token)
        self.get_failure(self.handler.get_room_hierarchy(create_requester(user2), self.space), AuthError)
        self.helper.send_state(self.space, event_type=EventTypes.RoomHistoryVisibility, body={'history_visibility': HistoryVisibility.WORLD_READABLE}, tok=self.token)
        result = self.get_success(self.handler.get_room_hierarchy(create_requester(user2), self.space))
        self._assert_hierarchy(result, expected)
        self.helper.send_state(self.space, event_type=EventTypes.RoomHistoryVisibility, body={'history_visibility': HistoryVisibility.JOINED}, tok=self.token)
        self.get_failure(self.handler.get_room_hierarchy(create_requester(user2), self.space), AuthError)
        self.helper.invite(self.space, targ=user2, tok=self.token)
        self.helper.join(self.space, user2, tok=token2)
        result = self.get_success(self.handler.get_room_hierarchy(create_requester(user2), self.space))
        self._assert_hierarchy(result, expected)
        self.get_failure(self.handler.get_room_hierarchy(create_requester(user2), '#not-a-space:' + self.hs.hostname), AuthError)

    def test_room_hierarchy_cache(self) -> None:
        if False:
            return 10
        'In-flight room hierarchy requests are deduplicated.'
        deferred1 = ensureDeferred(self.handler.get_room_hierarchy(create_requester(self.user), self.space))
        deferred2 = ensureDeferred(self.handler.get_room_hierarchy(create_requester(self.user), self.space))
        result1 = self.get_success(deferred1)
        result2 = self.get_success(deferred2)
        expected = [(self.space, [self.room]), (self.room, ())]
        self._assert_hierarchy(result1, expected)
        self._assert_hierarchy(result2, expected)
        self.assertIs(result1, result2)
        result3 = self.get_success(self.handler.get_room_hierarchy(create_requester(self.user), self.space))
        self._assert_hierarchy(result3, expected)
        self.assertIsNot(result1, result3)

    def test_room_hierarchy_cache_sharing(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Room hierarchy responses for different users are not shared.'
        user2 = self.register_user('user2', 'pass')
        self.helper.send_state(self.room, event_type=EventTypes.JoinRules, body={'join_rule': JoinRules.INVITE}, tok=self.token)
        deferred1 = ensureDeferred(self.handler.get_room_hierarchy(create_requester(self.user), self.space))
        deferred2 = ensureDeferred(self.handler.get_room_hierarchy(create_requester(user2), self.space))
        result1 = self.get_success(deferred1)
        result2 = self.get_success(deferred2)
        self._assert_hierarchy(result1, [(self.space, [self.room]), (self.room, ())])
        self._assert_hierarchy(result2, [(self.space, [self.room])])

    def _create_room_with_join_rule(self, join_rule: str, room_version: Optional[str]=None, **extra_content: Any) -> str:
        if False:
            while True:
                i = 10
        'Create a room with the given join rule and add it to the space.'
        room_id = self.helper.create_room_as(self.user, room_version=room_version, tok=self.token, extra_content={'initial_state': [{'type': EventTypes.JoinRules, 'state_key': '', 'content': {'join_rule': join_rule, **extra_content}}]})
        self._add_child(self.space, room_id, self.token)
        return room_id

    def test_filtering(self) -> None:
        if False:
            print('Hello World!')
        '\n        Rooms should be properly filtered to only include rooms the user has access to.\n        '
        user2 = self.register_user('user2', 'pass')
        token2 = self.login('user2', 'pass')
        public_room = self._create_room_with_join_rule(JoinRules.PUBLIC)
        knock_room = self._create_room_with_join_rule(JoinRules.KNOCK, room_version=RoomVersions.V7.identifier)
        not_invited_room = self._create_room_with_join_rule(JoinRules.INVITE)
        invited_room = self._create_room_with_join_rule(JoinRules.INVITE)
        self.helper.invite(invited_room, targ=user2, tok=self.token)
        restricted_room = self._create_room_with_join_rule(JoinRules.RESTRICTED, room_version=RoomVersions.V8.identifier, allow=[])
        restricted_accessible_room = self._create_room_with_join_rule(JoinRules.RESTRICTED, room_version=RoomVersions.V8.identifier, allow=[{'type': RestrictedJoinRuleTypes.ROOM_MEMBERSHIP, 'room_id': self.space, 'via': [self.hs.hostname]}])
        world_readable_room = self._create_room_with_join_rule(JoinRules.INVITE)
        self.helper.send_state(world_readable_room, event_type=EventTypes.RoomHistoryVisibility, body={'history_visibility': HistoryVisibility.WORLD_READABLE}, tok=self.token)
        joined_room = self._create_room_with_join_rule(JoinRules.INVITE)
        self.helper.invite(joined_room, targ=user2, tok=self.token)
        self.helper.join(joined_room, user2, tok=token2)
        self.helper.join(self.space, user2, tok=token2)
        expected = [(self.space, [self.room, public_room, knock_room, not_invited_room, invited_room, restricted_room, restricted_accessible_room, world_readable_room, joined_room]), (self.room, ()), (public_room, ()), (knock_room, ()), (invited_room, ()), (restricted_accessible_room, ()), (world_readable_room, ()), (joined_room, ())]
        result = self.get_success(self.handler.get_room_hierarchy(create_requester(user2), self.space))
        self._assert_hierarchy(result, expected)

    def test_complex_space(self) -> None:
        if False:
            print('Hello World!')
        '\n        Create a "complex" space to see how it handles things like loops and subspaces.\n        '
        user2 = self.register_user('user2', 'pass')
        token2 = self.login('user2', 'pass')
        room2 = self.helper.create_room_as(user2, is_public=False, tok=token2)
        self._add_child(self.space, room2, self.token)
        subspace = self.helper.create_room_as(self.user, tok=self.token, extra_content={'creation_content': {EventContentFields.ROOM_TYPE: RoomTypes.SPACE}})
        subroom = self.helper.create_room_as(self.user, tok=self.token)
        self._add_child(self.space, subspace, token=self.token)
        self._add_child(subspace, subroom, token=self.token)
        self._add_child(subspace, self.room, token=self.token)
        self._add_child(subspace, room2, self.token)
        expected = [(self.space, [self.room, room2, subspace]), (self.room, ()), (subspace, [subroom, self.room, room2]), (subroom, ())]
        result = self.get_success(self.handler.get_room_hierarchy(create_requester(self.user), self.space))
        self._assert_hierarchy(result, expected)

    def test_pagination(self) -> None:
        if False:
            print('Hello World!')
        'Test simple pagination works.'
        room_ids = []
        for i in range(1, 10):
            room = self.helper.create_room_as(self.user, tok=self.token)
            self._add_child(self.space, room, self.token, order=str(i))
            room_ids.append(room)
        room_ids.append(self.room)
        result = self.get_success(self.handler.get_room_hierarchy(create_requester(self.user), self.space, limit=7))
        expected: List[Tuple[str, Iterable[str]]] = [(self.space, room_ids)]
        expected += [(room_id, ()) for room_id in room_ids[:6]]
        self._assert_hierarchy(result, expected)
        self.assertIn('next_batch', result)
        result = self.get_success(self.handler.get_room_hierarchy(create_requester(self.user), self.space, limit=5, from_token=result['next_batch']))
        expected = [(room_id, ()) for room_id in room_ids[6:]]
        self._assert_hierarchy(result, expected)
        self.assertNotIn('next_batch', result)

    def test_invalid_pagination_token(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'An invalid pagination token, or changing other parameters, shoudl be rejected.'
        room_ids = []
        for i in range(1, 10):
            room = self.helper.create_room_as(self.user, tok=self.token)
            self._add_child(self.space, room, self.token, order=str(i))
            room_ids.append(room)
        room_ids.append(self.room)
        result = self.get_success(self.handler.get_room_hierarchy(create_requester(self.user), self.space, limit=7))
        self.assertIn('next_batch', result)
        self.get_failure(self.handler.get_room_hierarchy(create_requester(self.user), self.room, from_token=result['next_batch']), SynapseError)
        self.get_failure(self.handler.get_room_hierarchy(create_requester(self.user), self.space, suggested_only=True, from_token=result['next_batch']), SynapseError)
        self.get_failure(self.handler.get_room_hierarchy(create_requester(self.user), self.space, max_depth=0, from_token=result['next_batch']), SynapseError)
        self.get_failure(self.handler.get_room_hierarchy(create_requester(self.user), self.space, from_token='foo'), SynapseError)

    def test_max_depth(self) -> None:
        if False:
            i = 10
            return i + 15
        'Create a deep tree to test the max depth against.'
        spaces = [self.space]
        rooms = [self.room]
        for _ in range(5):
            spaces.append(self.helper.create_room_as(self.user, tok=self.token, extra_content={'creation_content': {EventContentFields.ROOM_TYPE: RoomTypes.SPACE}}))
            self._add_child(spaces[-2], spaces[-1], self.token)
            rooms.append(self.helper.create_room_as(self.user, tok=self.token))
            self._add_child(spaces[-1], rooms[-1], self.token)
        result = self.get_success(self.handler.get_room_hierarchy(create_requester(self.user), self.space, max_depth=0))
        expected: List[Tuple[str, Iterable[str]]] = [(spaces[0], [rooms[0], spaces[1]])]
        self._assert_hierarchy(result, expected)
        result = self.get_success(self.handler.get_room_hierarchy(create_requester(self.user), self.space, max_depth=1))
        expected += [(rooms[0], ()), (spaces[1], [rooms[1], spaces[2]])]
        self._assert_hierarchy(result, expected)
        result = self.get_success(self.handler.get_room_hierarchy(create_requester(self.user), self.space, max_depth=3))
        expected += [(rooms[1], ()), (spaces[2], [rooms[2], spaces[3]]), (rooms[2], ()), (spaces[3], [rooms[3], spaces[4]])]
        self._assert_hierarchy(result, expected)

    def test_unknown_room_version(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        If a room with an unknown room version is encountered it should not cause\n        the entire summary to skip.\n        '
        self.get_success(self.hs.get_datastores().main.db_pool.simple_update('rooms', keyvalues={'room_id': self.room}, updatevalues={'room_version': 'unknown-room-version'}, desc='updated-room-version'))
        self.hs.get_datastores().main.get_room_version_id.invalidate((self.room,))
        expected = [(self.space, [self.room])]
        result = self.get_success(self.handler.get_room_hierarchy(create_requester(self.user), self.space))
        self._assert_hierarchy(result, expected)

    def test_fed_complex(self) -> None:
        if False:
            return 10
        '\n        Return data over federation and ensure that it is handled properly.\n        '
        fed_hostname = self.hs.hostname + '2'
        subspace = '#subspace:' + fed_hostname
        subroom = '#subroom:' + fed_hostname
        requested_room_entry = _RoomEntry(subspace, {'room_id': subspace, 'world_readable': True, 'room_type': RoomTypes.SPACE}, [{'type': EventTypes.SpaceChild, 'room_id': subspace, 'state_key': subroom, 'content': {'via': [fed_hostname]}}])
        child_room = {'room_id': subroom, 'world_readable': True}

        async def summarize_remote_room_hierarchy(_self: Any, room: Any, suggested_only: bool) -> Tuple[Optional[_RoomEntry], Dict[str, JsonDict], Set[str]]:
            return (requested_room_entry, {subroom: child_room}, set())
        self._add_child(self.space, subspace, self.token)
        expected = [(self.space, [self.room, subspace]), (self.room, ()), (subspace, [subroom]), (subroom, ())]
        with mock.patch('synapse.handlers.room_summary.RoomSummaryHandler._summarize_remote_room_hierarchy', new=summarize_remote_room_hierarchy):
            result = self.get_success(self.handler.get_room_hierarchy(create_requester(self.user), self.space))
        self._assert_hierarchy(result, expected)

    def test_fed_filtering(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Rooms returned over federation should be properly filtered to only include\n        rooms the user has access to.\n        '
        fed_hostname = self.hs.hostname + '2'
        subspace = '#subspace:' + fed_hostname
        public_room = '#public:' + fed_hostname
        knock_room = '#knock:' + fed_hostname
        not_invited_room = '#not_invited:' + fed_hostname
        invited_room = '#invited:' + fed_hostname
        restricted_room = '#restricted:' + fed_hostname
        restricted_accessible_room = '#restricted_accessible:' + fed_hostname
        world_readable_room = '#world_readable:' + fed_hostname
        joined_room = self.helper.create_room_as(self.user, tok=self.token)
        self._poke_fed_invite(invited_room, '@remote:' + fed_hostname)
        children_rooms = ((public_room, {'room_id': public_room, 'world_readable': False, 'join_rule': JoinRules.PUBLIC}), (knock_room, {'room_id': knock_room, 'world_readable': False, 'join_rule': JoinRules.KNOCK}), (not_invited_room, {'room_id': not_invited_room, 'world_readable': False, 'join_rule': JoinRules.INVITE}), (invited_room, {'room_id': invited_room, 'world_readable': False, 'join_rule': JoinRules.INVITE}), (restricted_room, {'room_id': restricted_room, 'world_readable': False, 'join_rule': JoinRules.RESTRICTED, 'allowed_room_ids': []}), (restricted_accessible_room, {'room_id': restricted_accessible_room, 'world_readable': False, 'join_rule': JoinRules.RESTRICTED, 'allowed_room_ids': [self.room]}), (world_readable_room, {'room_id': world_readable_room, 'world_readable': True, 'join_rule': JoinRules.INVITE}), (joined_room, {'room_id': joined_room, 'world_readable': False, 'join_rule': JoinRules.INVITE}))
        subspace_room_entry = _RoomEntry(subspace, {'room_id': subspace, 'world_readable': True}, [{'type': EventTypes.SpaceChild, 'room_id': subspace, 'state_key': room_id, 'content': {'via': [fed_hostname]}} for (room_id, _) in children_rooms])

        async def summarize_remote_room_hierarchy(_self: Any, room: Any, suggested_only: bool) -> Tuple[Optional[_RoomEntry], Dict[str, JsonDict], Set[str]]:
            return (subspace_room_entry, dict(children_rooms), set())
        self._add_child(self.space, subspace, self.token)
        expected = [(self.space, [self.room, subspace]), (self.room, ()), (subspace, [public_room, knock_room, not_invited_room, invited_room, restricted_room, restricted_accessible_room, world_readable_room, joined_room]), (public_room, ()), (knock_room, ()), (invited_room, ()), (restricted_accessible_room, ()), (world_readable_room, ()), (joined_room, ())]
        with mock.patch('synapse.handlers.room_summary.RoomSummaryHandler._summarize_remote_room_hierarchy', new=summarize_remote_room_hierarchy):
            result = self.get_success(self.handler.get_room_hierarchy(create_requester(self.user), self.space))
        self._assert_hierarchy(result, expected)

    def test_fed_invited(self) -> None:
        if False:
            return 10
        '\n        A room which the user was invited to should be included in the response.\n\n        This differs from test_fed_filtering in that the room itself is being\n        queried over federation, instead of it being included as a sub-room of\n        a space in the response.\n        '
        fed_hostname = self.hs.hostname + '2'
        fed_room = '#subroom:' + fed_hostname
        self._poke_fed_invite(fed_room, '@remote:' + fed_hostname)
        fed_room_entry = _RoomEntry(fed_room, {'room_id': fed_room, 'world_readable': False, 'join_rule': JoinRules.INVITE})

        async def summarize_remote_room_hierarchy(_self: Any, room: Any, suggested_only: bool) -> Tuple[Optional[_RoomEntry], Dict[str, JsonDict], Set[str]]:
            return (fed_room_entry, {}, set())
        self._add_child(self.space, fed_room, self.token)
        expected = [(self.space, [self.room, fed_room]), (self.room, ()), (fed_room, ())]
        with mock.patch('synapse.handlers.room_summary.RoomSummaryHandler._summarize_remote_room_hierarchy', new=summarize_remote_room_hierarchy):
            result = self.get_success(self.handler.get_room_hierarchy(create_requester(self.user), self.space))
        self._assert_hierarchy(result, expected)

    def test_fed_caching(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Federation `/hierarchy` responses should be cached.\n        '
        fed_hostname = self.hs.hostname + '2'
        fed_subspace = '#space:' + fed_hostname
        fed_room = '#room:' + fed_hostname
        self._add_child(self.space, fed_subspace, self.token, via=[fed_hostname])
        federation_requests = 0

        async def get_room_hierarchy(_self: TransportLayerClient, destination: str, room_id: str, suggested_only: bool) -> JsonDict:
            nonlocal federation_requests
            federation_requests += 1
            return {'room': {'room_id': fed_subspace, 'world_readable': True, 'room_type': RoomTypes.SPACE, 'children_state': [{'type': EventTypes.SpaceChild, 'room_id': fed_subspace, 'state_key': fed_room, 'content': {'via': [fed_hostname]}}]}, 'children': [{'room_id': fed_room, 'world_readable': True}], 'inaccessible_children': []}
        expected = [(self.space, [self.room, fed_subspace]), (self.room, ()), (fed_subspace, [fed_room]), (fed_room, ())]
        with mock.patch('synapse.federation.transport.client.TransportLayerClient.get_room_hierarchy', new=get_room_hierarchy):
            result = self.get_success(self.handler.get_room_hierarchy(create_requester(self.user), self.space))
            self.assertEqual(federation_requests, 1)
            self._assert_hierarchy(result, expected)
            result = self.get_success(self.handler.get_room_hierarchy(create_requester(self.user), self.space))
            self.assertEqual(federation_requests, 1)
            self._assert_hierarchy(result, expected)
            self.reactor.advance(5 * 60 + 1)
            result = self.get_success(self.handler.get_room_hierarchy(create_requester(self.user), self.space))
            self.assertEqual(federation_requests, 2)
            self._assert_hierarchy(result, expected)

class RoomSummaryTestCase(unittest.HomeserverTestCase):
    servlets = [admin.register_servlets_for_client_rest_resource, room.register_servlets, login.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            print('Hello World!')
        self.hs = hs
        self.handler = self.hs.get_room_summary_handler()
        self.user = self.register_user('user', 'pass')
        self.token = self.login('user', 'pass')
        self.room = self.helper.create_room_as(self.user, tok=self.token)
        self.helper.send_state(self.room, event_type=EventTypes.JoinRules, body={'join_rule': JoinRules.INVITE}, tok=self.token)

    def test_own_room(self) -> None:
        if False:
            return 10
        'Test a simple room created by the requester.'
        result = self.get_success(self.handler.get_room_summary(self.user, self.room))
        self.assertEqual(result.get('room_id'), self.room)

    def test_visibility(self) -> None:
        if False:
            while True:
                i = 10
        'A user not in a private room cannot get its summary.'
        user2 = self.register_user('user2', 'pass')
        token2 = self.login('user2', 'pass')
        self.get_failure(self.handler.get_room_summary(user2, self.room), NotFoundError)
        self.helper.send_state(self.room, event_type=EventTypes.RoomHistoryVisibility, body={'history_visibility': HistoryVisibility.WORLD_READABLE}, tok=self.token)
        result = self.get_success(self.handler.get_room_summary(user2, self.room))
        self.assertEqual(result.get('room_id'), self.room)
        self.helper.send_state(self.room, event_type=EventTypes.RoomHistoryVisibility, body={'history_visibility': HistoryVisibility.JOINED}, tok=self.token)
        self.get_failure(self.handler.get_room_summary(user2, self.room), NotFoundError)
        self.helper.send_state(self.room, event_type=EventTypes.JoinRules, body={'join_rule': JoinRules.PUBLIC}, tok=self.token)
        result = self.get_success(self.handler.get_room_summary(user2, self.room))
        self.assertEqual(result.get('room_id'), self.room)
        self.helper.join(self.room, user2, tok=token2)
        self.helper.send_state(self.room, event_type=EventTypes.JoinRules, body={'join_rule': JoinRules.INVITE}, tok=self.token)
        result = self.get_success(self.handler.get_room_summary(user2, self.room))
        self.assertEqual(result.get('room_id'), self.room)

    def test_fed(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Return data over federation and ensure that it is handled properly.\n        '
        fed_hostname = self.hs.hostname + '2'
        fed_room = '#fed_room:' + fed_hostname
        requested_room_entry = _RoomEntry(fed_room, {'room_id': fed_room, 'world_readable': True})

        async def summarize_remote_room_hierarchy(_self: Any, room: Any, suggested_only: bool) -> Tuple[Optional[_RoomEntry], Dict[str, JsonDict], Set[str]]:
            return (requested_room_entry, {}, set())
        with mock.patch('synapse.handlers.room_summary.RoomSummaryHandler._summarize_remote_room_hierarchy', new=summarize_remote_room_hierarchy):
            result = self.get_success(self.handler.get_room_summary(self.user, fed_room, remote_room_hosts=[fed_hostname]))
        self.assertEqual(result.get('room_id'), fed_room)