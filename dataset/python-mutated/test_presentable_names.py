from typing import Iterable, List, Optional, Tuple, cast
from synapse.api.constants import EventTypes, Membership
from synapse.api.room_versions import RoomVersions
from synapse.events import EventBase, FrozenEvent
from synapse.push.presentable_names import calculate_room_name
from synapse.types import StateKey, StateMap
from tests import unittest

class MockDataStore:
    """
    A fake data store which stores a mapping of state key to event content.
    (I.e. the state key is used as the event ID.)
    """

    def __init__(self, events: Iterable[Tuple[StateKey, dict]]):
        if False:
            return 10
        '\n        Args:\n            events: A state map to event contents.\n        '
        self._events = {}
        for (i, (event_id, content)) in enumerate(events):
            self._events[event_id] = FrozenEvent({'event_id': '$event_id', 'type': event_id[0], 'sender': '@user:test', 'state_key': event_id[1], 'room_id': '#room:test', 'content': content, 'origin_server_ts': i}, RoomVersions.V1)

    async def get_event(self, event_id: str, allow_none: bool=False) -> Optional[FrozenEvent]:
        assert allow_none, 'Mock not configured for allow_none = False'
        state_key = cast(Tuple[str, str], tuple(event_id.split('|', 1)))
        return self._events.get(state_key)

    async def get_events(self, event_ids: Iterable[StateKey]) -> StateMap[EventBase]:
        return self._events

class PresentableNamesTestCase(unittest.HomeserverTestCase):
    USER_ID = '@test:test'
    OTHER_USER_ID = '@user:test'

    def _calculate_room_name(self, events: Iterable[Tuple[Tuple[str, str], dict]], user_id: str='', fallback_to_members: bool=True, fallback_to_single_member: bool=True) -> Optional[str]:
        if False:
            print('Hello World!')
        room_state_ids = {k[0]: '|'.join(k[0]) for k in events}
        return self.get_success(calculate_room_name(MockDataStore(events), room_state_ids, user_id or self.USER_ID, fallback_to_members, fallback_to_single_member))

    def test_name(self) -> None:
        if False:
            print('Hello World!')
        'A room name event should be used.'
        events: List[Tuple[Tuple[str, str], dict]] = [((EventTypes.Name, ''), {'name': 'test-name'})]
        self.assertEqual('test-name', self._calculate_room_name(events))
        events = [((EventTypes.Name, ''), {'foo': 1})]
        self.assertEqual('Empty Room', self._calculate_room_name(events))
        events = [((EventTypes.Name, ''), {'name': 1})]
        self.assertEqual(1, self._calculate_room_name(events))

    def test_canonical_alias(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'An canonical alias should be used.'
        events: List[Tuple[Tuple[str, str], dict]] = [((EventTypes.CanonicalAlias, ''), {'alias': '#test-name:test'})]
        self.assertEqual('#test-name:test', self._calculate_room_name(events))
        events = [((EventTypes.CanonicalAlias, ''), {'foo': 1})]
        self.assertEqual('Empty Room', self._calculate_room_name(events))
        events = [((EventTypes.CanonicalAlias, ''), {'alias': 'test-name'})]
        self.assertEqual('Empty Room', self._calculate_room_name(events))

    def test_invite(self) -> None:
        if False:
            print('Hello World!')
        'An invite has special behaviour.'
        events: List[Tuple[Tuple[str, str], dict]] = [((EventTypes.Member, self.USER_ID), {'membership': Membership.INVITE}), ((EventTypes.Member, self.OTHER_USER_ID), {'displayname': 'Other User'})]
        self.assertEqual('Invite from Other User', self._calculate_room_name(events))
        self.assertIsNone(self._calculate_room_name(events, fallback_to_single_member=False))
        self.assertIsNone(self._calculate_room_name(events, fallback_to_members=False))
        events = [((EventTypes.Member, self.USER_ID), {'membership': Membership.INVITE}), ((EventTypes.Member, self.OTHER_USER_ID), {'foo': 1})]
        self.assertEqual('Invite from @user:test', self._calculate_room_name(events))
        events = [((EventTypes.Member, self.USER_ID), {'membership': Membership.INVITE})]
        self.assertEqual('Room Invite', self._calculate_room_name(events))

    def test_no_members(self) -> None:
        if False:
            return 10
        'Behaviour of an empty room.'
        events: List[Tuple[Tuple[str, str], dict]] = []
        self.assertEqual('Empty Room', self._calculate_room_name(events))
        events = [((EventTypes.Member, self.OTHER_USER_ID), {'foo': 1}), ((EventTypes.Member, '@foo:test'), {'membership': 'foo'})]
        self.assertEqual('Empty Room', self._calculate_room_name(events))

    def test_no_other_members(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Behaviour of a room with no other members in it.'
        events = [((EventTypes.Member, self.USER_ID), {'membership': Membership.JOIN, 'displayname': 'Me'})]
        self.assertEqual('Me', self._calculate_room_name(events))
        events = [((EventTypes.Member, self.USER_ID), {'membership': Membership.JOIN})]
        self.assertEqual('@test:test', self._calculate_room_name(events))
        events = [((EventTypes.Member, self.OTHER_USER_ID), {'membership': Membership.JOIN})]
        self.assertEqual('nobody', self._calculate_room_name(events, user_id=self.OTHER_USER_ID))
        events = [((EventTypes.Member, self.OTHER_USER_ID), {'membership': Membership.JOIN}), ((EventTypes.ThirdPartyInvite, self.OTHER_USER_ID), {})]
        self.assertEqual('Inviting email address', self._calculate_room_name(events, user_id=self.OTHER_USER_ID))

    def test_one_other_member(self) -> None:
        if False:
            print('Hello World!')
        'Behaviour of a room with a single other member.'
        events = [((EventTypes.Member, self.USER_ID), {'membership': Membership.JOIN}), ((EventTypes.Member, self.OTHER_USER_ID), {'membership': Membership.JOIN, 'displayname': 'Other User'})]
        self.assertEqual('Other User', self._calculate_room_name(events))
        self.assertIsNone(self._calculate_room_name(events, fallback_to_single_member=False))
        events = [((EventTypes.Member, self.USER_ID), {'membership': Membership.JOIN}), ((EventTypes.Member, self.OTHER_USER_ID), {'membership': Membership.INVITE})]
        self.assertEqual('@user:test', self._calculate_room_name(events))

    def test_other_members(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Behaviour of a room with multiple other members.'
        events = [((EventTypes.Member, self.USER_ID), {'membership': Membership.JOIN}), ((EventTypes.Member, self.OTHER_USER_ID), {'membership': Membership.JOIN, 'displayname': 'Other User'}), ((EventTypes.Member, '@foo:test'), {'membership': Membership.JOIN})]
        self.assertEqual('Other User and @foo:test', self._calculate_room_name(events))
        events.append(((EventTypes.Member, '@fourth:test'), {'membership': Membership.INVITE}))
        self.assertEqual('Other User and 2 others', self._calculate_room_name(events))