import unittest
from typing import Any, Collection, Dict, Iterable, List, Optional
from parameterized import parameterized
from synapse import event_auth
from synapse.api.constants import EventContentFields
from synapse.api.errors import AuthError, SynapseError
from synapse.api.room_versions import EventFormatVersions, RoomVersion, RoomVersions
from synapse.events import EventBase, make_event_from_dict
from synapse.storage.databases.main.events_worker import EventRedactBehaviour
from synapse.types import JsonDict, get_domain_from_id
from tests.test_utils import get_awaitable_result

class _StubEventSourceStore:
    """A stub implementation of the EventSourceStore"""

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self._store: Dict[str, EventBase] = {}

    def add_event(self, event: EventBase) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._store[event.event_id] = event

    def add_events(self, events: Iterable[EventBase]) -> None:
        if False:
            while True:
                i = 10
        for event in events:
            self._store[event.event_id] = event

    async def get_events(self, event_ids: Collection[str], redact_behaviour: EventRedactBehaviour, get_prev_content: bool=False, allow_rejected: bool=False) -> Dict[str, EventBase]:
        assert allow_rejected
        assert not get_prev_content
        assert redact_behaviour == EventRedactBehaviour.as_is
        results = {}
        for e in event_ids:
            if e in self._store:
                results[e] = self._store[e]
        return results

class EventAuthTestCase(unittest.TestCase):

    def test_rejected_auth_events(self) -> None:
        if False:
            print('Hello World!')
        '\n        Events that refer to rejected events in their auth events are rejected\n        '
        creator = '@creator:example.com'
        auth_events = [_create_event(RoomVersions.V9, creator), _join_event(RoomVersions.V9, creator)]
        event_store = _StubEventSourceStore()
        event_store.add_events(auth_events)
        event = _random_state_event(RoomVersions.V9, creator, auth_events)
        get_awaitable_result(event_auth.check_state_independent_auth_rules(event_store, event))
        event_auth.check_state_dependent_auth_rules(event, auth_events)
        rejected_join_rules = _join_rules_event(RoomVersions.V9, creator, 'public')
        rejected_join_rules.rejected_reason = 'stinky'
        auth_events.append(rejected_join_rules)
        event_store.add_event(rejected_join_rules)
        with self.assertRaises(AuthError):
            get_awaitable_result(event_auth.check_state_independent_auth_rules(event_store, _random_state_event(RoomVersions.V9, creator)))
        auth_events.append(_join_rules_event(RoomVersions.V9, creator, 'public'))
        event_store.add_event(rejected_join_rules)
        with self.assertRaises(AuthError):
            get_awaitable_result(event_auth.check_state_independent_auth_rules(event_store, _random_state_event(RoomVersions.V9, creator)))

    def test_create_event_with_prev_events(self) -> None:
        if False:
            print('Hello World!')
        'A create event with prev_events should be rejected\n\n        https://spec.matrix.org/v1.3/rooms/v9/#authorization-rules\n        1: If type is m.room.create:\n            1. If it has any previous events, reject.\n        '
        creator = f'@creator:{TEST_DOMAIN}'
        good_event = make_event_from_dict({'room_id': TEST_ROOM_ID, 'type': 'm.room.create', 'state_key': '', 'sender': creator, 'content': {'creator': creator, 'room_version': RoomVersions.V9.identifier}, 'auth_events': [], 'prev_events': []}, room_version=RoomVersions.V9)
        bad_event = make_event_from_dict({**good_event.get_dict(), 'prev_events': ['$fakeevent']}, room_version=RoomVersions.V9)
        event_store = _StubEventSourceStore()
        get_awaitable_result(event_auth.check_state_independent_auth_rules(event_store, good_event))
        with self.assertRaises(AuthError):
            get_awaitable_result(event_auth.check_state_independent_auth_rules(event_store, bad_event))

    def test_duplicate_auth_events(self) -> None:
        if False:
            return 10
        'Events with duplicate auth_events should be rejected\n\n        https://spec.matrix.org/v1.3/rooms/v9/#authorization-rules\n        2. Reject if event has auth_events that:\n            1. have duplicate entries for a given type and state_key pair\n        '
        creator = '@creator:example.com'
        create_event = _create_event(RoomVersions.V9, creator)
        join_event1 = _join_event(RoomVersions.V9, creator)
        pl_event = _power_levels_event(RoomVersions.V9, creator, {'state_default': 30, 'users': {'creator': 100}})
        join_event2 = _join_event(RoomVersions.V9, creator)
        event_store = _StubEventSourceStore()
        event_store.add_events([create_event, join_event1, join_event2, pl_event])
        good_event = _random_state_event(RoomVersions.V9, creator, [create_event, join_event2, pl_event])
        bad_event = _random_state_event(RoomVersions.V9, creator, [create_event, join_event1, join_event2, pl_event])
        bad_event2 = _random_state_event(RoomVersions.V9, creator, [create_event, join_event2, join_event2, pl_event])
        get_awaitable_result(event_auth.check_state_independent_auth_rules(event_store, good_event))
        with self.assertRaises(AuthError):
            get_awaitable_result(event_auth.check_state_independent_auth_rules(event_store, bad_event))
        with self.assertRaises(AuthError):
            get_awaitable_result(event_auth.check_state_independent_auth_rules(event_store, bad_event2))

    def test_unexpected_auth_events(self) -> None:
        if False:
            i = 10
            return i + 15
        'Events with excess auth_events should be rejected\n\n        https://spec.matrix.org/v1.3/rooms/v9/#authorization-rules\n        2. Reject if event has auth_events that:\n           2. have entries whose type and state_key don’t match those specified by the\n              auth events selection algorithm described in the server specification.\n        '
        creator = '@creator:example.com'
        create_event = _create_event(RoomVersions.V9, creator)
        join_event = _join_event(RoomVersions.V9, creator)
        pl_event = _power_levels_event(RoomVersions.V9, creator, {'state_default': 30, 'users': {'creator': 100}})
        join_rules_event = _join_rules_event(RoomVersions.V9, creator, 'public')
        event_store = _StubEventSourceStore()
        event_store.add_events([create_event, join_event, pl_event, join_rules_event])
        good_event = _random_state_event(RoomVersions.V9, creator, [create_event, join_event, pl_event])
        bad_event = _random_state_event(RoomVersions.V9, creator, [create_event, join_event, pl_event, join_rules_event])
        get_awaitable_result(event_auth.check_state_independent_auth_rules(event_store, good_event))
        with self.assertRaises(AuthError):
            get_awaitable_result(event_auth.check_state_independent_auth_rules(event_store, bad_event))

    def test_random_users_cannot_send_state_before_first_pl(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Check that, before the first PL lands, the creator is the only user\n        that can send a state event.\n        '
        creator = '@creator:example.com'
        joiner = '@joiner:example.com'
        auth_events = [_create_event(RoomVersions.V1, creator), _join_event(RoomVersions.V1, creator), _join_event(RoomVersions.V1, joiner)]
        event_auth.check_state_dependent_auth_rules(_random_state_event(RoomVersions.V1, creator), auth_events)
        self.assertRaises(AuthError, event_auth.check_state_dependent_auth_rules, _random_state_event(RoomVersions.V1, joiner), auth_events)

    def test_state_default_level(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Check that users above the state_default level can send state and\n        those below cannot\n        '
        creator = '@creator:example.com'
        pleb = '@joiner:example.com'
        king = '@joiner2:example.com'
        auth_events = [_create_event(RoomVersions.V1, creator), _join_event(RoomVersions.V1, creator), _power_levels_event(RoomVersions.V1, creator, {'state_default': '30', 'users': {pleb: '29', king: '30'}}), _join_event(RoomVersions.V1, pleb), _join_event(RoomVersions.V1, king)]
        (self.assertRaises(AuthError, event_auth.check_state_dependent_auth_rules, _random_state_event(RoomVersions.V1, pleb), auth_events),)
        event_auth.check_state_dependent_auth_rules(_random_state_event(RoomVersions.V1, king), auth_events)

    def test_alias_event(self) -> None:
        if False:
            return 10
        'Alias events have special behavior up through room version 6.'
        creator = '@creator:example.com'
        other = '@other:example.com'
        auth_events = [_create_event(RoomVersions.V1, creator), _join_event(RoomVersions.V1, creator)]
        event_auth.check_state_dependent_auth_rules(_alias_event(RoomVersions.V1, creator), auth_events)
        with self.assertRaises(AuthError):
            event_auth.check_state_dependent_auth_rules(_alias_event(RoomVersions.V1, creator, state_key=''), auth_events)
        with self.assertRaises(AuthError):
            event_auth.check_state_dependent_auth_rules(_alias_event(RoomVersions.V1, creator, state_key='test.com'), auth_events)
        event_auth.check_state_dependent_auth_rules(_alias_event(RoomVersions.V1, other), auth_events)

    def test_msc2432_alias_event(self) -> None:
        if False:
            while True:
                i = 10
        'After MSC2432, alias events have no special behavior.'
        creator = '@creator:example.com'
        other = '@other:example.com'
        auth_events = [_create_event(RoomVersions.V6, creator), _join_event(RoomVersions.V6, creator)]
        event_auth.check_state_dependent_auth_rules(_alias_event(RoomVersions.V6, creator), auth_events)
        event_auth.check_state_dependent_auth_rules(_alias_event(RoomVersions.V6, creator, state_key=''), auth_events)
        event_auth.check_state_dependent_auth_rules(_alias_event(RoomVersions.V6, creator, state_key='test.com'), auth_events)
        with self.assertRaises(AuthError):
            event_auth.check_state_dependent_auth_rules(_alias_event(RoomVersions.V6, other), auth_events)

    @parameterized.expand([(RoomVersions.V1, True), (RoomVersions.V6, False)])
    def test_notifications(self, room_version: RoomVersion, allow_modification: bool) -> None:
        if False:
            return 10
        '\n        Notifications power levels get checked due to MSC2209.\n        '
        creator = '@creator:example.com'
        pleb = '@joiner:example.com'
        auth_events = [_create_event(room_version, creator), _join_event(room_version, creator), _power_levels_event(room_version, creator, {'state_default': '30', 'users': {pleb: '30'}}), _join_event(room_version, pleb)]
        pl_event = _power_levels_event(room_version, pleb, {'notifications': {'room': 100}})
        if allow_modification:
            event_auth.check_state_dependent_auth_rules(pl_event, auth_events)
        else:
            with self.assertRaises(AuthError):
                event_auth.check_state_dependent_auth_rules(pl_event, auth_events)

    def test_join_rules_public(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test joining a public room.\n        '
        creator = '@creator:example.com'
        pleb = '@joiner:example.com'
        auth_events = {('m.room.create', ''): _create_event(RoomVersions.V6, creator), ('m.room.member', creator): _join_event(RoomVersions.V6, creator), ('m.room.join_rules', ''): _join_rules_event(RoomVersions.V6, creator, 'public')}
        event_auth.check_state_dependent_auth_rules(_join_event(RoomVersions.V6, pleb), auth_events.values())
        with self.assertRaises(AuthError):
            event_auth.check_state_dependent_auth_rules(_member_event(RoomVersions.V6, pleb, 'join', sender=creator), auth_events.values())
        auth_events['m.room.member', pleb] = _member_event(RoomVersions.V6, pleb, 'ban')
        with self.assertRaises(AuthError):
            event_auth.check_state_dependent_auth_rules(_join_event(RoomVersions.V6, pleb), auth_events.values())
        auth_events['m.room.member', pleb] = _member_event(RoomVersions.V6, pleb, 'leave')
        event_auth.check_state_dependent_auth_rules(_join_event(RoomVersions.V6, pleb), auth_events.values())
        auth_events['m.room.member', pleb] = _member_event(RoomVersions.V6, pleb, 'join')
        event_auth.check_state_dependent_auth_rules(_join_event(RoomVersions.V6, pleb), auth_events.values())
        auth_events['m.room.member', pleb] = _member_event(RoomVersions.V6, pleb, 'invite', sender=creator)
        event_auth.check_state_dependent_auth_rules(_join_event(RoomVersions.V6, pleb), auth_events.values())

    def test_join_rules_invite(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test joining an invite only room.\n        '
        creator = '@creator:example.com'
        pleb = '@joiner:example.com'
        auth_events = {('m.room.create', ''): _create_event(RoomVersions.V6, creator), ('m.room.member', creator): _join_event(RoomVersions.V6, creator), ('m.room.join_rules', ''): _join_rules_event(RoomVersions.V6, creator, 'invite')}
        with self.assertRaises(AuthError):
            event_auth.check_state_dependent_auth_rules(_join_event(RoomVersions.V6, pleb), auth_events.values())
        with self.assertRaises(AuthError):
            event_auth.check_state_dependent_auth_rules(_member_event(RoomVersions.V6, pleb, 'join', sender=creator), auth_events.values())
        auth_events['m.room.member', pleb] = _member_event(RoomVersions.V6, pleb, 'ban')
        with self.assertRaises(AuthError):
            event_auth.check_state_dependent_auth_rules(_join_event(RoomVersions.V6, pleb), auth_events.values())
        auth_events['m.room.member', pleb] = _member_event(RoomVersions.V6, pleb, 'leave')
        with self.assertRaises(AuthError):
            event_auth.check_state_dependent_auth_rules(_join_event(RoomVersions.V6, pleb), auth_events.values())
        auth_events['m.room.member', pleb] = _member_event(RoomVersions.V6, pleb, 'join')
        event_auth.check_state_dependent_auth_rules(_join_event(RoomVersions.V6, pleb), auth_events.values())
        auth_events['m.room.member', pleb] = _member_event(RoomVersions.V6, pleb, 'invite', sender=creator)
        event_auth.check_state_dependent_auth_rules(_join_event(RoomVersions.V6, pleb), auth_events.values())

    def test_join_rules_restricted_old_room(self) -> None:
        if False:
            return 10
        'Old room versions should reject joins to restricted rooms'
        creator = '@creator:example.com'
        pleb = '@joiner:example.com'
        auth_events = {('m.room.create', ''): _create_event(RoomVersions.V6, creator), ('m.room.member', creator): _join_event(RoomVersions.V6, creator), ('m.room.power_levels', ''): _power_levels_event(RoomVersions.V6, creator, {'invite': 0}), ('m.room.join_rules', ''): _join_rules_event(RoomVersions.V6, creator, 'restricted')}
        with self.assertRaises(AuthError):
            event_auth.check_state_dependent_auth_rules(_join_event(RoomVersions.V6, pleb), auth_events.values())

    def test_join_rules_msc3083_restricted(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test joining a restricted room from MSC3083.\n\n        This is similar to the public test, but has some additional checks on\n        signatures.\n\n        The checks which care about signatures fake them by simply adding an\n        object of the proper form, not generating valid signatures.\n        '
        creator = '@creator:example.com'
        pleb = '@joiner:example.com'
        auth_events = {('m.room.create', ''): _create_event(RoomVersions.V8, creator), ('m.room.member', creator): _join_event(RoomVersions.V8, creator), ('m.room.power_levels', ''): _power_levels_event(RoomVersions.V8, creator, {'invite': 0}), ('m.room.join_rules', ''): _join_rules_event(RoomVersions.V8, creator, 'restricted')}
        authorised_join_event = _join_event(RoomVersions.V8, pleb, additional_content={EventContentFields.AUTHORISING_USER: '@creator:example.com'})
        event_auth.check_state_dependent_auth_rules(authorised_join_event, auth_events.values())
        pl_auth_events = auth_events.copy()
        pl_auth_events['m.room.power_levels', ''] = _power_levels_event(RoomVersions.V8, creator, {'invite': 100, 'users': {'@inviter:foo.test': 150}})
        pl_auth_events['m.room.member', '@inviter:foo.test'] = _join_event(RoomVersions.V8, '@inviter:foo.test')
        event_auth.check_state_dependent_auth_rules(_join_event(RoomVersions.V8, pleb, additional_content={EventContentFields.AUTHORISING_USER: '@inviter:foo.test'}), pl_auth_events.values())
        with self.assertRaises(AuthError):
            event_auth.check_state_dependent_auth_rules(_join_event(RoomVersions.V8, pleb), auth_events.values())
        pl_auth_events = auth_events.copy()
        pl_auth_events['m.room.power_levels', ''] = _power_levels_event(RoomVersions.V8, creator, {'invite': 100, 'users': {'@other:example.com': 150}})
        with self.assertRaises(AuthError):
            event_auth.check_state_dependent_auth_rules(_join_event(RoomVersions.V8, pleb, additional_content={EventContentFields.AUTHORISING_USER: '@other:example.com'}), auth_events.values())
        with self.assertRaises(AuthError):
            event_auth.check_state_dependent_auth_rules(_member_event(RoomVersions.V8, pleb, 'join', sender=creator, additional_content={EventContentFields.AUTHORISING_USER: '@inviter:foo.test'}), auth_events.values())
        auth_events['m.room.member', pleb] = _member_event(RoomVersions.V8, pleb, 'ban')
        with self.assertRaises(AuthError):
            event_auth.check_state_dependent_auth_rules(authorised_join_event, auth_events.values())
        auth_events['m.room.member', pleb] = _member_event(RoomVersions.V8, pleb, 'leave')
        event_auth.check_state_dependent_auth_rules(authorised_join_event, auth_events.values())
        auth_events['m.room.member', pleb] = _member_event(RoomVersions.V8, pleb, 'join')
        event_auth.check_state_dependent_auth_rules(_join_event(RoomVersions.V8, pleb), auth_events.values())
        auth_events['m.room.member', pleb] = _member_event(RoomVersions.V8, pleb, 'invite', sender=creator)
        event_auth.check_state_dependent_auth_rules(_join_event(RoomVersions.V8, pleb), auth_events.values())

    def test_room_v10_rejects_string_power_levels(self) -> None:
        if False:
            i = 10
            return i + 15
        pl_event_content = {'users_default': '42'}
        pl_event = make_event_from_dict({'room_id': TEST_ROOM_ID, **_maybe_get_event_id_dict_for_room_version(RoomVersions.V10), 'type': 'm.room.power_levels', 'sender': '@test:test.com', 'state_key': '', 'content': pl_event_content, 'signatures': {'test.com': {'ed25519:0': 'some9signature'}}}, room_version=RoomVersions.V10)
        pl_event2_content = {'events': {'m.room.name': '42', 'm.room.power_levels': 42}}
        pl_event2 = make_event_from_dict({'room_id': TEST_ROOM_ID, **_maybe_get_event_id_dict_for_room_version(RoomVersions.V10), 'type': 'm.room.power_levels', 'sender': '@test:test.com', 'state_key': '', 'content': pl_event2_content, 'signatures': {'test.com': {'ed25519:0': 'some9signature'}}}, room_version=RoomVersions.V10)
        with self.assertRaises(SynapseError):
            event_auth._check_power_levels(pl_event.room_version, pl_event, {('fake_type', 'fake_key'): pl_event2})
        with self.assertRaises(SynapseError):
            event_auth._check_power_levels(pl_event.room_version, pl_event2, {('fake_type', 'fake_key'): pl_event})

    def test_room_v10_rejects_other_non_integer_power_levels(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'We should reject PLs that are non-integer, non-string JSON values.\n\n        test_room_v10_rejects_string_power_levels above handles the string case.\n        '

        def create_event(pl_event_content: Dict[str, Any]) -> EventBase:
            if False:
                i = 10
                return i + 15
            return make_event_from_dict({'room_id': TEST_ROOM_ID, **_maybe_get_event_id_dict_for_room_version(RoomVersions.V10), 'type': 'm.room.power_levels', 'sender': '@test:test.com', 'state_key': '', 'content': pl_event_content, 'signatures': {'test.com': {'ed25519:0': 'some9signature'}}}, room_version=RoomVersions.V10)
        contents: Iterable[Dict[str, Any]] = [{'notifications': {'room': None}}, {'users': {'@alice:wonderland': []}}, {'users_default': {}}]
        for content in contents:
            event = create_event(content)
            with self.assertRaises(SynapseError):
                event_auth._check_power_levels(event.room_version, event, {})
TEST_DOMAIN = 'example.com'
TEST_ROOM_ID = f'!test_room:{TEST_DOMAIN}'

def _create_event(room_version: RoomVersion, user_id: str) -> EventBase:
    if False:
        while True:
            i = 10
    return make_event_from_dict({'room_id': TEST_ROOM_ID, **_maybe_get_event_id_dict_for_room_version(room_version), 'type': 'm.room.create', 'state_key': '', 'sender': user_id, 'content': {'creator': user_id}, 'auth_events': []}, room_version=room_version)

def _member_event(room_version: RoomVersion, user_id: str, membership: str, sender: Optional[str]=None, additional_content: Optional[dict]=None) -> EventBase:
    if False:
        return 10
    return make_event_from_dict({'room_id': TEST_ROOM_ID, **_maybe_get_event_id_dict_for_room_version(room_version), 'type': 'm.room.member', 'sender': sender or user_id, 'state_key': user_id, 'content': {'membership': membership, **(additional_content or {})}, 'auth_events': [], 'prev_events': []}, room_version=room_version)

def _join_event(room_version: RoomVersion, user_id: str, additional_content: Optional[dict]=None) -> EventBase:
    if False:
        print('Hello World!')
    return _member_event(room_version, user_id, 'join', additional_content=additional_content)

def _power_levels_event(room_version: RoomVersion, sender: str, content: JsonDict) -> EventBase:
    if False:
        return 10
    return make_event_from_dict({'room_id': TEST_ROOM_ID, **_maybe_get_event_id_dict_for_room_version(room_version), 'type': 'm.room.power_levels', 'sender': sender, 'state_key': '', 'content': content}, room_version=room_version)

def _alias_event(room_version: RoomVersion, sender: str, **kwargs: Any) -> EventBase:
    if False:
        i = 10
        return i + 15
    data = {'room_id': TEST_ROOM_ID, **_maybe_get_event_id_dict_for_room_version(room_version), 'type': 'm.room.aliases', 'sender': sender, 'state_key': get_domain_from_id(sender), 'content': {'aliases': []}}
    data.update(**kwargs)
    return make_event_from_dict(data, room_version=room_version)

def _build_auth_dict_for_room_version(room_version: RoomVersion, auth_events: Iterable[EventBase]) -> List:
    if False:
        return 10
    if room_version.event_format == EventFormatVersions.ROOM_V1_V2:
        return [(e.event_id, 'not_used') for e in auth_events]
    else:
        return [e.event_id for e in auth_events]

def _random_state_event(room_version: RoomVersion, sender: str, auth_events: Optional[Iterable[EventBase]]=None) -> EventBase:
    if False:
        print('Hello World!')
    if auth_events is None:
        auth_events = []
    return make_event_from_dict({'room_id': TEST_ROOM_ID, **_maybe_get_event_id_dict_for_room_version(room_version), 'type': 'test.state', 'sender': sender, 'state_key': '', 'content': {'membership': 'join'}, 'auth_events': _build_auth_dict_for_room_version(room_version, auth_events)}, room_version=room_version)

def _join_rules_event(room_version: RoomVersion, sender: str, join_rule: str) -> EventBase:
    if False:
        return 10
    return make_event_from_dict({'room_id': TEST_ROOM_ID, **_maybe_get_event_id_dict_for_room_version(room_version), 'type': 'm.room.join_rules', 'sender': sender, 'state_key': '', 'content': {'join_rule': join_rule}}, room_version=room_version)
event_count = 0

def _maybe_get_event_id_dict_for_room_version(room_version: RoomVersion) -> dict:
    if False:
        return 10
    'If this room version needs it, generate an event id'
    if room_version.event_format != EventFormatVersions.ROOM_V1_V2:
        return {}
    global event_count
    c = event_count
    event_count += 1
    return {'event_id': '!%i:example.com' % (c,)}