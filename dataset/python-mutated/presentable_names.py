import logging
import re
from typing import TYPE_CHECKING, Dict, Iterable, Optional
from synapse.api.constants import EventTypes, Membership
from synapse.events import EventBase
from synapse.types import StateMap
if TYPE_CHECKING:
    from synapse.storage.databases.main import DataStore
logger = logging.getLogger(__name__)
ALIAS_RE = re.compile('^#.*:.+$')
ALL_ALONE = 'Empty Room'

async def calculate_room_name(store: 'DataStore', room_state_ids: StateMap[str], user_id: str, fallback_to_members: bool=True, fallback_to_single_member: bool=True) -> Optional[str]:
    """
    Works out a user-facing name for the given room as per Matrix
    spec recommendations.
    Does not yet support internationalisation.
    Args:
        store: The data store to query.
        room_state_ids: Dictionary of the room's state IDs.
        user_id: The ID of the user to whom the room name is being presented
        fallback_to_members: If False, return None instead of generating a name
                             based on the room's members if the room has no
                             title or aliases.
        fallback_to_single_member: If False, return None instead of generating a
            name based on the user who invited this user to the room if the room
            has no title or aliases.

    Returns:
        A human readable name for the room, if possible.
    """
    if (EventTypes.Name, '') in room_state_ids:
        m_room_name = await store.get_event(room_state_ids[EventTypes.Name, ''], allow_none=True)
        if m_room_name and m_room_name.content and m_room_name.content.get('name'):
            return m_room_name.content['name']
    if (EventTypes.CanonicalAlias, '') in room_state_ids:
        canon_alias = await store.get_event(room_state_ids[EventTypes.CanonicalAlias, ''], allow_none=True)
        if canon_alias and canon_alias.content and canon_alias.content.get('alias') and _looks_like_an_alias(canon_alias.content['alias']):
            return canon_alias.content['alias']
    if not fallback_to_members:
        return None
    my_member_event = None
    if (EventTypes.Member, user_id) in room_state_ids:
        my_member_event = await store.get_event(room_state_ids[EventTypes.Member, user_id], allow_none=True)
    if my_member_event is not None and my_member_event.content.get('membership') == Membership.INVITE:
        if (EventTypes.Member, my_member_event.sender) in room_state_ids:
            inviter_member_event = await store.get_event(room_state_ids[EventTypes.Member, my_member_event.sender], allow_none=True)
            if inviter_member_event:
                if fallback_to_single_member:
                    return 'Invite from %s' % (name_from_member_event(inviter_member_event),)
                else:
                    return None
        else:
            return 'Room Invite'
    room_state_bytype_ids = _state_as_two_level_dict(room_state_ids)
    if EventTypes.Member in room_state_bytype_ids:
        member_events = await store.get_events(list(room_state_bytype_ids[EventTypes.Member].values()))
        all_members = [ev for ev in member_events.values() if ev.content.get('membership') == Membership.JOIN or ev.content.get('membership') == Membership.INVITE]
        all_members.sort(key=lambda e: e.origin_server_ts)
        other_members = [m for m in all_members if m.state_key != user_id]
    else:
        other_members = []
        all_members = []
    if len(other_members) == 0:
        if len(all_members) == 1:
            if all_members[0].sender == user_id:
                if EventTypes.ThirdPartyInvite in room_state_bytype_ids:
                    third_party_invites = room_state_bytype_ids[EventTypes.ThirdPartyInvite].values()
                    if len(third_party_invites) > 0:
                        return 'Inviting email address'
                    else:
                        return ALL_ALONE
            else:
                return name_from_member_event(all_members[0])
        else:
            return ALL_ALONE
    elif len(other_members) == 1 and (not fallback_to_single_member):
        return None
    return descriptor_from_member_events(other_members)

def descriptor_from_member_events(member_events: Iterable[EventBase]) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Get a description of the room based on the member events.\n\n    Args:\n        member_events: The events of a room.\n\n    Returns:\n        The room description\n    '
    member_events = list(member_events)
    if len(member_events) == 0:
        return 'nobody'
    elif len(member_events) == 1:
        return name_from_member_event(member_events[0])
    elif len(member_events) == 2:
        return '%s and %s' % (name_from_member_event(member_events[0]), name_from_member_event(member_events[1]))
    else:
        return '%s and %d others' % (name_from_member_event(member_events[0]), len(member_events) - 1)

def name_from_member_event(member_event: EventBase) -> str:
    if False:
        i = 10
        return i + 15
    if member_event.content and member_event.content.get('displayname'):
        return member_event.content['displayname']
    return member_event.state_key

def _state_as_two_level_dict(state: StateMap[str]) -> Dict[str, Dict[str, str]]:
    if False:
        while True:
            i = 10
    ret: Dict[str, Dict[str, str]] = {}
    for (k, v) in state.items():
        ret.setdefault(k[0], {})[k[1]] = v
    return ret

def _looks_like_an_alias(string: str) -> bool:
    if False:
        while True:
            i = 10
    return ALIAS_RE.match(string) is not None