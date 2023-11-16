import itertools
import logging
import re
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import attr
from synapse.api.constants import EventTypes, HistoryVisibility, JoinRules, Membership, RoomTypes
from synapse.api.errors import Codes, NotFoundError, StoreError, SynapseError, UnstableSpecAuthError, UnsupportedRoomVersionError
from synapse.api.ratelimiting import Ratelimiter
from synapse.config.ratelimiting import RatelimitSettings
from synapse.events import EventBase
from synapse.types import JsonDict, Requester, StrCollection
from synapse.util.caches.response_cache import ResponseCache
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)
MAX_ROOMS = 50
MAX_ROOMS_PER_SPACE = 50
MAX_SERVERS_PER_SPACE = 3

@attr.s(slots=True, frozen=True, auto_attribs=True)
class _PaginationKey:
    """The key used to find unique pagination session."""
    room_id: str
    suggested_only: bool
    max_depth: Optional[int]
    token: str

@attr.s(slots=True, frozen=True, auto_attribs=True)
class _PaginationSession:
    """The information that is stored for pagination."""
    creation_time_ms: int
    room_queue: List['_RoomQueueEntry']
    processed_rooms: Set[str]

class RoomSummaryHandler:
    _PAGINATION_SESSION_TYPE = 'room_hierarchy_pagination'
    _PAGINATION_SESSION_VALIDITY_PERIOD_MS = 5 * 60 * 1000

    def __init__(self, hs: 'HomeServer'):
        if False:
            print('Hello World!')
        self._event_auth_handler = hs.get_event_auth_handler()
        self._store = hs.get_datastores().main
        self._storage_controllers = hs.get_storage_controllers()
        self._event_serializer = hs.get_event_client_serializer()
        self._server_name = hs.hostname
        self._federation_client = hs.get_federation_client()
        self._ratelimiter = Ratelimiter(store=self._store, clock=hs.get_clock(), cfg=RatelimitSettings('<room summary>', per_second=5, burst_count=10))
        self._pagination_response_cache: ResponseCache[Tuple[str, str, bool, Optional[int], Optional[int], Optional[str]]] = ResponseCache(hs.get_clock(), 'get_room_hierarchy')
        self._msc3266_enabled = hs.config.experimental.msc3266_enabled

    async def get_room_hierarchy(self, requester: Requester, requested_room_id: str, suggested_only: bool=False, max_depth: Optional[int]=None, limit: Optional[int]=None, from_token: Optional[str]=None) -> JsonDict:
        """
        Implementation of the room hierarchy C-S API.

        Args:
            requester: The user ID of the user making this request.
            requested_room_id: The room ID to start the hierarchy at (the "root" room).
            suggested_only: Whether we should only return children with the "suggested"
                flag set.
            max_depth: The maximum depth in the tree to explore, must be a
                non-negative integer.

                0 would correspond to just the root room, 1 would include just
                the root room's children, etc.
            limit: An optional limit on the number of rooms to return per
                page. Must be a positive integer.
            from_token: An optional pagination token.

        Returns:
            The JSON hierarchy dictionary.
        """
        await self._ratelimiter.ratelimit(requester)
        return await self._pagination_response_cache.wrap((requester.user.to_string(), requested_room_id, suggested_only, max_depth, limit, from_token), self._get_room_hierarchy, requester.user.to_string(), requested_room_id, suggested_only, max_depth, limit, from_token)

    async def _get_room_hierarchy(self, requester: str, requested_room_id: str, suggested_only: bool=False, max_depth: Optional[int]=None, limit: Optional[int]=None, from_token: Optional[str]=None) -> JsonDict:
        """See docstring for SpaceSummaryHandler.get_room_hierarchy."""
        if not await self._is_local_room_accessible(requested_room_id, requester):
            raise UnstableSpecAuthError(403, 'User %s not in room %s, and room previews are disabled' % (requester, requested_room_id), errcode=Codes.NOT_JOINED)
        if from_token:
            try:
                pagination_session = await self._store.get_session(session_type=self._PAGINATION_SESSION_TYPE, session_id=from_token)
            except StoreError:
                raise SynapseError(400, 'Unknown pagination token', Codes.INVALID_PARAM)
            if requester != pagination_session['requester'] or requested_room_id != pagination_session['room_id'] or suggested_only != pagination_session['suggested_only'] or (max_depth != pagination_session['max_depth']):
                raise SynapseError(400, 'Unknown pagination token', Codes.INVALID_PARAM)
            room_queue = [_RoomQueueEntry(*fields) for fields in pagination_session['room_queue']]
            processed_rooms = set(pagination_session['processed_rooms'])
        else:
            room_queue = [_RoomQueueEntry(requested_room_id, ())]
            processed_rooms = set()
        rooms_result: List[JsonDict] = []
        if limit is None:
            limit = MAX_ROOMS
        else:
            limit = min(limit, MAX_ROOMS)
        while room_queue and len(rooms_result) < limit:
            queue_entry = room_queue.pop()
            room_id = queue_entry.room_id
            current_depth = queue_entry.depth
            if room_id in processed_rooms:
                continue
            logger.debug('Processing room %s', room_id)
            children_room_entries: Dict[str, JsonDict] = {}
            inaccessible_children: Set[str] = set()
            is_in_room = await self._store.is_host_joined(room_id, self._server_name)
            if is_in_room:
                room_entry = await self._summarize_local_room(requester, None, room_id, suggested_only)
            else:
                if queue_entry.remote_room and (queue_entry.remote_room.get('room_type') != RoomTypes.SPACE or (max_depth is not None and current_depth >= max_depth)):
                    room_entry = _RoomEntry(queue_entry.room_id, queue_entry.remote_room)
                else:
                    (room_entry, children_room_entries, inaccessible_children) = await self._summarize_remote_room_hierarchy(queue_entry, suggested_only)
                if room_entry and (not await self._is_remote_room_accessible(requester, queue_entry.room_id, room_entry.room)):
                    room_entry = None
            processed_rooms.add(room_id)
            if room_entry:
                rooms_result.append(room_entry.as_json(for_client=True))
                if max_depth is None or current_depth < max_depth:
                    room_queue.extend((_RoomQueueEntry(ev['state_key'], ev['content']['via'], current_depth + 1, children_room_entries.get(ev['state_key'])) for ev in reversed(room_entry.children_state_events) if ev['type'] == EventTypes.SpaceChild and ev['state_key'] not in inaccessible_children))
        result: JsonDict = {'rooms': rooms_result}
        if room_queue:
            result['next_batch'] = await self._store.create_session(session_type=self._PAGINATION_SESSION_TYPE, value={'requester': requester, 'room_id': requested_room_id, 'suggested_only': suggested_only, 'max_depth': max_depth, 'room_queue': [attr.astuple(room_entry) for room_entry in room_queue], 'processed_rooms': list(processed_rooms)}, expiry_ms=self._PAGINATION_SESSION_VALIDITY_PERIOD_MS)
        return result

    async def get_federation_hierarchy(self, origin: str, requested_room_id: str, suggested_only: bool) -> JsonDict:
        """
        Implementation of the room hierarchy Federation API.

        This is similar to get_room_hierarchy, but does not recurse into the space.
        It also considers whether anyone on the server may be able to access the
        room, as opposed to whether a specific user can.

        Args:
            origin: The server requesting the spaces summary.
            requested_room_id: The room ID to start the hierarchy at (the "root" room).
            suggested_only: whether we should only return children with the "suggested"
                flag set.

        Returns:
            The JSON hierarchy dictionary.
        """
        root_room_entry = await self._summarize_local_room(None, origin, requested_room_id, suggested_only)
        if root_room_entry is None:
            raise SynapseError(404, 'Unknown room: %s' % (requested_room_id,))
        children_rooms_result: List[JsonDict] = []
        inaccessible_children: List[str] = []
        for child_room in itertools.islice(root_room_entry.children_state_events, MAX_ROOMS_PER_SPACE):
            room_id = child_room.get('state_key')
            assert isinstance(room_id, str)
            if not await self._store.is_host_joined(room_id, self._server_name):
                continue
            room_entry = await self._summarize_local_room(None, origin, room_id, suggested_only, include_children=False)
            if room_entry:
                children_rooms_result.append(room_entry.room)
            else:
                inaccessible_children.append(room_id)
        return {'room': root_room_entry.as_json(), 'children': children_rooms_result, 'inaccessible_children': inaccessible_children}

    async def _summarize_local_room(self, requester: Optional[str], origin: Optional[str], room_id: str, suggested_only: bool, include_children: bool=True) -> Optional['_RoomEntry']:
        """
        Generate a room entry and a list of event entries for a given room.

        Args:
            requester:
                The user requesting the summary, if it is a local request. None
                if this is a federation request.
            origin:
                The server requesting the summary, if it is a federation request.
                None if this is a local request.
            room_id: The room ID to summarize.
            suggested_only: True if only suggested children should be returned.
                Otherwise, all children are returned.
            include_children:
                Whether to include the events of any children.

        Returns:
            A room entry if the room should be returned. None, otherwise.
        """
        if not await self._is_local_room_accessible(room_id, requester, origin):
            return None
        room_entry = await self._build_room_entry(room_id, for_federation=bool(origin))
        if room_entry.get('room_type') != RoomTypes.SPACE or not include_children:
            return _RoomEntry(room_id, room_entry)
        child_events = await self._get_child_events(room_id)
        if suggested_only:
            child_events = filter(_is_suggested_child_event, child_events)
        stripped_events: List[JsonDict] = [{'type': e.type, 'state_key': e.state_key, 'content': e.content, 'sender': e.sender, 'origin_server_ts': e.origin_server_ts} for e in child_events]
        return _RoomEntry(room_id, room_entry, stripped_events)

    async def _summarize_remote_room_hierarchy(self, room: '_RoomQueueEntry', suggested_only: bool) -> Tuple[Optional['_RoomEntry'], Dict[str, JsonDict], Set[str]]:
        """
        Request room entries and a list of event entries for a given room by querying a remote server.

        Args:
            room: The room to summarize.
            suggested_only: True if only suggested children should be returned.
                Otherwise, all children are returned.

        Returns:
            A tuple of:
                The room entry.
                Partial room data return over federation.
                A set of inaccessible children room IDs.
        """
        room_id = room.room_id
        logger.info('Requesting summary for %s via %s', room_id, room.via)
        via = itertools.islice(room.via, MAX_SERVERS_PER_SPACE)
        try:
            (room_response, children_state_events, children, inaccessible_children) = await self._federation_client.get_room_hierarchy(via, room_id, suggested_only=suggested_only)
        except Exception as e:
            logger.warning('Unable to get hierarchy of %s via federation: %s', room_id, e, exc_info=logger.isEnabledFor(logging.DEBUG))
            return (None, {}, set())
        children_by_room_id = {c['room_id']: c for c in children if 'room_id' in c and isinstance(c['room_id'], str)}
        return (_RoomEntry(room_id, room_response, children_state_events), children_by_room_id, set(inaccessible_children))

    async def _is_local_room_accessible(self, room_id: str, requester: Optional[str], origin: Optional[str]=None) -> bool:
        """
        Calculate whether the room should be shown to the requester.

        It should return true if:

        * The requesting user is joined or can join the room (per MSC3173); or
        * The origin server has any user that is joined or can join the room; or
        * The history visibility is set to world readable.

        Args:
            room_id: The room ID to check accessibility of.
            requester:
                The user making the request, if it is a local request.
                None if this is a federation request.
            origin:
                The server making the request, if it is a federation request.
                None if this is a local request.

        Returns:
             True if the room is accessible to the requesting user or server.
        """
        state_ids = await self._storage_controllers.state.get_current_state_ids(room_id)
        if not state_ids:
            if requester and await self._store.get_invite_for_local_user_in_room(requester, room_id):
                return True
            logger.info('room %s is unknown, omitting from summary', room_id)
            return False
        try:
            room_version = await self._store.get_room_version(room_id)
        except UnsupportedRoomVersionError:
            return False
        join_rules_event_id = state_ids.get((EventTypes.JoinRules, ''))
        if join_rules_event_id:
            join_rules_event = await self._store.get_event(join_rules_event_id)
            join_rule = join_rules_event.content.get('join_rule')
            if join_rule == JoinRules.PUBLIC or (room_version.knock_join_rule and join_rule == JoinRules.KNOCK) or (room_version.knock_restricted_join_rule and join_rule == JoinRules.KNOCK_RESTRICTED):
                return True
        hist_vis_event_id = state_ids.get((EventTypes.RoomHistoryVisibility, ''))
        if hist_vis_event_id:
            hist_vis_ev = await self._store.get_event(hist_vis_event_id)
            hist_vis = hist_vis_ev.content.get('history_visibility')
            if hist_vis == HistoryVisibility.WORLD_READABLE:
                return True
        if requester:
            member_event_id = state_ids.get((EventTypes.Member, requester), None)
            if member_event_id:
                member_event = await self._store.get_event(member_event_id)
                if member_event.membership in (Membership.JOIN, Membership.INVITE):
                    return True
            if await self._event_auth_handler.has_restricted_join_rules(state_ids, room_version):
                allowed_rooms = await self._event_auth_handler.get_rooms_that_allow_join(state_ids)
                if await self._event_auth_handler.is_user_in_rooms(allowed_rooms, requester):
                    return True
        elif origin:
            if await self._event_auth_handler.is_host_in_room(room_id, origin) or await self._store.is_host_invited(room_id, origin):
                return True
            if await self._event_auth_handler.has_restricted_join_rules(state_ids, room_version):
                allowed_rooms = await self._event_auth_handler.get_rooms_that_allow_join(state_ids)
                for space_id in allowed_rooms:
                    if await self._event_auth_handler.is_host_in_room(space_id, origin):
                        return True
        logger.info('room %s is unpeekable and requester %s is not a member / not allowed to join, omitting from summary', room_id, requester or origin)
        return False

    async def _is_remote_room_accessible(self, requester: Optional[str], room_id: str, room: JsonDict) -> bool:
        """
        Calculate whether the room received over federation should be shown to the requester.

        It should return true if:

        * The requester is joined or can join the room (per MSC3173).
        * The history visibility is set to world readable.

        Note that the local server is not in the requested room (which is why the
        remote call was made in the first place), but the user could have access
        due to an invite, etc.

        Args:
            requester: The user requesting the summary. If not passed only world
                readability is checked.
            room_id: The room ID returned over federation.
            room: The summary of the room returned over federation.

        Returns:
            True if the room is accessible to the requesting user.
        """
        if room.get('join_rule') in (JoinRules.PUBLIC, JoinRules.KNOCK, JoinRules.KNOCK_RESTRICTED) or room.get('world_readable') is True:
            return True
        elif not requester:
            return False
        allowed_rooms = room.get('allowed_room_ids')
        if allowed_rooms and isinstance(allowed_rooms, list):
            if await self._event_auth_handler.is_user_in_rooms(allowed_rooms, requester):
                return True
        return await self._is_local_room_accessible(room_id, requester)

    async def _build_room_entry(self, room_id: str, for_federation: bool) -> JsonDict:
        """
        Generate en entry summarising a single room.

        Args:
            room_id: The room ID to summarize.
            for_federation: True if this is a summary requested over federation
                (which includes additional fields).

        Returns:
            The JSON dictionary for the room.
        """
        stats = await self._store.get_room_with_stats(room_id)
        assert stats is not None, 'unable to retrieve stats for %s' % (room_id,)
        entry: JsonDict = {'room_id': stats.room_id, 'name': stats.name, 'topic': stats.topic, 'canonical_alias': stats.canonical_alias, 'num_joined_members': stats.joined_members, 'avatar_url': stats.avatar, 'join_rule': stats.join_rules, 'world_readable': stats.history_visibility == HistoryVisibility.WORLD_READABLE, 'guest_can_join': stats.guest_access == 'can_join', 'room_type': stats.room_type}
        if self._msc3266_enabled:
            entry['im.nheko.summary.version'] = stats.version
            entry['im.nheko.summary.encryption'] = stats.encryption
        if for_federation:
            current_state_ids = await self._storage_controllers.state.get_current_state_ids(room_id)
            room_version = await self._store.get_room_version(room_id)
            if await self._event_auth_handler.has_restricted_join_rules(current_state_ids, room_version):
                allowed_rooms = await self._event_auth_handler.get_rooms_that_allow_join(current_state_ids)
                if allowed_rooms:
                    entry['allowed_room_ids'] = allowed_rooms
        room_entry = {k: v for (k, v) in entry.items() if v is not None}
        return room_entry

    async def _get_child_events(self, room_id: str) -> Iterable[EventBase]:
        """
        Get the child events for a given room.

        The returned results are sorted for stability.

        Args:
            room_id: The room id to get the children of.

        Returns:
            An iterable of sorted child events.
        """
        current_state_ids = await self._storage_controllers.state.get_current_state_ids(room_id)
        events = await self._store.get_events_as_list([event_id for (key, event_id) in current_state_ids.items() if key[0] == EventTypes.SpaceChild])
        return sorted(filter(_has_valid_via, events), key=_child_events_comparison_key)

    async def get_room_summary(self, requester: Optional[str], room_id: str, remote_room_hosts: Optional[List[str]]=None) -> JsonDict:
        """
        Implementation of the room summary C-S API from MSC3266

        Args:
            requester:  user id of the user making this request, will be None
                for unauthenticated requests

            room_id: room id to summarise.

            remote_room_hosts: a list of homeservers to try fetching data through
                if we don't know it ourselves

        Returns:
            summary dict to return
        """
        is_in_room = await self._store.is_host_joined(room_id, self._server_name)
        if is_in_room:
            room_entry = await self._summarize_local_room(requester, None, room_id, suggested_only=False, include_children=False)
            if not room_entry:
                raise NotFoundError('Room not found or is not accessible')
            room_summary = room_entry.room
            if requester:
                (membership, _) = await self._store.get_local_current_membership_for_user_in_room(requester, room_id)
                room_summary['membership'] = membership or 'leave'
        else:
            if remote_room_hosts is None:
                raise SynapseError(400, 'Missing via to query remote room')
            (room_entry, children_room_entries, inaccessible_children) = await self._summarize_remote_room_hierarchy(_RoomQueueEntry(room_id, remote_room_hosts), suggested_only=True)
            if not room_entry or not await self._is_remote_room_accessible(requester, room_entry.room_id, room_entry.room):
                raise NotFoundError('Room not found or is not accessible')
            room = dict(room_entry.room)
            room.pop('allowed_room_ids', None)
            if requester:
                (membership, _) = await self._store.get_local_current_membership_for_user_in_room(requester, room_id)
                room['membership'] = membership or 'leave'
            return room
        return room_summary

@attr.s(frozen=True, slots=True, auto_attribs=True)
class _RoomQueueEntry:
    room_id: str
    via: StrCollection
    depth: int = 0
    remote_room: Optional[JsonDict] = None

@attr.s(frozen=True, slots=True, auto_attribs=True)
class _RoomEntry:
    room_id: str
    room: JsonDict
    children_state_events: Sequence[JsonDict] = ()

    def as_json(self, for_client: bool=False) -> JsonDict:
        if False:
            while True:
                i = 10
        '\n        Returns a JSON dictionary suitable for the room hierarchy endpoint.\n\n        It returns the room summary including the stripped m.space.child events\n        as a sub-key.\n\n        Args:\n            for_client: If true, any server-server only fields are stripped from\n                the result.\n\n        '
        result = dict(self.room)
        if for_client:
            result.pop('allowed_room_ids', False)
        result['children_state'] = self.children_state_events
        return result

def _has_valid_via(e: EventBase) -> bool:
    if False:
        for i in range(10):
            print('nop')
    via = e.content.get('via')
    if not via or not isinstance(via, list):
        return False
    for v in via:
        if not isinstance(v, str):
            logger.debug('Ignoring edge event %s with invalid via entry', e.event_id)
            return False
    return True

def _is_suggested_child_event(edge_event: EventBase) -> bool:
    if False:
        while True:
            i = 10
    suggested = edge_event.content.get('suggested')
    if isinstance(suggested, bool) and suggested:
        return True
    logger.debug('Ignorning not-suggested child %s', edge_event.state_key)
    return False
_INVALID_ORDER_CHARS_RE = re.compile('[^\\x20-\\x7E]')

def _child_events_comparison_key(child: EventBase) -> Tuple[bool, Optional[str], int, str]:
    if False:
        print('Hello World!')
    "\n    Generate a value for comparing two child events for ordering.\n\n    The rules for ordering are:\n\n    1. The 'order' key, if it is valid.\n    2. The 'origin_server_ts' of the 'm.space.child' event.\n    3. The 'room_id'.\n\n    Args:\n        child: The event for generating a comparison key.\n\n    Returns:\n        The comparison key as a tuple of:\n            False if the ordering is valid.\n            The 'order' field or None if it is not given or invalid.\n            The 'origin_server_ts' field.\n            The room ID.\n    "
    order = child.content.get('order')
    if not isinstance(order, str):
        order = None
    elif len(order) > 50 or _INVALID_ORDER_CHARS_RE.search(order):
        order = None
    return (order is None, order, child.origin_server_ts, child.room_id)