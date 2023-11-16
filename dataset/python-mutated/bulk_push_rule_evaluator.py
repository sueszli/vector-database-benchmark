import logging
from typing import TYPE_CHECKING, Any, Collection, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast
from prometheus_client import Counter
from twisted.internet.defer import Deferred
from synapse.api.constants import MAIN_TIMELINE, EventContentFields, EventTypes, Membership, RelationTypes
from synapse.api.room_versions import PushRuleRoomFlag
from synapse.event_auth import auth_types_for_event, get_user_power_level
from synapse.events import EventBase, relation_from_event
from synapse.events.snapshot import EventContext
from synapse.logging.context import make_deferred_yieldable, run_in_background
from synapse.state import POWER_KEY
from synapse.storage.databases.main.roommember import EventIdMembership
from synapse.storage.roommember import ProfileInfo
from synapse.synapse_rust.push import FilteredPushRules, PushRuleEvaluator
from synapse.types import JsonValue
from synapse.types.state import StateFilter
from synapse.util import unwrapFirstError
from synapse.util.async_helpers import gather_results
from synapse.util.caches import register_cache
from synapse.util.metrics import measure_func
from synapse.visibility import filter_event_for_clients_with_state
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)
push_rules_invalidation_counter = Counter('synapse_push_bulk_push_rule_evaluator_push_rules_invalidation_counter', '')
push_rules_state_size_counter = Counter('synapse_push_bulk_push_rule_evaluator_push_rules_state_size_counter', '')
STATE_EVENT_TYPES_TO_MARK_UNREAD = {EventTypes.Topic, EventTypes.Name, EventTypes.RoomAvatar, EventTypes.Tombstone}
SENTINEL = object()

def _should_count_as_unread(event: EventBase, context: EventContext) -> bool:
    if False:
        i = 10
        return i + 15
    if context.rejected or event.internal_metadata.is_soft_failed():
        return False
    if not event.is_state() and event.type == EventTypes.Message and (event.content.get('msgtype') == 'm.notice'):
        return False
    relates_to = relation_from_event(event)
    if relates_to and relates_to.rel_type == RelationTypes.REPLACE:
        return False
    body = event.content.get('body')
    if isinstance(body, str) and body:
        return True
    if event.is_state() and event.type in STATE_EVENT_TYPES_TO_MARK_UNREAD:
        return True
    if not event.is_state() and event.type == EventTypes.Encrypted:
        return True
    return False

class BulkPushRuleEvaluator:
    """Calculates the outcome of push rules for an event for all users in the
    room at once.
    """

    def __init__(self, hs: 'HomeServer'):
        if False:
            for i in range(10):
                print('nop')
        self.hs = hs
        self.store = hs.get_datastores().main
        self.clock = hs.get_clock()
        self._event_auth_handler = hs.get_event_auth_handler()
        self.should_calculate_push_rules = self.hs.config.push.enable_push
        self._related_event_match_enabled = self.hs.config.experimental.msc3664_enabled
        self.room_push_rule_cache_metrics = register_cache('cache', 'room_push_rule_cache', cache=[], resizable=False)

    async def _get_rules_for_event(self, event: EventBase) -> Mapping[str, FilteredPushRules]:
        """Get the push rules for all users who may need to be notified about
        the event.

        Note: this does not check if the user is allowed to see the event.

        Returns:
            Mapping of user ID to their push rules.
        """
        if event.type == EventTypes.Member:
            local_users: Sequence[str] = []
            if event.sender != event.state_key and self.hs.is_mine_id(event.state_key):
                target_already_in_room = await self.store.check_local_user_in_room(event.state_key, event.room_id)
                if target_already_in_room:
                    local_users = [event.state_key]
        else:
            local_users = await self.store.get_local_users_in_room(event.room_id)
        local_users = [u for u in local_users if not self.store.get_if_app_services_interested_in_user(u)]
        if event.type == EventTypes.Member and event.membership == Membership.INVITE:
            invited = event.state_key
            if invited and self.hs.is_mine_id(invited) and (invited not in local_users):
                local_users.append(invited)
        if not local_users:
            return {}
        rules_by_user = await self.store.bulk_get_push_rules(local_users)
        logger.debug('Users in room: %s', local_users)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Returning push rules for %r %r', event.room_id, list(rules_by_user.keys()))
        return rules_by_user

    async def _get_power_levels_and_sender_level(self, event: EventBase, context: EventContext, event_id_to_event: Mapping[str, EventBase]) -> Tuple[dict, Optional[int]]:
        """
        Given an event and an event context, get the power level event relevant to the event
        and the power level of the sender of the event.
        Args:
            event: event to check
            context: context of event to check
            event_id_to_event: a mapping of event_id to event for a set of events being
            batch persisted. This is needed as the sought-after power level event may
            be in this batch rather than the DB
        """
        if event.internal_metadata.is_outlier():
            return ({}, None)
        event_types = auth_types_for_event(event.room_version, event)
        prev_state_ids = await context.get_prev_state_ids(StateFilter.from_types(event_types))
        pl_event_id = prev_state_ids.get(POWER_KEY)
        if pl_event_id:
            pl_event = event_id_to_event.get(pl_event_id)
            if pl_event:
                auth_events = {POWER_KEY: pl_event}
            else:
                auth_events = {POWER_KEY: await self.store.get_event(pl_event_id)}
        else:
            auth_events_ids = self._event_auth_handler.compute_auth_events(event, prev_state_ids, for_verification=False)
            auth_events_dict = await self.store.get_events(auth_events_ids)
            for auth_event_id in auth_events_ids:
                auth_event = event_id_to_event.get(auth_event_id)
                if auth_event:
                    auth_events_dict[auth_event_id] = auth_event
            auth_events = {(e.type, e.state_key): e for e in auth_events_dict.values()}
        sender_level = get_user_power_level(event.sender, auth_events)
        pl_event = auth_events.get(POWER_KEY)
        return (pl_event.content if pl_event else {}, sender_level)

    async def _related_events(self, event: EventBase) -> Dict[str, Dict[str, JsonValue]]:
        """Fetches the related events for 'event'. Sets the im.vector.is_falling_back key if the event is from a fallback relation

        Returns:
            Mapping of relation type to flattened events.
        """
        related_events: Dict[str, Dict[str, JsonValue]] = {}
        if self._related_event_match_enabled:
            related_event_id = event.content.get('m.relates_to', {}).get('event_id')
            relation_type = event.content.get('m.relates_to', {}).get('rel_type')
            if related_event_id is not None and relation_type is not None:
                related_event = await self.store.get_event(related_event_id, allow_none=True)
                if related_event is not None:
                    related_events[relation_type] = _flatten_dict(related_event)
            reply_event_id = event.content.get('m.relates_to', {}).get('m.in_reply_to', {}).get('event_id')
            if reply_event_id is not None:
                related_event = await self.store.get_event(reply_event_id, allow_none=True)
                if related_event is not None:
                    related_events['m.in_reply_to'] = _flatten_dict(related_event)
                    if relation_type == 'm.thread' and event.content.get('m.relates_to', {}).get('is_falling_back', False):
                        related_events['m.in_reply_to']['im.vector.is_falling_back'] = ''
        return related_events

    async def action_for_events_by_user(self, events_and_context: List[Tuple[EventBase, EventContext]]) -> None:
        """Given a list of events and their associated contexts, evaluate the push rules
        for each event, check if the message should increment the unread count, and
        insert the results into the event_push_actions_staging table.
        """
        if not self.should_calculate_push_rules:
            return
        event_id_to_event = {e.event_id: e for (e, _) in events_and_context}
        for (event, context) in events_and_context:
            await self._action_for_event_by_user(event, context, event_id_to_event)

    @measure_func('action_for_event_by_user')
    async def _action_for_event_by_user(self, event: EventBase, context: EventContext, event_id_to_event: Mapping[str, EventBase]) -> None:
        if not event.internal_metadata.is_notifiable() or event.room_id in self.hs.config.server.rooms_to_exclude_from_sync:
            return
        count_as_unread = False
        if self.hs.config.experimental.msc2654_enabled:
            count_as_unread = _should_count_as_unread(event, context)
        rules_by_user = await self._get_rules_for_event(event)
        actions_by_user: Dict[str, Collection[Union[Mapping, str]]] = {}
        (room_member_count, (power_levels, sender_power_level), related_events, profiles) = await make_deferred_yieldable(cast('Deferred[Tuple[int, Tuple[dict, Optional[int]], Dict[str, Dict[str, JsonValue]], Mapping[str, ProfileInfo]]]', gather_results((run_in_background(self.store.get_number_joined_users_in_room, event.room_id), run_in_background(self._get_power_levels_and_sender_level, event, context, event_id_to_event), run_in_background(self._related_events, event), run_in_background(self.store.get_subset_users_in_room_with_profiles, event.room_id, rules_by_user.keys())), consumeErrors=True).addErrback(unwrapFirstError)))
        relation = relation_from_event(event)
        thread_id = MAIN_TIMELINE
        if relation:
            if relation.rel_type == RelationTypes.THREAD:
                thread_id = relation.parent_id
            else:
                thread_id = await self.store.get_thread_id(relation.parent_id)
        notification_levels = power_levels.get('notifications', {})
        if not event.room_version.enforce_int_power_levels:
            keys = list(notification_levels.keys())
            for key in keys:
                level = notification_levels.get(key, SENTINEL)
                if level is not SENTINEL and type(level) is not int:
                    try:
                        notification_levels[key] = int(level)
                    except (TypeError, ValueError):
                        del notification_levels[key]
        has_mentions = EventContentFields.MENTIONS in event.content
        evaluator = PushRuleEvaluator(_flatten_dict(event), has_mentions, room_member_count, sender_power_level, notification_levels, related_events, self._related_event_match_enabled, event.room_version.msc3931_push_features, self.hs.config.experimental.msc1767_enabled)
        for (uid, rules) in rules_by_user.items():
            if event.sender == uid:
                continue
            display_name = None
            profile = profiles.get(uid)
            if profile:
                display_name = profile.display_name
            if not display_name:
                if event.type == EventTypes.Member and event.state_key == uid:
                    display_name = event.content.get('displayname', None)
                    if not isinstance(display_name, str):
                        display_name = None
            if count_as_unread:
                actions_by_user[uid] = []
            actions = evaluator.run(rules, uid, display_name)
            if 'notify' in actions:
                actions_by_user[uid] = actions
        if not actions_by_user:
            return
        uids_with_visibility = await filter_event_for_clients_with_state(self.store, actions_by_user.keys(), event, context)
        for user_id in set(actions_by_user).difference(uids_with_visibility):
            actions_by_user.pop(user_id, None)
        await self.store.add_push_actions_to_staging(event.event_id, actions_by_user, count_as_unread, thread_id)
MemberMap = Dict[str, Optional[EventIdMembership]]
Rule = Dict[str, dict]
RulesByUser = Dict[str, List[Rule]]
StateGroup = Union[object, int]

def _is_simple_value(value: Any) -> bool:
    if False:
        i = 10
        return i + 15
    return isinstance(value, (bool, str)) or type(value) is int or value is None

def _flatten_dict(d: Union[EventBase, Mapping[str, Any]], prefix: Optional[List[str]]=None, result: Optional[Dict[str, JsonValue]]=None) -> Dict[str, JsonValue]:
    if False:
        while True:
            i = 10
    '\n    Given a JSON dictionary (or event) which might contain sub dictionaries,\n    flatten it into a single layer dictionary by combining the keys & sub-keys.\n\n    String, integer, boolean, null or lists of those values are kept. All others are dropped.\n\n    Transforms:\n\n        {"foo": {"bar": "test"}}\n\n    To:\n\n        {"foo.bar": "test"}\n\n    Args:\n        d: The event or content to continue flattening.\n        prefix: The key prefix (from outer dictionaries).\n        result: The result to mutate.\n\n    Returns:\n        The resulting dictionary.\n    '
    if prefix is None:
        prefix = []
    if result is None:
        result = {}
    for (key, value) in d.items():
        key = key.replace('\\', '\\\\').replace('.', '\\.')
        if _is_simple_value(value):
            result['.'.join(prefix + [key])] = value
        elif isinstance(value, (list, tuple)):
            result['.'.join(prefix + [key])] = [v for v in value if _is_simple_value(v)]
        elif isinstance(value, Mapping):
            _flatten_dict(value, prefix=prefix + [key], result=result)
    if isinstance(d, EventBase) and PushRuleRoomFlag.EXTENSIBLE_EVENTS in d.room_version.msc3931_push_features:
        markup = d.get('content').get('m.markup')
        if d.room_version.identifier.startswith('org.matrix.msc1767.'):
            markup = d.get('content').get('org.matrix.msc1767.markup')
        if markup is not None and isinstance(markup, list):
            text = ''
            for rep in markup:
                if not isinstance(rep, dict):
                    break
                if rep.get('mimetype', 'text/plain') == 'text/plain':
                    rep_text = rep.get('body')
                    if rep_text is not None and isinstance(rep_text, str):
                        text = rep_text.lower()
                        break
            result['content.body'] = text
    return result