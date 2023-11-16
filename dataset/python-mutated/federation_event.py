import collections
import itertools
import logging
from http import HTTPStatus
from typing import TYPE_CHECKING, Collection, Container, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from prometheus_client import Counter, Histogram
from synapse import event_auth
from synapse.api.constants import EventContentFields, EventTypes, GuestAccess, Membership, RejectedReason, RoomEncryptionAlgorithms
from synapse.api.errors import AuthError, Codes, EventSizeError, FederationError, FederationPullAttemptBackoffError, HttpResponseException, PartialStateConflictError, RequestSendFailed, SynapseError
from synapse.api.room_versions import KNOWN_ROOM_VERSIONS, RoomVersion, RoomVersions
from synapse.event_auth import auth_types_for_event, check_state_dependent_auth_rules, check_state_independent_auth_rules, validate_event_for_room_version
from synapse.events import EventBase
from synapse.events.snapshot import EventContext, UnpersistedEventContextBase
from synapse.federation.federation_client import InvalidResponseError, PulledPduInfo
from synapse.logging.context import nested_logging_context
from synapse.logging.opentracing import SynapseTags, set_tag, start_active_span, tag_args, trace
from synapse.metrics.background_process_metrics import run_as_background_process
from synapse.replication.http.devices import ReplicationMultiUserDevicesResyncRestServlet
from synapse.replication.http.federation import ReplicationFederationSendEventsRestServlet
from synapse.state import StateResolutionStore
from synapse.storage.databases.main.events_worker import EventRedactBehaviour
from synapse.types import PersistedEventPosition, RoomStreamToken, StateMap, StrCollection, UserID, get_domain_from_id
from synapse.types.state import StateFilter
from synapse.util.async_helpers import Linearizer, concurrently_execute
from synapse.util.iterutils import batch_iter, partition
from synapse.util.retryutils import NotRetryingDestination
from synapse.util.stringutils import shortstr
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)
soft_failed_event_counter = Counter('synapse_federation_soft_failed_events_total', 'Events received over federation that we marked as soft_failed')
backfill_processing_after_timer = Histogram('synapse_federation_backfill_processing_after_time_seconds', 'sec', [], buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0, 180.0, '+Inf'))

class FederationEventHandler:
    """Handles events that originated from federation.

    Responsible for handing incoming events and passing them on to the rest
    of the homeserver (including auth and state conflict resolutions)
    """

    def __init__(self, hs: 'HomeServer'):
        if False:
            i = 10
            return i + 15
        self._clock = hs.get_clock()
        self._store = hs.get_datastores().main
        self._storage_controllers = hs.get_storage_controllers()
        self._state_storage_controller = self._storage_controllers.state
        self._state_handler = hs.get_state_handler()
        self._event_creation_handler = hs.get_event_creation_handler()
        self._event_auth_handler = hs.get_event_auth_handler()
        self._message_handler = hs.get_message_handler()
        self._bulk_push_rule_evaluator = hs.get_bulk_push_rule_evaluator()
        self._state_resolution_handler = hs.get_state_resolution_handler()
        self._get_room_member_handler = hs.get_room_member_handler
        self._federation_client = hs.get_federation_client()
        self._third_party_event_rules = hs.get_module_api_callbacks().third_party_event_rules
        self._notifier = hs.get_notifier()
        self._is_mine_id = hs.is_mine_id
        self._is_mine_server_name = hs.is_mine_server_name
        self._server_name = hs.hostname
        self._instance_name = hs.get_instance_name()
        self._config = hs.config
        self._ephemeral_messages_enabled = hs.config.server.enable_ephemeral_messages
        self._send_events = ReplicationFederationSendEventsRestServlet.make_client(hs)
        if hs.config.worker.worker_app:
            self._multi_user_device_resync = ReplicationMultiUserDevicesResyncRestServlet.make_client(hs)
        else:
            self._device_list_updater = hs.get_device_handler().device_list_updater
        self.room_queues: Dict[str, List[Tuple[EventBase, str]]] = {}
        self._room_pdu_linearizer = Linearizer('fed_room_pdu')

    async def on_receive_pdu(self, origin: str, pdu: EventBase) -> None:
        """Process a PDU received via a federation /send/ transaction

        Args:
            origin: server which initiated the /send/ transaction. Will
                be used to fetch missing events or state.
            pdu: received PDU
        """
        assert not pdu.internal_metadata.outlier
        room_id = pdu.room_id
        event_id = pdu.event_id
        existing = await self._store.get_event(event_id, allow_none=True, allow_rejected=True)
        if existing:
            if not existing.internal_metadata.is_outlier():
                logger.info('Ignoring received event %s which we have already seen', event_id)
                return
            if pdu.internal_metadata.is_outlier():
                logger.info('Ignoring received outlier %s which we already have as an outlier', event_id)
                return
            logger.info('De-outliering event %s', event_id)
        try:
            self._sanity_check_event(pdu)
        except SynapseError as err:
            logger.warning('Received event failed sanity checks')
            raise FederationError('ERROR', err.code, err.msg, affected=pdu.event_id)
        if room_id in self.room_queues:
            logger.info('Queuing PDU from %s for now: join in progress', origin)
            self.room_queues[room_id].append((pdu, origin))
            return
        is_in_room = await self._event_auth_handler.is_host_in_room(room_id, self._server_name)
        if not is_in_room:
            logger.info("Ignoring PDU from %s as we're not in the room", origin)
            return None
        prevs = set(pdu.prev_event_ids())
        seen = await self._store.have_events_in_timeline(prevs)
        missing_prevs = prevs - seen
        if missing_prevs:
            min_depth = await self._store.get_min_depth(pdu.room_id)
            logger.debug('min_depth: %d', min_depth)
            if min_depth is not None and pdu.depth > min_depth:
                logger.info('Acquiring room lock to fetch %d missing prev_events: %s', len(missing_prevs), shortstr(missing_prevs))
                async with self._room_pdu_linearizer.queue(pdu.room_id):
                    logger.info('Acquired room lock to fetch %d missing prev_events', len(missing_prevs))
                    try:
                        await self._get_missing_events_for_pdu(origin, pdu, prevs, min_depth)
                    except Exception as e:
                        raise Exception('Error fetching missing prev_events for %s: %s' % (event_id, e)) from e
                seen = await self._store.have_events_in_timeline(prevs)
                missing_prevs = prevs - seen
                if not missing_prevs:
                    logger.info('Found all missing prev_events')
            if missing_prevs:
                logger.warning('Rejecting: failed to fetch %d prev events: %s', len(missing_prevs), shortstr(missing_prevs))
                raise FederationError('ERROR', 403, "Your server isn't divulging details about prev_events referenced in this event.", affected=pdu.event_id)
        try:
            context = await self._state_handler.compute_event_context(pdu)
            await self._process_received_pdu(origin, pdu, context)
        except PartialStateConflictError:
            logger.info('Room %s was un-partial stated while processing the PDU, trying again.', room_id)
            context = await self._state_handler.compute_event_context(pdu)
            await self._process_received_pdu(origin, pdu, context)

    async def on_send_membership_event(self, origin: str, event: EventBase) -> Tuple[EventBase, EventContext]:
        """
        We have received a join/leave/knock event for a room via send_join/leave/knock.

        Verify that event and send it into the room on the remote homeserver's behalf.

        This is quite similar to on_receive_pdu, with the following principal
        differences:
          * only membership events are permitted (and only events with
            sender==state_key -- ie, no kicks or bans)
          * *We* send out the event on behalf of the remote server.
          * We enforce the membership restrictions of restricted rooms.
          * Rejected events result in an exception rather than being stored.

        There are also other differences, however it is not clear if these are by
        design or omission. In particular, we do not attempt to backfill any missing
        prev_events.

        Args:
            origin: The homeserver of the remote (joining/invited/knocking) user.
            event: The member event that has been signed by the remote homeserver.

        Returns:
            The event and context of the event after inserting it into the room graph.

        Raises:
            RuntimeError if any prev_events are missing
            SynapseError if the event is not accepted into the room
            PartialStateConflictError if the room was un-partial stated in between
                computing the state at the event and persisting it. The caller should
                retry exactly once in this case.
        """
        logger.debug('on_send_membership_event: Got event: %s, signatures: %s', event.event_id, event.signatures)
        if get_domain_from_id(event.sender) != origin:
            logger.info('Got send_membership request for user %r from different origin %s', event.sender, origin)
            raise SynapseError(403, 'User not from origin', Codes.FORBIDDEN)
        if event.sender != event.state_key:
            raise SynapseError(400, 'state_key and sender must match', Codes.BAD_JSON)
        assert not event.internal_metadata.outlier
        event.internal_metadata.send_on_behalf_of = origin
        context = await self._state_handler.compute_event_context(event)
        await self._check_event_auth(origin, event, context)
        if context.rejected:
            raise SynapseError(403, f'{event.membership} event was rejected', Codes.FORBIDDEN)
        if event.membership == Membership.JOIN:
            await self.check_join_restrictions(context, event)
        if event.membership == Membership.KNOCK:
            (event_allowed, _) = await self._third_party_event_rules.check_event_allowed(event, context)
            if not event_allowed:
                logger.info('Sending of knock %s forbidden by third-party rules', event)
                raise SynapseError(403, 'This event is not allowed in this context', Codes.FORBIDDEN)
        await self._event_creation_handler.cache_joined_hosts_for_events([(event, context)])
        await self._check_for_soft_fail(event, context=context, origin=origin)
        await self._run_push_actions_and_persist_event(event, context)
        return (event, context)

    async def check_join_restrictions(self, context: UnpersistedEventContextBase, event: EventBase) -> None:
        """Check that restrictions in restricted join rules are matched

        Called when we receive a join event via send_join.

        Raises an auth error if the restrictions are not matched.
        """
        prev_state_ids = await context.get_prev_state_ids()
        user_id = event.state_key
        prev_member_event_id = prev_state_ids.get((EventTypes.Member, user_id), None)
        prev_membership = None
        if prev_member_event_id:
            prev_member_event = await self._store.get_event(prev_member_event_id)
            prev_membership = prev_member_event.membership
        await self._event_auth_handler.check_restricted_join_rules(prev_state_ids, event.room_version, user_id, prev_membership)

    @trace
    async def process_remote_join(self, origin: str, room_id: str, auth_events: List[EventBase], state: List[EventBase], event: EventBase, room_version: RoomVersion, partial_state: bool) -> int:
        """Persists the events returned by a send_join

        Checks the auth chain is valid (and passes auth checks) for the
        state and event. Then persists all of the events.
        Notifies about the persisted events where appropriate.

        Args:
            origin: Where the events came from
            room_id:
            auth_events
            state
            event
            room_version: The room version we expect this room to have, and
                will raise if it doesn't match the version in the create event.
            partial_state: True if the state omits non-critical membership events

        Returns:
            The stream ID after which all events have been persisted.

        Raises:
            SynapseError if the response is in some way invalid.
            PartialStateConflictError if the homeserver is already in the room and it
                has been un-partial stated.
        """
        create_event = None
        for e in state:
            if (e.type, e.state_key) == (EventTypes.Create, ''):
                create_event = e
                break
        if create_event is None:
            raise SynapseError(400, 'No create event in state')
        room_version_id = create_event.content.get('room_version', RoomVersions.V1.identifier)
        if room_version.identifier != room_version_id:
            raise SynapseError(400, 'Room version mismatch')
        await self._auth_and_persist_outliers(room_id, itertools.chain(auth_events, state))
        logger.info('Peristing join-via-remote %s (partial_state: %s)', event, partial_state)
        with nested_logging_context(suffix=event.event_id):
            if partial_state:
                prev_event_ids = set(event.prev_event_ids())
                seen_event_ids = await self._store.have_events_in_timeline(prev_event_ids)
                missing_event_ids = prev_event_ids - seen_event_ids
                state_maps_to_resolve: List[StateMap[str]] = []
                state_maps_to_resolve.extend((await self._state_storage_controller.get_state_groups_ids(room_id, seen_event_ids, await_full_state=False)).values())
                if missing_event_ids or len(state_maps_to_resolve) == 0:
                    state_maps_to_resolve.append({(e.type, e.state_key): e.event_id for e in state})
                state_ids_before_event = await self._state_resolution_handler.resolve_events_with_store(event.room_id, room_version.identifier, state_maps_to_resolve, event_map=None, state_res_store=StateResolutionStore(self._store))
            else:
                state_ids_before_event = {(e.type, e.state_key): e.event_id for e in state}
            context = await self._state_handler.compute_event_context(event, state_ids_before_event=state_ids_before_event, partial_state=partial_state)
            await self._check_event_auth(origin, event, context)
            if context.rejected:
                raise SynapseError(403, 'Join event was rejected')
            event.internal_metadata.proactively_send = False
            stream_id_after_persist = await self.persist_events_and_notify(room_id, [(event, context)])
            return stream_id_after_persist

    async def update_state_for_partial_state_event(self, destination: str, event: EventBase) -> None:
        """Recalculate the state at an event as part of a de-partial-stating process

        Args:
            destination: server to request full state from
            event: partial-state event to be de-partial-stated

        Raises:
            FederationPullAttemptBackoffError if we are are deliberately not attempting
                to pull the given event over federation because we've already done so
                recently and are backing off.
            FederationError if we fail to request state from the remote server.
        """
        logger.info('Updating state for %s', event.event_id)
        with nested_logging_context(suffix=event.event_id):
            context = await self._compute_event_context_with_maybe_missing_prevs(destination, event)
            if context.partial_state:
                logger.warning("%s still has prev_events with partial state: can't de-partial-state it yet", event.event_id)
                return
            await self._check_event_auth(None, event, context)
            await self._store.update_state_for_partial_state_event(event, context)
            self._state_storage_controller.notify_event_un_partial_stated(event.event_id)
            self._notifier.notify_replication()

    @trace
    async def backfill(self, dest: str, room_id: str, limit: int, extremities: StrCollection) -> None:
        """Trigger a backfill request to `dest` for the given `room_id`

        This will attempt to get more events from the remote. If the other side
        has no new events to offer, this will return an empty list.

        As the events are received, we check their signatures, and also do some
        sanity-checking on them. If any of the backfilled events are invalid,
        this method throws a SynapseError.

        We might also raise an InvalidResponseError if the response from the remote
        server is just bogus.

        TODO: make this more useful to distinguish failures of the remote
        server from invalid events (there is probably no point in trying to
        re-fetch invalid events from every other HS in the room.)
        """
        if self._is_mine_server_name(dest):
            raise SynapseError(400, "Can't backfill from self.")
        events = await self._federation_client.backfill(dest, room_id, limit=limit, extremities=extremities)
        if not events:
            return
        with backfill_processing_after_timer.time():
            for ev in events:
                if ev.room_id != room_id:
                    raise InvalidResponseError(f'Remote server {dest} returned event {ev.event_id} which is in room {ev.room_id}, when we were backfilling in {room_id}')
            await self._process_pulled_events(dest, events, backfilled=True)

    @trace
    async def _get_missing_events_for_pdu(self, origin: str, pdu: EventBase, prevs: Set[str], min_depth: int) -> None:
        """
        Args:
            origin: Origin of the pdu. Will be called to get the missing events
            pdu: received pdu
            prevs: List of event ids which we are missing
            min_depth: Minimum depth of events to return.
        """
        room_id = pdu.room_id
        event_id = pdu.event_id
        seen = await self._store.have_events_in_timeline(prevs)
        if not prevs - seen:
            return
        latest_frozen = await self._store.get_latest_event_ids_in_room(room_id)
        latest = seen | latest_frozen
        logger.info('Requesting missing events between %s and %s', shortstr(latest), event_id)
        try:
            missing_events = await self._federation_client.get_missing_events(origin, room_id, earliest_events_ids=list(latest), latest_events=[pdu], limit=10, min_depth=min_depth, timeout=60000)
        except (RequestSendFailed, HttpResponseException, NotRetryingDestination) as e:
            logger.warning('Failed to get prev_events: %s', e)
            return
        logger.info('Got %d prev_events', len(missing_events))
        await self._process_pulled_events(origin, missing_events, backfilled=False)

    @trace
    async def _process_pulled_events(self, origin: str, events: Collection[EventBase], backfilled: bool) -> None:
        """Process a batch of events we have pulled from a remote server

        Pulls in any events required to auth the events, persists the received events,
        and notifies clients, if appropriate.

        Assumes the events have already had their signatures and hashes checked.

        Params:
            origin: The server we received these events from
            events: The received events.
            backfilled: True if this is part of a historical batch of events (inhibits
                notification to clients, and validation of device keys.)
        """
        set_tag(SynapseTags.FUNC_ARG_PREFIX + 'event_ids', str([event.event_id for event in events]))
        set_tag(SynapseTags.FUNC_ARG_PREFIX + 'event_ids.length', str(len(events)))
        set_tag(SynapseTags.FUNC_ARG_PREFIX + 'backfilled', str(backfilled))
        logger.debug('processing pulled backfilled=%s events=%s', backfilled, ['event_id=%s,depth=%d,body=%s,prevs=%s\n' % (event.event_id, event.depth, event.content.get('body', event.type), event.prev_event_ids()) for event in events])
        existing_events_map = await self._store._get_events_from_db([event.event_id for event in events])
        new_events: List[EventBase] = []
        for event in events:
            event_id = event.event_id
            if event_id in existing_events_map:
                existing_event = existing_events_map[event_id]
                if not existing_event.event.internal_metadata.is_outlier():
                    logger.info('_process_pulled_event: Ignoring received event %s which we have already seen', event.event_id)
                    continue
                logger.info('De-outliering event %s', event_id)
            new_events.append(event)
        set_tag(SynapseTags.RESULT_PREFIX + 'new_events.length', str(len(new_events)))

        @trace
        async def _process_new_pulled_events(new_events: Collection[EventBase]) -> None:
            sorted_events = sorted(new_events, key=lambda x: x.depth)
            for ev in sorted_events:
                with nested_logging_context(ev.event_id):
                    await self._process_pulled_event(origin, ev, backfilled=backfilled)
        event_ids_with_failed_pull_attempts = await self._store.get_event_ids_with_failed_pull_attempts([event.event_id for event in new_events])
        (events_with_failed_pull_attempts, fresh_events) = partition(new_events, lambda e: e.event_id in event_ids_with_failed_pull_attempts)
        set_tag(SynapseTags.FUNC_ARG_PREFIX + 'events_with_failed_pull_attempts', str(event_ids_with_failed_pull_attempts))
        set_tag(SynapseTags.RESULT_PREFIX + 'events_with_failed_pull_attempts.length', str(len(events_with_failed_pull_attempts)))
        set_tag(SynapseTags.FUNC_ARG_PREFIX + 'fresh_events', str([event.event_id for event in fresh_events]))
        set_tag(SynapseTags.RESULT_PREFIX + 'fresh_events.length', str(len(fresh_events)))
        if len(events_with_failed_pull_attempts) > 0:
            run_as_background_process('_process_new_pulled_events_with_failed_pull_attempts', _process_new_pulled_events, events_with_failed_pull_attempts)
        if len(fresh_events) > 0:
            await _process_new_pulled_events(fresh_events)

    @trace
    @tag_args
    async def _process_pulled_event(self, origin: str, event: EventBase, backfilled: bool) -> None:
        """Process a single event that we have pulled from a remote server

        Pulls in any events required to auth the event, persists the received event,
        and notifies clients, if appropriate.

        Assumes the event has already had its signatures and hashes checked.

        This is somewhat equivalent to on_receive_pdu, but applies somewhat different
        logic in the case that we are missing prev_events (in particular, it just
        requests the state at that point, rather than triggering a get_missing_events) -
        so is appropriate when we have pulled the event from a remote server, rather
        than having it pushed to us.

        Params:
            origin: The server we received this event from
            events: The received event
            backfilled: True if this is part of a historical batch of events (inhibits
                notification to clients, and validation of device keys.)
        """
        logger.info('Processing pulled event %s', event)
        assert not event.internal_metadata.is_outlier(), 'Outlier event passed to _process_pulled_event. To persist an event as a non-outlier, make sure to pass in a copy without `event.internal_metadata.outlier = true`.'
        event_id = event.event_id
        try:
            self._sanity_check_event(event)
        except SynapseError as err:
            logger.warning('Event %s failed sanity check: %s', event_id, err)
            await self._store.record_event_failed_pull_attempt(event.room_id, event_id, str(err))
            return
        try:
            try:
                context = await self._compute_event_context_with_maybe_missing_prevs(origin, event)
                await self._process_received_pdu(origin, event, context, backfilled=backfilled)
            except PartialStateConflictError:
                context = await self._compute_event_context_with_maybe_missing_prevs(origin, event)
                if context.partial_state:
                    raise AssertionError(f'Event {event.event_id} still has a partial resolved state after room {event.room_id} was un-partial stated')
                await self._process_received_pdu(origin, event, context, backfilled=backfilled)
        except FederationPullAttemptBackoffError as exc:
            logger.warning('_process_pulled_event: %s', exc)
        except FederationError as e:
            await self._store.record_event_failed_pull_attempt(event.room_id, event_id, str(e))
            if e.code == 403:
                logger.warning('Pulled event %s failed history check.', event_id)
            else:
                raise

    @trace
    async def _compute_event_context_with_maybe_missing_prevs(self, dest: str, event: EventBase) -> EventContext:
        """Build an EventContext structure for a non-outlier event whose prev_events may
        be missing.

        This is used when we have pulled a batch of events from a remote server, and may
        not have all the prev_events.

        To build an EventContext, we need to calculate the state before the event. If we
        already have all the prev_events for `event`, we can simply use the state after
        the prev_events to calculate the state before `event`.

        Otherwise, the missing prevs become new backwards extremities, and we fall back
        to asking the remote server for the state after each missing `prev_event`,
        and resolving across them.

        That's ok provided we then resolve the state against other bits of the DAG
        before using it - in other words, that the received event `event` is not going
        to become the only forwards_extremity in the room (which will ensure that you
        can't just take over a room by sending an event, withholding its prev_events,
        and declaring yourself to be an admin in the subsequent state request).

        In other words: we should only call this method if `event` has been *pulled*
        as part of a batch of missing prev events, or similar.

        Params:
            dest: the remote server to ask for state at the missing prevs. Typically,
                this will be the server we got `event` from.
            event: an event to check for missing prevs.

        Returns:
            The event context.

        Raises:
            FederationPullAttemptBackoffError if we are are deliberately not attempting
                to pull one of the given event's `prev_event`s over federation because
                we've already done so recently and are backing off.
            FederationError if we fail to get the state from the remote server after any
                missing `prev_event`s.
        """
        room_id = event.room_id
        event_id = event.event_id
        prevs = set(event.prev_event_ids())
        seen = await self._store.have_events_in_timeline(prevs)
        missing_prevs = prevs - seen
        prevs_with_pull_backoff = await self._store.get_event_ids_to_not_pull_from_backoff(room_id, missing_prevs)
        if len(prevs_with_pull_backoff) > 0:
            raise FederationPullAttemptBackoffError(event_ids=prevs_with_pull_backoff.keys(), message=f'While computing context for event={event_id}, not attempting to pull missing prev_events={list(prevs_with_pull_backoff.keys())} because we already tried to pull recently (backing off).', retry_after_ms=max(prevs_with_pull_backoff.values()) - self._clock.time_msec())
        if not missing_prevs:
            return await self._state_handler.compute_event_context(event)
        logger.info('Event %s is missing prev_events %s: calculating state for a backwards extremity', event_id, shortstr(missing_prevs))
        try:
            partial_state_flags = await self._store.get_partial_state_events(seen)
            partial_state = any(partial_state_flags.values())
            ours = await self._state_storage_controller.get_state_groups_ids(room_id, seen, await_full_state=False)
            state_maps: List[StateMap[str]] = list(ours.values())
            del ours
            for p in missing_prevs:
                logger.info('Requesting state after missing prev_event %s', p)
                with nested_logging_context(p):
                    remote_state_map = await self._get_state_ids_after_missing_prev_event(dest, room_id, p)
                    state_maps.append(remote_state_map)
            room_version = await self._store.get_room_version_id(room_id)
            state_map = await self._state_resolution_handler.resolve_events_with_store(room_id, room_version, state_maps, event_map={event_id: event}, state_res_store=StateResolutionStore(self._store))
        except Exception as e:
            logger.warning('Error attempting to resolve state at missing prev_events: %s', e)
            raise FederationError('ERROR', 403, "We can't get valid state history.", affected=event_id)
        return await self._state_handler.compute_event_context(event, state_ids_before_event=state_map, partial_state=partial_state)

    @trace
    @tag_args
    async def _get_state_ids_after_missing_prev_event(self, destination: str, room_id: str, event_id: str) -> StateMap[str]:
        """Requests all of the room state at a given event from a remote homeserver.

        Args:
            destination: The remote homeserver to query for the state.
            room_id: The id of the room we're interested in.
            event_id: The id of the event we want the state at.

        Returns:
            The event ids of the state *after* the given event.

        Raises:
            InvalidResponseError: if the remote homeserver's response contains fields
                of the wrong type.
        """
        (state_event_ids, auth_event_ids) = await self._federation_client.get_room_state_ids(destination, room_id, event_id=event_id)
        logger.debug('state_ids returned %i state events, %i auth events', len(state_event_ids), len(auth_event_ids))
        desired_events = set(state_event_ids)
        desired_events.add(event_id)
        logger.debug('Fetching %i events from cache/store', len(desired_events))
        have_events = await self._store.have_seen_events(room_id, desired_events)
        missing_desired_event_ids = desired_events - have_events
        logger.debug('We are missing %i events (got %i)', len(missing_desired_event_ids), len(have_events))
        missing_auth_event_ids = set(auth_event_ids) - have_events
        missing_auth_event_ids.difference_update(await self._store.have_seen_events(room_id, missing_auth_event_ids))
        logger.debug('We are also missing %i auth events', len(missing_auth_event_ids))
        missing_event_ids = missing_desired_event_ids | missing_auth_event_ids
        set_tag(SynapseTags.RESULT_PREFIX + 'missing_auth_event_ids', str(missing_auth_event_ids))
        set_tag(SynapseTags.RESULT_PREFIX + 'missing_auth_event_ids.length', str(len(missing_auth_event_ids)))
        set_tag(SynapseTags.RESULT_PREFIX + 'missing_desired_event_ids', str(missing_desired_event_ids))
        set_tag(SynapseTags.RESULT_PREFIX + 'missing_desired_event_ids.length', str(len(missing_desired_event_ids)))
        if len(missing_event_ids) * 10 >= len(auth_event_ids) + len(state_event_ids):
            logger.debug('Requesting complete state from remote')
            await self._get_state_and_persist(destination, room_id, event_id)
        else:
            logger.debug('Fetching %i events from remote', len(missing_event_ids))
            await self._get_events_and_persist(destination=destination, room_id=room_id, event_ids=missing_event_ids)
        state_map = {}
        event_metadata = await self._store.get_metadata_for_events(state_event_ids)
        for (state_event_id, metadata) in event_metadata.items():
            if metadata.room_id != room_id:
                logger.warning('Remote server %s claims event %s in room %s is an auth/state event in room %s', destination, state_event_id, metadata.room_id, room_id)
                continue
            if metadata.state_key is None:
                logger.warning('Remote server gave us non-state event in state: %s', state_event_id)
                continue
            state_map[metadata.event_type, metadata.state_key] = state_event_id
        remote_event = await self._store.get_event(event_id, allow_none=True, allow_rejected=True, redact_behaviour=EventRedactBehaviour.as_is)
        if not remote_event:
            raise Exception('Unable to get missing prev_event %s' % (event_id,))
        failed_to_fetch = desired_events - event_metadata.keys()
        failed_to_fetch.discard(event_id)
        if failed_to_fetch:
            logger.warning('Failed to fetch missing state events for %s %s', event_id, failed_to_fetch)
            set_tag(SynapseTags.RESULT_PREFIX + 'failed_to_fetch', str(failed_to_fetch))
            set_tag(SynapseTags.RESULT_PREFIX + 'failed_to_fetch.length', str(len(failed_to_fetch)))
        if remote_event.is_state() and remote_event.rejected_reason is None:
            state_map[remote_event.type, remote_event.state_key] = remote_event.event_id
        return state_map

    @trace
    @tag_args
    async def _get_state_and_persist(self, destination: str, room_id: str, event_id: str) -> None:
        """Get the complete room state at a given event, and persist any new events
        as outliers"""
        room_version = await self._store.get_room_version(room_id)
        (auth_events, state_events) = await self._federation_client.get_room_state(destination, room_id, event_id=event_id, room_version=room_version)
        logger.info('/state returned %i events', len(auth_events) + len(state_events))
        await self._auth_and_persist_outliers(room_id, itertools.chain(auth_events, state_events))
        if not await self._store.have_seen_event(room_id, event_id):
            await self._get_events_and_persist(destination=destination, room_id=room_id, event_ids=(event_id,))

    @trace
    async def _process_received_pdu(self, origin: str, event: EventBase, context: EventContext, backfilled: bool=False) -> None:
        """Called when we have a new non-outlier event.

        This is called when we have a new event to add to the room DAG. This can be
        due to:
           * events received directly via a /send request
           * events retrieved via get_missing_events after a /send request
           * events backfilled after a client request.

        It's not currently used for events received from incoming send_{join,knock,leave}
        requests (which go via on_send_membership_event), nor for joins created by a
        remote join dance (which go via process_remote_join).

        We need to do auth checks and put it through the StateHandler.

        Args:
            origin: server sending the event

            event: event to be persisted

            context: The `EventContext` to persist the event with.

            backfilled: True if this is part of a historical batch of events (inhibits
                notification to clients, and validation of device keys.)

        PartialStateConflictError: if the room was un-partial stated in between
            computing the state at the event and persisting it. The caller should
            recompute `context` and retry exactly once when this happens.
        """
        logger.debug('Processing event: %s', event)
        assert not event.internal_metadata.outlier
        try:
            await self._check_event_auth(origin, event, context)
        except AuthError as e:
            raise FederationError('ERROR', e.code, e.msg, affected=event.event_id)
        if not backfilled and (not context.rejected):
            await self._check_for_soft_fail(event, context=context, origin=origin)
        await self._run_push_actions_and_persist_event(event, context, backfilled)
        if backfilled or context.rejected:
            return
        await self._maybe_kick_guest_users(event)
        if event.type == EventTypes.Encrypted:
            device_id = event.content.get('device_id')
            sender_key = event.content.get('sender_key')
            cached_devices = await self._store.get_cached_devices_for_user(event.sender)
            resync = False
            device = None
            if device_id is not None:
                device = cached_devices.get(device_id)
                if device is None:
                    logger.info('Received event from remote device not in our cache: %s %s', event.sender, device_id)
                    resync = True
            if sender_key is not None:
                current_keys: Container[str] = []
                if device:
                    keys = device.get('keys', {}).get('keys', {})
                    if event.content.get('algorithm') == RoomEncryptionAlgorithms.MEGOLM_V1_AES_SHA2:
                        key_name = 'curve25519:%s' % (device_id,)
                        current_keys = [keys.get(key_name)]
                    else:
                        current_keys = keys.values()
                elif device_id:
                    pass
                else:
                    current_keys = [key for device in cached_devices.values() for key in device.get('keys', {}).get('keys', {}).values()]
                if sender_key not in current_keys:
                    logger.info('Received event from remote device with unexpected sender key: %s %s: %s', event.sender, device_id or '<no device_id>', sender_key)
                    resync = True
            if resync:
                run_as_background_process('resync_device_due_to_pdu', self._resync_device, event.sender)

    async def _resync_device(self, sender: str) -> None:
        """We have detected that the device list for the given user may be out
        of sync, so we try and resync them.
        """
        try:
            await self._store.mark_remote_users_device_caches_as_stale((sender,))
            if self._config.worker.worker_app:
                await self._multi_user_device_resync(user_ids=[sender])
            else:
                await self._device_list_updater.multi_user_device_resync(user_ids=[sender])
        except Exception:
            logger.exception('Failed to resync device for %s', sender)

    async def backfill_event_id(self, destinations: StrCollection, room_id: str, event_id: str) -> PulledPduInfo:
        """Backfill a single event and persist it as a non-outlier which means
        we also pull in all of the state and auth events necessary for it.

        Args:
            destination: The homeserver to pull the given event_id from.
            room_id: The room where the event is from.
            event_id: The event ID to backfill.

        Raises:
            FederationError if we are unable to find the event from the destination
        """
        logger.info('backfill_event_id: event_id=%s', event_id)
        room_version = await self._store.get_room_version(room_id)
        pulled_pdu_info = await self._federation_client.get_pdu(destinations, event_id, room_version)
        if not pulled_pdu_info:
            raise FederationError('ERROR', 404, f'Unable to find event_id={event_id} from remote servers to backfill.', affected=event_id)
        await self._process_pulled_events(pulled_pdu_info.pull_origin, [pulled_pdu_info.pdu], backfilled=True)
        return pulled_pdu_info

    @trace
    @tag_args
    async def _get_events_and_persist(self, destination: str, room_id: str, event_ids: StrCollection) -> None:
        """Fetch the given events from a server, and persist them as outliers.

        This function *does not* recursively get missing auth events of the
        newly fetched events. Callers must include in the `event_ids` argument
        any missing events from the auth chain.

        Logs a warning if we can't find the given event.
        """
        room_version = await self._store.get_room_version(room_id)
        events: List[EventBase] = []

        async def get_event(event_id: str) -> None:
            with nested_logging_context(event_id):
                try:
                    pulled_pdu_info = await self._federation_client.get_pdu([destination], event_id, room_version)
                    if pulled_pdu_info is None:
                        logger.warning("Server %s didn't return event %s", destination, event_id)
                        return
                    events.append(pulled_pdu_info.pdu)
                except Exception as e:
                    logger.warning('Error fetching missing state/auth event %s: %s %s', event_id, type(e), e)
        await concurrently_execute(get_event, event_ids, 5)
        logger.info('Fetched %i events of %i requested', len(events), len(event_ids))
        await self._auth_and_persist_outliers(room_id, events)

    @trace
    async def _auth_and_persist_outliers(self, room_id: str, events: Iterable[EventBase]) -> None:
        """Persist a batch of outlier events fetched from remote servers.

        We first sort the events to make sure that we process each event's auth_events
        before the event itself.

        We then mark the events as outliers, persist them to the database, and, where
        appropriate (eg, an invite), awake the notifier.

        Params:
            room_id: the room that the events are meant to be in (though this has
               not yet been checked)
            events: the events that have been fetched
        """
        event_map = {event.event_id: event for event in events}
        event_ids = event_map.keys()
        set_tag(SynapseTags.FUNC_ARG_PREFIX + 'event_ids', str(event_ids))
        set_tag(SynapseTags.FUNC_ARG_PREFIX + 'event_ids.length', str(len(event_ids)))
        seen_remotes = await self._store.have_seen_events(room_id, event_map.keys())
        for s in seen_remotes:
            event_map.pop(s, None)
        while event_map:
            roots = tuple((ev for ev in event_map.values() if not any((aid in event_map for aid in ev.auth_event_ids()))))
            if not roots:
                logger.warning('Loop found in auth events while fetching missing state/auth events: %s', shortstr(event_map.keys()))
                return
            logger.info('Persisting %i of %i remaining outliers: %s', len(roots), len(event_map), shortstr((e.event_id for e in roots)))
            await self._auth_and_persist_outliers_inner(room_id, roots)
            for ev in roots:
                del event_map[ev.event_id]

    async def _auth_and_persist_outliers_inner(self, room_id: str, fetched_events: Collection[EventBase]) -> None:
        """Helper for _auth_and_persist_outliers

        Persists a batch of events where we have (theoretically) already persisted all
        of their auth events.

        Marks the events as outliers, auths them, persists them to the database, and,
        where appropriate (eg, an invite), awakes the notifier.

        Params:
            origin: where the events came from
            room_id: the room that the events are meant to be in (though this has
               not yet been checked)
            fetched_events: the events to persist
        """
        auth_events = {aid for event in fetched_events for aid in event.auth_event_ids()}
        persisted_events = await self._store.get_events(auth_events, allow_rejected=True)
        events_and_contexts_to_persist: List[Tuple[EventBase, EventContext]] = []

        async def prep(event: EventBase) -> None:
            with nested_logging_context(suffix=event.event_id):
                auth = []
                for auth_event_id in event.auth_event_ids():
                    ae = persisted_events.get(auth_event_id)
                    if not ae:
                        logger.warning('Dropping event %s, which relies on auth_event %s, which could not be found', event, auth_event_id)
                        return
                    auth.append(ae)
                event.internal_metadata.outlier = True
                context = EventContext.for_outlier(self._storage_controllers)
                try:
                    validate_event_for_room_version(event)
                    await check_state_independent_auth_rules(self._store, event)
                    check_state_dependent_auth_rules(event, auth)
                except AuthError as e:
                    logger.warning('Rejecting %r because %s', event, e)
                    context.rejected = RejectedReason.AUTH_ERROR
                except EventSizeError as e:
                    if e.unpersistable:
                        raise e
                    logger.warning('While validating received event %r: %s', event, e)
                    context.rejected = RejectedReason.OVERSIZED_EVENT
            events_and_contexts_to_persist.append((event, context))
        for event in fetched_events:
            await prep(event)
        await self.persist_events_and_notify(room_id, events_and_contexts_to_persist, backfilled=True)

    @trace
    async def _check_event_auth(self, origin: Optional[str], event: EventBase, context: EventContext) -> None:
        """
        Checks whether an event should be rejected (for failing auth checks).

        Args:
            origin: The host the event originates from. This is used to fetch
               any missing auth events. It can be set to None, but only if we are
               sure that we already have all the auth events.
            event: The event itself.
            context:
                The event context.

        Raises:
            AuthError if we were unable to find copies of the event's auth events.
               (Most other failures just cause us to set `context.rejected`.)
        """
        assert not event.internal_metadata.outlier
        try:
            validate_event_for_room_version(event)
        except AuthError as e:
            logger.warning('While validating received event %r: %s', event, e)
            context.rejected = RejectedReason.AUTH_ERROR
            return
        except EventSizeError as e:
            if e.unpersistable:
                raise e
            logger.warning('While validating received event %r: %s', event, e)
            context.rejected = RejectedReason.OVERSIZED_EVENT
            return
        claimed_auth_events = await self._load_or_fetch_auth_events_for_event(origin, event)
        set_tag(SynapseTags.RESULT_PREFIX + 'claimed_auth_events', str([ev.event_id for ev in claimed_auth_events]))
        set_tag(SynapseTags.RESULT_PREFIX + 'claimed_auth_events.length', str(len(claimed_auth_events)))
        try:
            await check_state_independent_auth_rules(self._store, event)
            check_state_dependent_auth_rules(event, claimed_auth_events)
        except AuthError as e:
            logger.warning('While checking auth of %r against auth_events: %s', event, e)
            context.rejected = RejectedReason.AUTH_ERROR
            return
        if context.partial_state:
            room_version = await self._store.get_room_version_id(event.room_id)
            local_state_id_map = await context.get_prev_state_ids()
            claimed_auth_events_id_map = {(ev.type, ev.state_key): ev.event_id for ev in claimed_auth_events}
            state_for_auth_id_map = await self._state_resolution_handler.resolve_events_with_store(event.room_id, room_version, [local_state_id_map, claimed_auth_events_id_map], event_map=None, state_res_store=StateResolutionStore(self._store))
        else:
            event_types = event_auth.auth_types_for_event(event.room_version, event)
            state_for_auth_id_map = await context.get_prev_state_ids(StateFilter.from_types(event_types))
        calculated_auth_event_ids = self._event_auth_handler.compute_auth_events(event, state_for_auth_id_map, for_verification=True)
        if collections.Counter(event.auth_event_ids()) == collections.Counter(calculated_auth_event_ids):
            return
        calculated_auth_events = await self._store.get_events_as_list(calculated_auth_event_ids)
        claimed_auth_event_map = {(e.type, e.state_key): e for e in claimed_auth_events}
        calculated_auth_event_map = {(e.type, e.state_key): e for e in calculated_auth_events}
        logger.info("event's auth_events are different to our calculated auth_events. Claimed but not calculated: %s. Calculated but not claimed: %s", [ev for (k, ev) in claimed_auth_event_map.items() if k not in calculated_auth_event_map or calculated_auth_event_map[k].event_id != ev.event_id], [ev for (k, ev) in calculated_auth_event_map.items() if k not in claimed_auth_event_map or claimed_auth_event_map[k].event_id != ev.event_id])
        try:
            check_state_dependent_auth_rules(event, calculated_auth_events)
        except AuthError as e:
            logger.warning('While checking auth of %r against room state before the event: %s', event, e)
            context.rejected = RejectedReason.AUTH_ERROR

    @trace
    async def _maybe_kick_guest_users(self, event: EventBase) -> None:
        if event.type != EventTypes.GuestAccess:
            return
        guest_access = event.content.get(EventContentFields.GUEST_ACCESS)
        if guest_access == GuestAccess.CAN_JOIN:
            return
        current_state = await self._storage_controllers.state.get_current_state(event.room_id)
        current_state_list = list(current_state.values())
        await self._get_room_member_handler().kick_guest_users(current_state_list)

    async def _check_for_soft_fail(self, event: EventBase, context: EventContext, origin: str) -> None:
        """Checks if we should soft fail the event; if so, marks the event as
        such.

        Does nothing for events in rooms with partial state, since we may not have an
        accurate membership event for the sender in the current state.

        Args:
            event
            context: The `EventContext` which we are about to persist the event with.
            origin: The host the event originates from.
        """
        if await self._store.is_partial_state_room(event.room_id):
            return
        extrem_ids = await self._store.get_latest_event_ids_in_room(event.room_id)
        prev_event_ids = set(event.prev_event_ids())
        if extrem_ids == prev_event_ids:
            return
        room_version = await self._store.get_room_version_id(event.room_id)
        room_version_obj = KNOWN_ROOM_VERSIONS[room_version]
        auth_types = auth_types_for_event(room_version_obj, event)
        seen_event_ids = await self._store.have_events_in_timeline(prev_event_ids)
        has_missing_prevs = bool(prev_event_ids - seen_event_ids)
        if has_missing_prevs:
            state_sets_d = await self._state_storage_controller.get_state_groups_ids(event.room_id, extrem_ids)
            state_sets: List[StateMap[str]] = list(state_sets_d.values())
            state_ids = await context.get_prev_state_ids()
            state_sets.append(state_ids)
            current_state_ids = await self._state_resolution_handler.resolve_events_with_store(event.room_id, room_version, state_sets, event_map=None, state_res_store=StateResolutionStore(self._store))
        else:
            current_state_ids = await self._state_storage_controller.get_current_state_ids(event.room_id, StateFilter.from_types(auth_types))
        logger.debug('Doing soft-fail check for %s: state %s', event.event_id, current_state_ids)
        current_state_ids_list = [e for (k, e) in current_state_ids.items() if k in auth_types]
        current_auth_events = await self._store.get_events_as_list(current_state_ids_list)
        try:
            check_state_dependent_auth_rules(event, current_auth_events)
        except AuthError as e:
            logger.warning('Soft-failing %r (from %s) because %s', event, e, origin, extra={'room_id': event.room_id, 'mxid': event.sender, 'hs': origin})
            soft_failed_event_counter.inc()
            event.internal_metadata.soft_failed = True

    async def _load_or_fetch_auth_events_for_event(self, destination: Optional[str], event: EventBase) -> Collection[EventBase]:
        """Fetch this event's auth_events, from database or remote

        Loads any of the auth_events that we already have from the database/cache. If
        there are any that are missing, calls /event_auth to get the complete auth
        chain for the event (and then attempts to load the auth_events again).

        If any of the auth_events cannot be found, raises an AuthError. This can happen
        for a number of reasons; eg: the events don't exist, or we were unable to talk
        to `destination`, or we couldn't validate the signature on the event (which
        in turn has multiple potential causes).

        Args:
            destination: where to send the /event_auth request. Typically the server
               that sent us `event` in the first place.

               If this is None, no attempt is made to load any missing auth events:
               rather, an AssertionError is raised if there are any missing events.

            event: the event whose auth_events we want

        Returns:
            all of the events listed in `event.auth_events_ids`, after deduplication

        Raises:
            AssertionError if some auth events were missing and no `destination` was
            supplied.

            AuthError if we were unable to fetch the auth_events for any reason.
        """
        event_auth_event_ids = set(event.auth_event_ids())
        event_auth_events = await self._store.get_events(event_auth_event_ids, allow_rejected=True)
        missing_auth_event_ids = event_auth_event_ids.difference(event_auth_events.keys())
        if not missing_auth_event_ids:
            return event_auth_events.values()
        if destination is None:
            raise AssertionError('_load_or_fetch_auth_events_for_event() called with no destination for an event with missing auth_events')
        logger.info('Event %s refers to unknown auth events %s: fetching auth chain', event, missing_auth_event_ids)
        try:
            await self._get_remote_auth_chain_for_event(destination, event.room_id, event.event_id)
        except Exception as e:
            logger.warning('Failed to get auth chain for %s: %s', event, e)
        extra_auth_events = await self._store.get_events(missing_auth_event_ids, allow_rejected=True)
        missing_auth_event_ids.difference_update(extra_auth_events.keys())
        event_auth_events.update(extra_auth_events)
        if not missing_auth_event_ids:
            return event_auth_events.values()
        logger.warning('Missing auth events for %s: %s', event, shortstr(missing_auth_event_ids))
        raise AuthError(code=HTTPStatus.FORBIDDEN, msg='Auth events could not be found')

    @trace
    @tag_args
    async def _get_remote_auth_chain_for_event(self, destination: str, room_id: str, event_id: str) -> None:
        """If we are missing some of an event's auth events, attempt to request them

        Args:
            destination: where to fetch the auth tree from
            room_id: the room in which we are lacking auth events
            event_id: the event for which we are lacking auth events
        """
        try:
            remote_events = await self._federation_client.get_event_auth(destination, room_id, event_id)
        except RequestSendFailed as e1:
            logger.info('Failed to get event auth from remote: %s', e1)
            return
        logger.info('/event_auth returned %i events', len(remote_events))
        remote_auth_events = (e for e in remote_events if e.event_id != event_id)
        await self._auth_and_persist_outliers(room_id, remote_auth_events)

    @trace
    async def _run_push_actions_and_persist_event(self, event: EventBase, context: EventContext, backfilled: bool=False) -> None:
        """Run the push actions for a received event, and persist it.

        Args:
            event: The event itself.
            context: The event context.
            backfilled: True if the event was backfilled.

        PartialStateConflictError: if attempting to persist a partial state event in
            a room that has been un-partial stated.
        """
        assert not event.internal_metadata.outlier
        if not backfilled and (not context.rejected):
            min_depth = await self._store.get_min_depth(event.room_id)
            if min_depth is None or min_depth > event.depth:
                logger.info('Skipping push actions for old event with depth %s < %s', event.depth, min_depth)
            else:
                await self._bulk_push_rule_evaluator.action_for_events_by_user([(event, context)])
        try:
            await self.persist_events_and_notify(event.room_id, [(event, context)], backfilled=backfilled)
        except Exception:
            await self._store.remove_push_actions_from_staging(event.event_id)
            raise

    async def persist_events_and_notify(self, room_id: str, event_and_contexts: Sequence[Tuple[EventBase, EventContext]], backfilled: bool=False) -> int:
        """Persists events and tells the notifier/pushers about them, if
        necessary.

        Args:
            room_id: The room ID of events being persisted.
            event_and_contexts: Sequence of events with their associated
                context that should be persisted. All events must belong to
                the same room.
            backfilled: Whether these events are a result of
                backfilling or not

        Returns:
            The stream ID after which all events have been persisted.

        Raises:
            PartialStateConflictError: if attempting to persist a partial state event in
                a room that has been un-partial stated.
        """
        if not event_and_contexts:
            return self._store.get_room_max_stream_ordering()
        instance = self._config.worker.events_shard_config.get_instance(room_id)
        if instance != self._instance_name:
            result = {}
            try:
                for batch in batch_iter(event_and_contexts, 200):
                    result = await self._send_events(instance_name=instance, store=self._store, room_id=room_id, event_and_contexts=batch, backfilled=backfilled)
            except SynapseError as e:
                if e.code == HTTPStatus.CONFLICT:
                    raise PartialStateConflictError()
                raise
            return result['max_stream_id']
        else:
            assert self._storage_controllers.persistence
            (events, max_stream_token) = await self._storage_controllers.persistence.persist_events(event_and_contexts, backfilled=backfilled)
            self._notifier.notify_replication()
            if self._ephemeral_messages_enabled:
                for event in events:
                    self._message_handler.maybe_schedule_expiry(event)
            if not backfilled:
                with start_active_span('notify_persisted_events'):
                    set_tag(SynapseTags.RESULT_PREFIX + 'event_ids', str([ev.event_id for ev in events]))
                    set_tag(SynapseTags.RESULT_PREFIX + 'event_ids.length', str(len(events)))
                    for event in events:
                        await self._notify_persisted_event(event, max_stream_token)
            return max_stream_token.stream

    async def _notify_persisted_event(self, event: EventBase, max_stream_token: RoomStreamToken) -> None:
        """Checks to see if notifier/pushers should be notified about the
        event or not.

        Args:
            event:
            max_stream_token: The max_stream_id returned by persist_events
        """
        extra_users = []
        if event.type == EventTypes.Member:
            target_user_id = event.state_key
            if event.internal_metadata.is_outlier():
                if event.membership != Membership.INVITE:
                    if not self._is_mine_id(target_user_id):
                        return
            target_user = UserID.from_string(target_user_id)
            extra_users.append(target_user)
        elif event.internal_metadata.is_outlier():
            return
        assert event.internal_metadata.stream_ordering
        event_pos = PersistedEventPosition(self._instance_name, event.internal_metadata.stream_ordering)
        await self._notifier.on_new_room_events([(event, event_pos)], max_stream_token, extra_users=extra_users)
        if event.type == EventTypes.Member and event.membership == Membership.JOIN:
            self._notifier.notify_user_joined_room(event.event_id, event.room_id)
        if event.type == EventTypes.ServerACL:
            self._state_storage_controller.get_server_acl_for_room.invalidate((event.room_id,))

    def _sanity_check_event(self, ev: EventBase) -> None:
        if False:
            print('Hello World!')
        "\n        Do some early sanity checks of a received event\n\n        In particular, checks it doesn't have an excessive number of\n        prev_events or auth_events, which could cause a huge state resolution\n        or cascade of event fetches.\n\n        Args:\n            ev: event to be checked\n\n        Raises:\n            SynapseError if the event does not pass muster\n        "
        if len(ev.prev_event_ids()) > 20:
            logger.warning('Rejecting event %s which has %i prev_events', ev.event_id, len(ev.prev_event_ids()))
            raise SynapseError(HTTPStatus.BAD_REQUEST, 'Too many prev_events')
        if len(ev.auth_event_ids()) > 10:
            logger.warning('Rejecting event %s which has %i auth_events', ev.event_id, len(ev.auth_event_ids()))
            raise SynapseError(HTTPStatus.BAD_REQUEST, 'Too many auth_events')