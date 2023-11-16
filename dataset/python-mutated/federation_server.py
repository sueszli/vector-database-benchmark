import logging
import random
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Collection, Dict, List, Mapping, Optional, Tuple, Union
from prometheus_client import Counter, Gauge, Histogram
from twisted.python import failure
from synapse.api.constants import Direction, EduTypes, EventContentFields, EventTypes, Membership
from synapse.api.errors import AuthError, Codes, FederationError, IncompatibleRoomVersionError, NotFoundError, PartialStateConflictError, SynapseError, UnsupportedRoomVersionError
from synapse.api.room_versions import KNOWN_ROOM_VERSIONS, RoomVersion
from synapse.crypto.event_signing import compute_event_signature
from synapse.events import EventBase
from synapse.events.snapshot import EventContext
from synapse.federation.federation_base import FederationBase, InvalidEventSignatureError, event_from_pdu_json
from synapse.federation.persistence import TransactionActions
from synapse.federation.units import Edu, Transaction
from synapse.handlers.worker_lock import NEW_EVENT_DURING_PURGE_LOCK_NAME
from synapse.http.servlet import assert_params_in_dict
from synapse.logging.context import make_deferred_yieldable, nested_logging_context, run_in_background
from synapse.logging.opentracing import SynapseTags, log_kv, set_tag, start_active_span_from_edu, tag_args, trace
from synapse.metrics.background_process_metrics import wrap_as_background_process
from synapse.replication.http.federation import ReplicationFederationSendEduRestServlet, ReplicationGetQueryRestServlet
from synapse.storage.databases.main.lock import Lock
from synapse.storage.databases.main.roommember import extract_heroes_from_room_summary
from synapse.storage.roommember import MemberSummary
from synapse.types import JsonDict, StateMap, UserID, get_domain_from_id
from synapse.util import unwrapFirstError
from synapse.util.async_helpers import Linearizer, concurrently_execute, gather_results
from synapse.util.caches.response_cache import ResponseCache
from synapse.util.stringutils import parse_server_name
if TYPE_CHECKING:
    from synapse.server import HomeServer
TRANSACTION_CONCURRENCY_LIMIT = 10
logger = logging.getLogger(__name__)
received_pdus_counter = Counter('synapse_federation_server_received_pdus', '')
received_edus_counter = Counter('synapse_federation_server_received_edus', '')
received_queries_counter = Counter('synapse_federation_server_received_queries', '', ['type'])
pdu_process_time = Histogram('synapse_federation_server_pdu_process_time', 'Time taken to process an event')
last_pdu_ts_metric = Gauge('synapse_federation_last_received_pdu_time', 'The timestamp of the last PDU which was successfully received from the given domain', labelnames=('server_name',))
_INBOUND_EVENT_HANDLING_LOCK_NAME = 'federation_inbound_pdu'

class FederationServer(FederationBase):

    def __init__(self, hs: 'HomeServer'):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(hs)
        self.server_name = hs.hostname
        self.handler = hs.get_federation_handler()
        self._spam_checker_module_callbacks = hs.get_module_api_callbacks().spam_checker
        self._federation_event_handler = hs.get_federation_event_handler()
        self.state = hs.get_state_handler()
        self._event_auth_handler = hs.get_event_auth_handler()
        self._room_member_handler = hs.get_room_member_handler()
        self._e2e_keys_handler = hs.get_e2e_keys_handler()
        self._worker_lock_handler = hs.get_worker_locks_handler()
        self._state_storage_controller = hs.get_storage_controllers().state
        self.device_handler = hs.get_device_handler()
        hs.get_directory_handler()
        self._server_linearizer = Linearizer('fed_server')
        self._active_transactions: Dict[str, str] = {}
        self._transaction_resp_cache: ResponseCache[Tuple[str, str]] = ResponseCache(hs.get_clock(), 'fed_txn_handler', timeout_ms=30000)
        self.transaction_actions = TransactionActions(self.store)
        self.registry = hs.get_federation_registry()
        self._state_resp_cache: ResponseCache[Tuple[str, Optional[str]]] = ResponseCache(hs.get_clock(), 'state_resp', timeout_ms=30000)
        self._state_ids_resp_cache: ResponseCache[Tuple[str, str]] = ResponseCache(hs.get_clock(), 'state_ids_resp', timeout_ms=30000)
        self._federation_metrics_domains = hs.config.federation.federation_metrics_domains
        self._room_prejoin_state_types = hs.config.api.room_prejoin_state
        self._started_handling_of_staged_events = False

    @wrap_as_background_process('_handle_old_staged_events')
    async def _handle_old_staged_events(self) -> None:
        """Handle old staged events by fetching all rooms that have staged
        events and start the processing of each of those rooms.
        """
        room_ids = await self.store.get_all_rooms_with_staged_incoming_events()
        random.shuffle(room_ids)
        for room_id in room_ids:
            room_version = await self.store.get_room_version(room_id)
            lock = await self.store.try_acquire_lock(_INBOUND_EVENT_HANDLING_LOCK_NAME, room_id)
            if lock:
                logger.info('Handling old staged inbound events in %s', room_id)
                self._process_incoming_pdus_in_room_inner(room_id, room_version, lock)
            await self._clock.sleep(random.uniform(0, 0.1))

    async def on_backfill_request(self, origin: str, room_id: str, versions: List[str], limit: int) -> Tuple[int, Dict[str, Any]]:
        async with self._server_linearizer.queue((origin, room_id)):
            (origin_host, _) = parse_server_name(origin)
            await self.check_server_matches_acl(origin_host, room_id)
            pdus = await self.handler.on_backfill_request(origin, room_id, versions, limit)
            res = self._transaction_dict_from_pdus(pdus)
        return (200, res)

    async def on_timestamp_to_event_request(self, origin: str, room_id: str, timestamp: int, direction: Direction) -> Tuple[int, Dict[str, Any]]:
        """When we receive a federated `/timestamp_to_event` request,
        handle all of the logic for validating and fetching the event.

        Args:
            origin: The server we received the event from
            room_id: Room to fetch the event from
            timestamp: The point in time (inclusive) we should navigate from in
                the given direction to find the closest event.
            direction: indicates whether we should navigate forward
                or backward from the given timestamp to find the closest event.

        Returns:
            Tuple indicating the response status code and dictionary response
            body including `event_id`.
        """
        async with self._server_linearizer.queue((origin, room_id)):
            (origin_host, _) = parse_server_name(origin)
            await self.check_server_matches_acl(origin_host, room_id)
            event_id = await self.store.get_event_id_for_timestamp(room_id, timestamp, direction)
            if event_id:
                event = await self.store.get_event(event_id, allow_none=False, allow_rejected=False)
                return (200, {'event_id': event_id, 'origin_server_ts': event.origin_server_ts})
        raise SynapseError(404, 'Unable to find event from %s in direction %s' % (timestamp, direction), errcode=Codes.NOT_FOUND)

    async def on_incoming_transaction(self, origin: str, transaction_id: str, destination: str, transaction_data: JsonDict) -> Tuple[int, JsonDict]:
        if not self._started_handling_of_staged_events:
            self._started_handling_of_staged_events = True
            self._handle_old_staged_events()
            self._clock.looping_call(self._handle_old_staged_events, 60 * 1000)
        request_time = self._clock.time_msec()
        transaction = Transaction(transaction_id=transaction_id, destination=destination, origin=origin, origin_server_ts=transaction_data.get('origin_server_ts'), pdus=transaction_data.get('pdus'), edus=transaction_data.get('edus'))
        if not transaction_id:
            raise Exception('Transaction missing transaction_id')
        logger.debug('[%s] Got transaction', transaction_id)
        if len(transaction.pdus) > 50 or len(transaction.edus) > 100:
            logger.info('Transaction PDU or EDU count too large. Returning 400')
            return (400, {})
        current_transaction = self._active_transactions.get(origin)
        if current_transaction and current_transaction != transaction_id:
            logger.warning('Received another txn %s from %s while still processing %s', transaction_id, origin, current_transaction)
            return (429, {'errcode': Codes.UNKNOWN, 'error': 'Too many concurrent transactions'})
        return await self._transaction_resp_cache.wrap((origin, transaction_id), self._on_incoming_transaction_inner, origin, transaction, request_time)

    async def _on_incoming_transaction_inner(self, origin: str, transaction: Transaction, request_time: int) -> Tuple[int, Dict[str, Any]]:
        assert origin not in self._active_transactions
        self._active_transactions[origin] = transaction.transaction_id
        try:
            result = await self._handle_incoming_transaction(origin, transaction, request_time)
            return result
        finally:
            del self._active_transactions[origin]

    async def _handle_incoming_transaction(self, origin: str, transaction: Transaction, request_time: int) -> Tuple[int, Dict[str, Any]]:
        """Process an incoming transaction and return the HTTP response

        Args:
            origin: the server making the request
            transaction: incoming transaction
            request_time: timestamp that the HTTP request arrived at

        Returns:
            HTTP response code and body
        """
        existing_response = await self.transaction_actions.have_responded(origin, transaction)
        if existing_response:
            logger.debug("[%s] We've already responded to this request", transaction.transaction_id)
            return existing_response
        logger.debug('[%s] Transaction is new', transaction.transaction_id)
        (pdu_results, _) = await make_deferred_yieldable(gather_results((run_in_background(self._handle_pdus_in_txn, origin, transaction, request_time), run_in_background(self._handle_edus_in_txn, origin, transaction)), consumeErrors=True).addErrback(unwrapFirstError))
        response = {'pdus': pdu_results}
        logger.debug('Returning: %s', str(response))
        await self.transaction_actions.set_response(origin, transaction, 200, response)
        return (200, response)

    async def _handle_pdus_in_txn(self, origin: str, transaction: Transaction, request_time: int) -> Dict[str, dict]:
        """Process the PDUs in a received transaction.

        Args:
            origin: the server making the request
            transaction: incoming transaction
            request_time: timestamp that the HTTP request arrived at

        Returns:
            A map from event ID of a processed PDU to any errors we should
            report back to the sending server.
        """
        received_pdus_counter.inc(len(transaction.pdus))
        (origin_host, _) = parse_server_name(origin)
        pdus_by_room: Dict[str, List[EventBase]] = {}
        newest_pdu_ts = 0
        for p in transaction.pdus:
            if 'unsigned' in p:
                unsigned = p['unsigned']
                if 'age' in unsigned:
                    p['age'] = unsigned['age']
            if 'age' in p:
                p['age_ts'] = request_time - int(p['age'])
                del p['age']
            possible_event_id = p.get('event_id', '<Unknown>')
            room_id = p.get('room_id')
            if not room_id:
                logger.info('Ignoring PDU as does not have a room_id. Event ID: %s', possible_event_id)
                continue
            try:
                room_version = await self.store.get_room_version(room_id)
            except NotFoundError:
                logger.info('Ignoring PDU for unknown room_id: %s', room_id)
                continue
            except UnsupportedRoomVersionError as e:
                logger.info('Ignoring PDU: %s', e)
                continue
            event = event_from_pdu_json(p, room_version)
            pdus_by_room.setdefault(room_id, []).append(event)
            if event.origin_server_ts > newest_pdu_ts:
                newest_pdu_ts = event.origin_server_ts
        pdu_results = {}

        async def process_pdus_for_room(room_id: str) -> None:
            with nested_logging_context(room_id):
                logger.debug('Processing PDUs for %s', room_id)
                try:
                    await self.check_server_matches_acl(origin_host, room_id)
                except AuthError as e:
                    logger.warning('Ignoring PDUs for room %s from banned server', room_id)
                    for pdu in pdus_by_room[room_id]:
                        event_id = pdu.event_id
                        pdu_results[event_id] = e.error_dict(self.hs.config)
                    return
                for pdu in pdus_by_room[room_id]:
                    pdu_results[pdu.event_id] = await process_pdu(pdu)

        async def process_pdu(pdu: EventBase) -> JsonDict:
            """
            Processes a pushed PDU sent to us via a `/send` transaction

            Returns:
                JsonDict representing a "PDU Processing Result" that will be bundled up
                with the other processed PDU's in the `/send` transaction and sent back
                to remote homeserver.
            """
            event_id = pdu.event_id
            with nested_logging_context(event_id):
                try:
                    await self._handle_received_pdu(origin, pdu)
                    return {}
                except FederationError as e:
                    logger.warning('Error handling PDU %s: %s', event_id, e)
                    return {'error': str(e)}
                except Exception as e:
                    f = failure.Failure()
                    logger.error('Failed to handle PDU %s', event_id, exc_info=(f.type, f.value, f.getTracebackObject()))
                    return {'error': str(e)}
        await concurrently_execute(process_pdus_for_room, pdus_by_room.keys(), TRANSACTION_CONCURRENCY_LIMIT)
        if newest_pdu_ts and origin in self._federation_metrics_domains:
            last_pdu_ts_metric.labels(server_name=origin).set(newest_pdu_ts / 1000)
        return pdu_results

    async def _handle_edus_in_txn(self, origin: str, transaction: Transaction) -> None:
        """Process the EDUs in a received transaction."""

        async def _process_edu(edu_dict: JsonDict) -> None:
            received_edus_counter.inc()
            edu = Edu(origin=origin, destination=self.server_name, edu_type=edu_dict['edu_type'], content=edu_dict['content'])
            await self.registry.on_edu(edu.edu_type, origin, edu.content)
        await concurrently_execute(_process_edu, transaction.edus, TRANSACTION_CONCURRENCY_LIMIT)

    async def on_room_state_request(self, origin: str, room_id: str, event_id: str) -> Tuple[int, JsonDict]:
        await self._event_auth_handler.assert_host_in_room(room_id, origin)
        (origin_host, _) = parse_server_name(origin)
        await self.check_server_matches_acl(origin_host, room_id)
        async with self._server_linearizer.queue((origin, room_id)):
            resp = await self._state_resp_cache.wrap((room_id, event_id), self._on_context_state_request_compute, room_id, event_id)
        return (200, resp)

    @trace
    @tag_args
    async def on_state_ids_request(self, origin: str, room_id: str, event_id: str) -> Tuple[int, JsonDict]:
        if not event_id:
            raise NotImplementedError('Specify an event')
        await self._event_auth_handler.assert_host_in_room(room_id, origin)
        (origin_host, _) = parse_server_name(origin)
        await self.check_server_matches_acl(origin_host, room_id)
        resp = await self._state_ids_resp_cache.wrap((room_id, event_id), self._on_state_ids_request_compute, room_id, event_id)
        return (200, resp)

    @trace
    @tag_args
    async def _on_state_ids_request_compute(self, room_id: str, event_id: str) -> JsonDict:
        state_ids = await self.handler.get_state_ids_for_pdu(room_id, event_id)
        auth_chain_ids = await self.store.get_auth_chain_ids(room_id, state_ids)
        return {'pdu_ids': state_ids, 'auth_chain_ids': list(auth_chain_ids)}

    async def _on_context_state_request_compute(self, room_id: str, event_id: str) -> Dict[str, list]:
        pdus: Collection[EventBase]
        event_ids = await self.handler.get_state_ids_for_pdu(room_id, event_id)
        pdus = await self.store.get_events_as_list(event_ids)
        auth_chain = await self.store.get_auth_chain(room_id, [pdu.event_id for pdu in pdus])
        return {'pdus': [pdu.get_pdu_json() for pdu in pdus], 'auth_chain': [pdu.get_pdu_json() for pdu in auth_chain]}

    async def on_pdu_request(self, origin: str, event_id: str) -> Tuple[int, Union[JsonDict, str]]:
        pdu = await self.handler.get_persisted_pdu(origin, event_id)
        if pdu:
            return (200, self._transaction_dict_from_pdus([pdu]))
        else:
            return (404, '')

    async def on_query_request(self, query_type: str, args: Dict[str, str]) -> Tuple[int, Dict[str, Any]]:
        received_queries_counter.labels(query_type).inc()
        resp = await self.registry.on_query(query_type, args)
        return (200, resp)

    async def on_make_join_request(self, origin: str, room_id: str, user_id: str, supported_versions: List[str]) -> Dict[str, Any]:
        (origin_host, _) = parse_server_name(origin)
        await self.check_server_matches_acl(origin_host, room_id)
        room_version = await self.store.get_room_version_id(room_id)
        if room_version not in supported_versions:
            logger.warning('Room version %s not in %s', room_version, supported_versions)
            raise IncompatibleRoomVersionError(room_version=room_version)
        await self._room_member_handler._join_rate_per_room_limiter.ratelimit(requester=None, key=room_id, update=False)
        pdu = await self.handler.on_make_join_request(origin, room_id, user_id)
        return {'event': pdu.get_templated_pdu_json(), 'room_version': room_version}

    async def on_invite_request(self, origin: str, content: JsonDict, room_version_id: str) -> Dict[str, Any]:
        room_version = KNOWN_ROOM_VERSIONS.get(room_version_id)
        if not room_version:
            raise SynapseError(400, 'Homeserver does not support this room version', Codes.UNSUPPORTED_ROOM_VERSION)
        pdu = event_from_pdu_json(content, room_version)
        (origin_host, _) = parse_server_name(origin)
        await self.check_server_matches_acl(origin_host, pdu.room_id)
        try:
            pdu = await self._check_sigs_and_hash(room_version, pdu)
        except InvalidEventSignatureError as e:
            errmsg = f'event id {pdu.event_id}: {e}'
            logger.warning('%s', errmsg)
            raise SynapseError(403, errmsg, Codes.FORBIDDEN)
        ret_pdu = await self.handler.on_invite_request(origin, pdu, room_version)
        time_now = self._clock.time_msec()
        return {'event': ret_pdu.get_pdu_json(time_now)}

    async def on_send_join_request(self, origin: str, content: JsonDict, room_id: str, caller_supports_partial_state: bool=False) -> Dict[str, Any]:
        set_tag(SynapseTags.SEND_JOIN_RESPONSE_IS_PARTIAL_STATE, caller_supports_partial_state)
        await self._room_member_handler._join_rate_per_room_limiter.ratelimit(requester=None, key=room_id, update=False)
        (event, context) = await self._on_send_membership_event(origin, content, Membership.JOIN, room_id)
        prev_state_ids = await context.get_prev_state_ids()
        state_event_ids: Collection[str]
        servers_in_room: Optional[Collection[str]]
        if caller_supports_partial_state:
            summary = await self.store.get_room_summary(room_id)
            state_event_ids = _get_event_ids_for_partial_state_join(event, prev_state_ids, summary)
            servers_in_room = await self.state.get_hosts_in_room_at_events(room_id, event_ids=event.prev_event_ids())
        else:
            state_event_ids = prev_state_ids.values()
            servers_in_room = None
        auth_chain_event_ids = await self.store.get_auth_chain_ids(room_id, state_event_ids)
        if caller_supports_partial_state:
            auth_chain_event_ids.difference_update(state_event_ids)
        auth_chain_events = await self.store.get_events_as_list(auth_chain_event_ids)
        state_events = await self.store.get_events_as_list(state_event_ids)
        time_now = self._clock.time_msec()
        event_json = event.get_pdu_json(time_now)
        resp = {'event': event_json, 'state': [p.get_pdu_json(time_now) for p in state_events], 'auth_chain': [p.get_pdu_json(time_now) for p in auth_chain_events], 'members_omitted': caller_supports_partial_state}
        if servers_in_room is not None:
            resp['servers_in_room'] = list(servers_in_room)
        return resp

    async def on_make_leave_request(self, origin: str, room_id: str, user_id: str) -> Dict[str, Any]:
        (origin_host, _) = parse_server_name(origin)
        await self.check_server_matches_acl(origin_host, room_id)
        pdu = await self.handler.on_make_leave_request(origin, room_id, user_id)
        room_version = await self.store.get_room_version_id(room_id)
        return {'event': pdu.get_templated_pdu_json(), 'room_version': room_version}

    async def on_send_leave_request(self, origin: str, content: JsonDict, room_id: str) -> dict:
        logger.debug('on_send_leave_request: content: %s', content)
        await self._on_send_membership_event(origin, content, Membership.LEAVE, room_id)
        return {}

    async def on_make_knock_request(self, origin: str, room_id: str, user_id: str, supported_versions: List[str]) -> JsonDict:
        """We've received a /make_knock/ request, so we create a partial knock
        event for the room and hand that back, along with the room version, to the knocking
        homeserver. We do *not* persist or process this event until the other server has
        signed it and sent it back.

        Args:
            origin: The (verified) server name of the requesting server.
            room_id: The room to create the knock event in.
            user_id: The user to create the knock for.
            supported_versions: The room versions supported by the requesting server.

        Returns:
            The partial knock event.
        """
        (origin_host, _) = parse_server_name(origin)
        if await self.store.is_partial_state_room(room_id):
            raise SynapseError(404, 'Unable to handle /make_knock right now; this server is not fully joined.', errcode=Codes.NOT_FOUND)
        await self.check_server_matches_acl(origin_host, room_id)
        room_version = await self.store.get_room_version(room_id)
        if room_version.identifier not in supported_versions:
            logger.warning('Room version %s not in %s', room_version.identifier, supported_versions)
            raise IncompatibleRoomVersionError(room_version=room_version.identifier)
        if not room_version.knock_join_rule:
            raise SynapseError(403, 'This room version does not support knocking', errcode=Codes.FORBIDDEN)
        pdu = await self.handler.on_make_knock_request(origin, room_id, user_id)
        return {'event': pdu.get_templated_pdu_json(), 'room_version': room_version.identifier}

    async def on_send_knock_request(self, origin: str, content: JsonDict, room_id: str) -> Dict[str, List[JsonDict]]:
        """
        We have received a knock event for a room. Verify and send the event into the room
        on the knocking homeserver's behalf. Then reply with some stripped state from the
        room for the knockee.

        Args:
            origin: The remote homeserver of the knocking user.
            content: The content of the request.
            room_id: The ID of the room to knock on.

        Returns:
            The stripped room state.
        """
        (_, context) = await self._on_send_membership_event(origin, content, Membership.KNOCK, room_id)
        stripped_room_state = await self.store.get_stripped_room_state_from_event_context(context, self._room_prejoin_state_types)
        return {'knock_room_state': stripped_room_state}

    async def _on_send_membership_event(self, origin: str, content: JsonDict, membership_type: str, room_id: str) -> Tuple[EventBase, EventContext]:
        """Handle an on_send_{join,leave,knock} request

        Does some preliminary validation before passing the request on to the
        federation handler.

        Args:
            origin: The (authenticated) requesting server
            content: The body of the send_* request - a complete membership event
            membership_type: The expected membership type (join or leave, depending
                on the endpoint)
            room_id: The room_id from the request, to be validated against the room_id
                in the event

        Returns:
            The event and context of the event after inserting it into the room graph.

        Raises:
            SynapseError if there is a problem with the request, including things like
               the room_id not matching or the event not being authorized.
        """
        assert_params_in_dict(content, ['room_id'])
        if content['room_id'] != room_id:
            raise SynapseError(400, 'Room ID in body does not match that in request path', Codes.BAD_JSON)
        room_version = await self.store.get_room_version(room_id)
        if await self.store.is_partial_state_room(room_id):
            logger.info(f"Rejecting /send_{membership_type} to %s because it's a partial state room", room_id)
            raise SynapseError(404, f'Unable to handle /send_{membership_type} right now; this server is not fully joined.', errcode=Codes.NOT_FOUND)
        if membership_type == Membership.KNOCK and (not room_version.knock_join_rule):
            raise SynapseError(403, 'This room version does not support knocking', errcode=Codes.FORBIDDEN)
        event = event_from_pdu_json(content, room_version)
        if event.type != EventTypes.Member or not event.is_state():
            raise SynapseError(400, 'Not an m.room.member event', Codes.BAD_JSON)
        if event.content.get('membership') != membership_type:
            raise SynapseError(400, 'Not a %s event' % membership_type, Codes.BAD_JSON)
        (origin_host, _) = parse_server_name(origin)
        await self.check_server_matches_acl(origin_host, event.room_id)
        logger.debug('_on_send_membership_event: pdu sigs: %s', event.signatures)
        if room_version.restricted_join_rule and event.membership == Membership.JOIN and (EventContentFields.AUTHORISING_USER in event.content):
            authorising_server = get_domain_from_id(event.content[EventContentFields.AUTHORISING_USER])
            if not self._is_mine_server_name(authorising_server):
                raise SynapseError(400, f'Cannot authorise membership event for {authorising_server}. We can only authorise requests from our own homeserver')
            event.signatures.update(compute_event_signature(room_version, event.get_pdu_json(), self.hs.hostname, self.hs.signing_key))
        try:
            event = await self._check_sigs_and_hash(room_version, event)
        except InvalidEventSignatureError as e:
            errmsg = f'event id {event.event_id}: {e}'
            logger.warning('%s', errmsg)
            raise SynapseError(403, errmsg, Codes.FORBIDDEN)
        try:
            return await self._federation_event_handler.on_send_membership_event(origin, event)
        except PartialStateConflictError:
            logger.info('Room %s was un-partial stated during `on_send_membership_event`, trying again.', room_id)
            return await self._federation_event_handler.on_send_membership_event(origin, event)

    async def on_event_auth(self, origin: str, room_id: str, event_id: str) -> Tuple[int, Dict[str, Any]]:
        async with self._server_linearizer.queue((origin, room_id)):
            await self._event_auth_handler.assert_host_in_room(room_id, origin)
            (origin_host, _) = parse_server_name(origin)
            await self.check_server_matches_acl(origin_host, room_id)
            time_now = self._clock.time_msec()
            auth_pdus = await self.handler.on_event_auth(event_id)
            res = {'auth_chain': [a.get_pdu_json(time_now) for a in auth_pdus]}
        return (200, res)

    async def on_query_client_keys(self, origin: str, content: Dict[str, str]) -> Tuple[int, Dict[str, Any]]:
        return await self.on_query_request('client_keys', content)

    async def on_query_user_devices(self, origin: str, user_id: str) -> Tuple[int, Dict[str, Any]]:
        keys = await self.device_handler.on_federation_query_user_devices(user_id)
        return (200, keys)

    @trace
    async def on_claim_client_keys(self, query: List[Tuple[str, str, str, int]], always_include_fallback_keys: bool) -> Dict[str, Any]:
        if any((not self.hs.is_mine(UserID.from_string(user_id)) for (user_id, _, _, _) in query)):
            raise SynapseError(400, 'User is not hosted on this homeserver')
        log_kv({'message': 'Claiming one time keys.', 'user, device pairs': query})
        results = await self._e2e_keys_handler.claim_local_one_time_keys(query, always_include_fallback_keys=always_include_fallback_keys)
        json_result: Dict[str, Dict[str, Dict[str, JsonDict]]] = {}
        for result in results:
            for (user_id, device_keys) in result.items():
                for (device_id, keys) in device_keys.items():
                    for (key_id, key) in keys.items():
                        json_result.setdefault(user_id, {}).setdefault(device_id, {})[key_id] = key
        logger.info('Claimed one-time-keys: %s', ','.join(('%s for %s:%s' % (key_id, user_id, device_id) for (user_id, user_keys) in json_result.items() for (device_id, device_keys) in user_keys.items() for (key_id, _) in device_keys.items())))
        return {'one_time_keys': json_result}

    async def on_get_missing_events(self, origin: str, room_id: str, earliest_events: List[str], latest_events: List[str], limit: int) -> Dict[str, list]:
        async with self._server_linearizer.queue((origin, room_id)):
            (origin_host, _) = parse_server_name(origin)
            await self.check_server_matches_acl(origin_host, room_id)
            logger.debug('on_get_missing_events: earliest_events: %r, latest_events: %r, limit: %d', earliest_events, latest_events, limit)
            missing_events = await self.handler.on_get_missing_events(origin, room_id, earliest_events, latest_events, limit)
            if len(missing_events) < 5:
                logger.debug('Returning %d events: %r', len(missing_events), missing_events)
            else:
                logger.debug('Returning %d events', len(missing_events))
            time_now = self._clock.time_msec()
        return {'events': [ev.get_pdu_json(time_now) for ev in missing_events]}

    async def on_openid_userinfo(self, token: str) -> Optional[str]:
        ts_now_ms = self._clock.time_msec()
        return await self.store.get_user_id_for_open_id_token(token, ts_now_ms)

    def _transaction_dict_from_pdus(self, pdu_list: List[EventBase]) -> JsonDict:
        if False:
            return 10
        'Returns a new Transaction containing the given PDUs suitable for\n        transmission.\n        '
        time_now = self._clock.time_msec()
        pdus = [p.get_pdu_json(time_now) for p in pdu_list]
        return Transaction(transaction_id='', origin=self.server_name, pdus=pdus, origin_server_ts=int(time_now), destination='').get_dict()

    async def _handle_received_pdu(self, origin: str, pdu: EventBase) -> None:
        """Process a PDU received in a federation /send/ transaction.

        If the event is invalid, then this method throws a FederationError.
        (The error will then be logged and sent back to the sender (which
        probably won't do anything with it), and other events in the
        transaction will be processed as normal).

        It is likely that we'll then receive other events which refer to
        this rejected_event in their prev_events, etc.  When that happens,
        we'll attempt to fetch the rejected event again, which will presumably
        fail, so those second-generation events will also get rejected.

        Eventually, we get to the point where there are more than 10 events
        between any new events and the original rejected event. Since we
        only try to backfill 10 events deep on received pdu, we then accept the
        new event, possibly introducing a discontinuity in the DAG, with new
        forward extremities, so normal service is approximately returned,
        until we try to backfill across the discontinuity.

        Args:
            origin: server which sent the pdu
            pdu: received pdu

        Raises: FederationError if the signatures / hash do not match, or
            if the event was unacceptable for any other reason (eg, too large,
            too many prev_events, couldn't find the prev_events)
        """
        room_version = await self.store.get_room_version(pdu.room_id)
        try:
            pdu = await self._check_sigs_and_hash(room_version, pdu)
        except InvalidEventSignatureError as e:
            logger.warning('event id %s: %s', pdu.event_id, e)
            raise FederationError('ERROR', 403, str(e), affected=pdu.event_id)
        if await self._spam_checker_module_callbacks.should_drop_federated_event(pdu):
            logger.warning('Unstaged federated event contains spam, dropping %s', pdu.event_id)
            return
        await self.store.insert_received_event_to_staging(origin, pdu)
        lock = await self.store.try_acquire_lock(_INBOUND_EVENT_HANDLING_LOCK_NAME, pdu.room_id)
        if lock:
            self._process_incoming_pdus_in_room_inner(pdu.room_id, room_version, lock, origin, pdu)

    async def _get_next_nonspam_staged_event_for_room(self, room_id: str, room_version: RoomVersion) -> Optional[Tuple[str, EventBase]]:
        """Fetch the first non-spam event from staging queue.

        Args:
            room_id: the room to fetch the first non-spam event in.
            room_version: the version of the room.

        Returns:
            The first non-spam event in that room.
        """
        while True:
            next = await self.store.get_next_staged_event_for_room(room_id, room_version)
            if next is None:
                return None
            (origin, event) = next
            if await self._spam_checker_module_callbacks.should_drop_federated_event(event):
                logger.warning('Staged federated event contains spam, dropping %s', event.event_id)
                continue
            return next

    @wrap_as_background_process('_process_incoming_pdus_in_room_inner')
    async def _process_incoming_pdus_in_room_inner(self, room_id: str, room_version: RoomVersion, lock: Lock, latest_origin: Optional[str]=None, latest_event: Optional[EventBase]=None) -> None:
        """Process events in the staging area for the given room.

        The latest_origin and latest_event args are the latest origin and event
        received (or None to simply pull the next event from the database).
        """
        if latest_event is not None and latest_origin is not None:
            result = await self.store.get_next_staged_event_id_for_room(room_id)
            if result is None:
                latest_origin = None
                latest_event = None
            else:
                (next_origin, next_event_id) = result
                if next_origin != latest_origin or next_event_id != latest_event.event_id:
                    latest_origin = None
                    latest_event = None
        if latest_origin is None or latest_event is None:
            next = await self.store.get_next_staged_event_for_room(room_id, room_version)
            if not next:
                await lock.release()
                return
            (origin, event) = next
        else:
            origin = latest_origin
            event = latest_event
        while True:
            async with lock:
                logger.info('handling received PDU in room %s: %s', room_id, event)
                try:
                    with nested_logging_context(event.event_id):
                        async with self._worker_lock_handler.acquire_read_write_lock(NEW_EVENT_DURING_PURGE_LOCK_NAME, room_id, write=False):
                            await self._federation_event_handler.on_receive_pdu(origin, event)
                except FederationError as e:
                    logger.warning('Error handling PDU %s: %s', event.event_id, e)
                except Exception:
                    f = failure.Failure()
                    logger.error('Failed to handle PDU %s', event.event_id, exc_info=(f.type, f.value, f.getTracebackObject()))
                received_ts = await self.store.remove_received_event_from_staging(origin, event.event_id)
                if received_ts is not None:
                    pdu_process_time.observe((self._clock.time_msec() - received_ts) / 1000)
            next = await self._get_next_nonspam_staged_event_for_room(room_id, room_version)
            if not next:
                break
            (origin, event) = next
            pruned = await self.store.prune_staged_events_in_room(room_id, room_version)
            if pruned:
                next = await self.store.get_next_staged_event_for_room(room_id, room_version)
                if not next:
                    break
                (origin, event) = next
            new_lock = await self.store.try_acquire_lock(_INBOUND_EVENT_HANDLING_LOCK_NAME, room_id)
            if not new_lock:
                return
            lock = new_lock

    async def exchange_third_party_invite(self, sender_user_id: str, target_user_id: str, room_id: str, signed: Dict) -> None:
        await self.handler.exchange_third_party_invite(sender_user_id, target_user_id, room_id, signed)

    async def on_exchange_third_party_invite_request(self, event_dict: Dict) -> None:
        await self.handler.on_exchange_third_party_invite_request(event_dict)

    async def check_server_matches_acl(self, server_name: str, room_id: str) -> None:
        """Check if the given server is allowed by the server ACLs in the room

        Args:
            server_name: name of server, *without any port part*
            room_id: ID of the room to check

        Raises:
            AuthError if the server does not match the ACL
        """
        server_acl_evaluator = await self._storage_controllers.state.get_server_acl_for_room(room_id)
        if server_acl_evaluator and (not server_acl_evaluator.server_matches_acl_event(server_name)):
            raise AuthError(code=403, msg='Server is banned from room')

class FederationHandlerRegistry:
    """Allows classes to register themselves as handlers for a given EDU or
    query type for incoming federation traffic.
    """

    def __init__(self, hs: 'HomeServer'):
        if False:
            while True:
                i = 10
        self.config = hs.config
        self.clock = hs.get_clock()
        self._instance_name = hs.get_instance_name()
        self._get_query_client = ReplicationGetQueryRestServlet.make_client(hs)
        self._send_edu = ReplicationFederationSendEduRestServlet.make_client(hs)
        self.edu_handlers: Dict[str, Callable[[str, dict], Awaitable[None]]] = {}
        self.query_handlers: Dict[str, Callable[[dict], Awaitable[JsonDict]]] = {}
        self._edu_type_to_instance: Dict[str, List[str]] = {}

    def register_edu_handler(self, edu_type: str, handler: Callable[[str, JsonDict], Awaitable[None]]) -> None:
        if False:
            while True:
                i = 10
        'Sets the handler callable that will be used to handle an incoming\n        federation EDU of the given type.\n\n        Args:\n            edu_type: The type of the incoming EDU to register handler for\n            handler: A callable invoked on incoming EDU\n                of the given type. The arguments are the origin server name and\n                the EDU contents.\n        '
        if edu_type in self.edu_handlers:
            raise KeyError('Already have an EDU handler for %s' % (edu_type,))
        logger.info('Registering federation EDU handler for %r', edu_type)
        self.edu_handlers[edu_type] = handler

    def register_query_handler(self, query_type: str, handler: Callable[[dict], Awaitable[JsonDict]]) -> None:
        if False:
            while True:
                i = 10
        'Sets the handler callable that will be used to handle an incoming\n        federation query of the given type.\n\n        Args:\n            query_type: Category name of the query, which should match\n                the string used by make_query.\n            handler: Invoked to handle\n                incoming queries of this type. The return will be yielded\n                on and the result used as the response to the query request.\n        '
        if query_type in self.query_handlers:
            raise KeyError('Already have a Query handler for %s' % (query_type,))
        logger.info('Registering federation query handler for %r', query_type)
        self.query_handlers[query_type] = handler

    def register_instances_for_edu(self, edu_type: str, instance_names: List[str]) -> None:
        if False:
            i = 10
            return i + 15
        'Register that the EDU handler is on multiple instances.'
        self._edu_type_to_instance[edu_type] = instance_names

    async def on_edu(self, edu_type: str, origin: str, content: dict) -> None:
        if not self.config.server.track_presence and edu_type == EduTypes.PRESENCE:
            return
        handler = self.edu_handlers.get(edu_type)
        if handler:
            with start_active_span_from_edu(content, 'handle_edu'):
                try:
                    await handler(origin, content)
                except SynapseError as e:
                    logger.info('Failed to handle edu %r: %r', edu_type, e)
                except Exception:
                    logger.exception('Failed to handle edu %r', edu_type)
            return
        instances = self._edu_type_to_instance.get(edu_type, ['master'])
        if self._instance_name not in instances:
            route_to = random.choice(instances)
            try:
                await self._send_edu(instance_name=route_to, edu_type=edu_type, origin=origin, content=content)
            except SynapseError as e:
                logger.info('Failed to handle edu %r: %r', edu_type, e)
            except Exception:
                logger.exception('Failed to handle edu %r', edu_type)
            return
        logger.warning('No handler registered for EDU type %s', edu_type)

    async def on_query(self, query_type: str, args: dict) -> JsonDict:
        handler = self.query_handlers.get(query_type)
        if handler:
            return await handler(args)
        if self._instance_name == 'master':
            return await self._get_query_client(query_type=query_type, args=args)
        logger.warning('No handler registered for query type %s', query_type)
        raise NotFoundError("No handler for Query type '%s'" % (query_type,))

def _get_event_ids_for_partial_state_join(join_event: EventBase, prev_state_ids: StateMap[str], summary: Mapping[str, MemberSummary]) -> Collection[str]:
    if False:
        for i in range(10):
            print('nop')
    'Calculate state to be returned in a partial_state send_join\n\n    Args:\n        join_event: the join event being send_joined\n        prev_state_ids: the event ids of the state before the join\n\n    Returns:\n        the event ids to be returned\n    '
    state_event_ids = {event_id for ((event_type, state_key), event_id) in prev_state_ids.items() if event_type != EventTypes.Member}
    current_membership_event_id = prev_state_ids.get((EventTypes.Member, join_event.state_key))
    if current_membership_event_id is not None:
        state_event_ids.add(current_membership_event_id)
    name_id = prev_state_ids.get((EventTypes.Name, ''))
    canonical_alias_id = prev_state_ids.get((EventTypes.CanonicalAlias, ''))
    if not name_id and (not canonical_alias_id):
        heroes = extract_heroes_from_room_summary(summary, join_event.state_key)
        for hero in heroes:
            membership_event_id = prev_state_ids.get((EventTypes.Member, hero))
            if membership_event_id:
                state_event_ids.add(membership_event_id)
    return state_event_ids