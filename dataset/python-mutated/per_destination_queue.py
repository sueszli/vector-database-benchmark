import datetime
import logging
from types import TracebackType
from typing import TYPE_CHECKING, Dict, Hashable, Iterable, List, Optional, Tuple, Type
import attr
from prometheus_client import Counter
from synapse.api.constants import EduTypes
from synapse.api.errors import FederationDeniedError, HttpResponseException, RequestSendFailed
from synapse.api.presence import UserPresenceState
from synapse.events import EventBase
from synapse.federation.units import Edu
from synapse.handlers.presence import format_user_presence_state
from synapse.logging import issue9533_logger
from synapse.logging.opentracing import SynapseTags, set_tag
from synapse.metrics import sent_transactions_counter
from synapse.metrics.background_process_metrics import run_as_background_process
from synapse.types import JsonDict, ReadReceipt
from synapse.util.retryutils import NotRetryingDestination, get_retry_limiter
from synapse.visibility import filter_events_for_server
if TYPE_CHECKING:
    import synapse.server
MAX_EDUS_PER_TRANSACTION = 100
logger = logging.getLogger(__name__)
sent_edus_counter = Counter('synapse_federation_client_sent_edus', 'Total number of EDUs successfully sent')
sent_edus_by_type = Counter('synapse_federation_client_sent_edus_by_type', 'Number of sent EDUs successfully sent, by event type', ['type'])
CATCHUP_RETRY_INTERVAL = 60 * 60 * 1000

class PerDestinationQueue:
    """
    Manages the per-destination transmission queues.

    Args:
        hs
        transaction_sender
        destination: the server_name of the destination that we are managing
            transmission for.
    """

    def __init__(self, hs: 'synapse.server.HomeServer', transaction_manager: 'synapse.federation.sender.TransactionManager', destination: str):
        if False:
            while True:
                i = 10
        self._server_name = hs.hostname
        self._clock = hs.get_clock()
        self._storage_controllers = hs.get_storage_controllers()
        self._store = hs.get_datastores().main
        self._transaction_manager = transaction_manager
        self._instance_name = hs.get_instance_name()
        self._federation_shard_config = hs.config.worker.federation_shard_config
        self._state = hs.get_state_handler()
        self._should_send_on_this_instance = True
        if not self._federation_shard_config.should_handle(self._instance_name, destination):
            logger.error('Create a per destination queue for %s on wrong worker', destination)
            self._should_send_on_this_instance = False
        self._destination = destination
        self.transmission_loop_running = False
        self._new_data_to_send = False
        self._catching_up: bool = True
        self._catchup_last_skipped: int = 0
        self._last_successful_stream_ordering: Optional[int] = None
        self._pending_pdus: List[EventBase] = []
        self._pending_edus: List[Edu] = []
        self._pending_edus_keyed: Dict[Tuple[str, Hashable], Edu] = {}
        self._pending_presence: Dict[str, UserPresenceState] = {}
        self._pending_receipt_edus: List[Dict[str, Dict[str, Dict[str, dict]]]] = []
        self._rrs_pending_flush = False
        self._last_device_stream_id = 0
        self._last_device_list_stream_id = 0

    def __str__(self) -> str:
        if False:
            return 10
        return 'PerDestinationQueue[%s]' % self._destination

    def pending_pdu_count(self) -> int:
        if False:
            while True:
                i = 10
        return len(self._pending_pdus)

    def pending_edu_count(self) -> int:
        if False:
            while True:
                i = 10
        return len(self._pending_edus) + len(self._pending_presence) + len(self._pending_edus_keyed)

    def send_pdu(self, pdu: EventBase) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Add a PDU to the queue, and start the transmission loop if necessary\n\n        Args:\n            pdu: pdu to send\n        '
        if not self._catching_up or self._last_successful_stream_ordering is None:
            self._pending_pdus.append(pdu)
        else:
            assert pdu.internal_metadata.stream_ordering
            self._catchup_last_skipped = pdu.internal_metadata.stream_ordering
        self.attempt_new_transaction()

    def send_presence(self, states: Iterable[UserPresenceState], start_loop: bool=True) -> None:
        if False:
            while True:
                i = 10
        'Add presence updates to the queue.\n\n        Args:\n            states: Presence updates to send\n            start_loop: Whether to start the transmission loop if not already\n                running.\n\n        Args:\n            states: presence to send\n        '
        self._pending_presence.update({state.user_id: state for state in states})
        self._new_data_to_send = True
        if start_loop:
            self.attempt_new_transaction()

    def queue_read_receipt(self, receipt: ReadReceipt) -> None:
        if False:
            print('Hello World!')
        "Add a RR to the list to be sent. Doesn't start the transmission loop yet\n        (see flush_read_receipts_for_room)\n\n        Args:\n            receipt: receipt to be queued\n        "
        serialized_receipt: JsonDict = {'event_ids': receipt.event_ids, 'data': receipt.data}
        if receipt.thread_id is not None:
            serialized_receipt['data']['thread_id'] = receipt.thread_id
        for edu in self._pending_receipt_edus:
            receipt_content = edu.setdefault(receipt.room_id, {}).setdefault(receipt.receipt_type, {})
            if receipt.user_id not in receipt_content or receipt_content[receipt.user_id].get('thread_id') == receipt.thread_id:
                receipt_content[receipt.user_id] = serialized_receipt
                break
        else:
            self._pending_receipt_edus.append({receipt.room_id: {receipt.receipt_type: {receipt.user_id: serialized_receipt}}})

    def flush_read_receipts_for_room(self, room_id: str) -> None:
        if False:
            print('Hello World!')
        for edu in self._pending_receipt_edus:
            if room_id in edu:
                self._rrs_pending_flush = True
                self.attempt_new_transaction()
                break

    def send_keyed_edu(self, edu: Edu, key: Hashable) -> None:
        if False:
            i = 10
            return i + 15
        self._pending_edus_keyed[edu.edu_type, key] = edu
        self.attempt_new_transaction()

    def send_edu(self, edu: Edu) -> None:
        if False:
            print('Hello World!')
        self._pending_edus.append(edu)
        self.attempt_new_transaction()

    def mark_new_data(self) -> None:
        if False:
            print('Hello World!')
        'Marks that the destination has new data to send, without starting a\n        new transaction.\n\n        If a transaction loop is already in progress then a new transaction will\n        be attempted when the current one finishes.\n        '
        self._new_data_to_send = True

    def attempt_new_transaction(self) -> None:
        if False:
            print('Hello World!')
        'Try to start a new transaction to this destination\n\n        If there is already a transaction in progress to this destination,\n        returns immediately. Otherwise kicks off the process of sending a\n        transaction in the background.\n        '
        self._new_data_to_send = True
        if self.transmission_loop_running:
            logger.debug('TX [%s] Transaction already in progress', self._destination)
            return
        if not self._should_send_on_this_instance:
            logger.error('Trying to start a transaction to %s on wrong worker', self._destination)
            return
        logger.debug('TX [%s] Starting transaction loop', self._destination)
        run_as_background_process('federation_transaction_transmission_loop', self._transaction_transmission_loop)

    async def _transaction_transmission_loop(self) -> None:
        pending_pdus: List[EventBase] = []
        try:
            self.transmission_loop_running = True
            await get_retry_limiter(self._destination, self._clock, self._store)
            if self._catching_up:
                await self._catch_up_transmission_loop()
                if self._catching_up:
                    return
            pending_pdus = []
            while True:
                self._new_data_to_send = False
                async with _TransactionQueueManager(self) as (pending_pdus, pending_edus):
                    if not pending_pdus and (not pending_edus):
                        logger.debug('TX [%s] Nothing to send', self._destination)
                        if self._new_data_to_send:
                            continue
                        else:
                            return
                    if pending_pdus:
                        logger.debug('TX [%s] len(pending_pdus_by_dest[dest]) = %d', self._destination, len(pending_pdus))
                    await self._transaction_manager.send_new_transaction(self._destination, pending_pdus, pending_edus)
                    sent_transactions_counter.inc()
                    sent_edus_counter.inc(len(pending_edus))
                    for edu in pending_edus:
                        sent_edus_by_type.labels(edu.edu_type).inc()
        except NotRetryingDestination as e:
            logger.debug('TX [%s] not ready for retry yet (next retry at %s) - dropping transaction for now', self._destination, datetime.datetime.fromtimestamp((e.retry_last_ts + e.retry_interval) / 1000.0))
            if e.retry_interval > CATCHUP_RETRY_INTERVAL:
                self._pending_edus = []
                self._pending_edus_keyed = {}
                self._pending_presence = {}
                self._pending_receipt_edus = []
                self._start_catching_up()
        except FederationDeniedError as e:
            logger.info(e)
        except HttpResponseException as e:
            logger.warning('TX [%s] Received %d response to transaction: %s', self._destination, e.code, e)
        except RequestSendFailed as e:
            logger.warning('TX [%s] Failed to send transaction: %s', self._destination, e)
            for p in pending_pdus:
                logger.info('Failed to send event %s to %s', p.event_id, self._destination)
        except Exception:
            logger.exception('TX [%s] Failed to send transaction', self._destination)
            for p in pending_pdus:
                logger.info('Failed to send event %s to %s', p.event_id, self._destination)
        finally:
            self.transmission_loop_running = False

    async def _catch_up_transmission_loop(self) -> None:
        first_catch_up_check = self._last_successful_stream_ordering is None
        if first_catch_up_check:
            self._last_successful_stream_ordering = await self._store.get_destination_last_successful_stream_ordering(self._destination)
        _tmp_last_successful_stream_ordering = self._last_successful_stream_ordering
        if _tmp_last_successful_stream_ordering is None:
            self._catching_up = False
            return
        last_successful_stream_ordering: int = _tmp_last_successful_stream_ordering
        while True:
            event_ids = await self._store.get_catch_up_room_event_ids(self._destination, last_successful_stream_ordering)
            if not event_ids:
                if self._catchup_last_skipped > last_successful_stream_ordering:
                    continue
                self._catching_up = False
                break
            if first_catch_up_check:
                self._start_catching_up()
            catchup_pdus = await self._store.get_events_as_list(event_ids)
            if not catchup_pdus:
                raise AssertionError('No events retrieved when we asked for %r. This should not happen.' % event_ids)
            logger.info('Catching up destination %s with %d PDUs', self._destination, len(catchup_pdus))
            for pdu in catchup_pdus:
                extrems = await self._store.get_prev_events_for_room(pdu.room_id)
                if pdu.event_id in extrems:
                    room_catchup_pdus = [pdu]
                elif await self._store.is_partial_state_room(pdu.room_id):
                    room_catchup_pdus = [pdu]
                else:
                    extrem_events = await self._store.get_events_as_list(extrems)
                    new_pdus = []
                    for p in extrem_events:
                        assert p.internal_metadata.stream_ordering
                        if p.internal_metadata.stream_ordering < last_successful_stream_ordering:
                            continue
                        new_pdus.append(p)
                    new_pdus = await filter_events_for_server(self._storage_controllers, self._destination, self._server_name, new_pdus, redact=False, filter_out_erased_senders=True, filter_out_remote_partial_state_events=True)
                    if new_pdus:
                        room_catchup_pdus = new_pdus
                    else:
                        room_catchup_pdus = [pdu]
                logger.info('Catching up rooms to %s: %r', self._destination, pdu.room_id)
                await self._transaction_manager.send_new_transaction(self._destination, room_catchup_pdus, [])
                sent_transactions_counter.inc()
                assert pdu.internal_metadata.stream_ordering
                last_successful_stream_ordering = pdu.internal_metadata.stream_ordering
                self._last_successful_stream_ordering = last_successful_stream_ordering
                await self._store.set_destination_last_successful_stream_ordering(self._destination, last_successful_stream_ordering)

    def _get_receipt_edus(self, force_flush: bool, limit: int) -> Iterable[Edu]:
        if False:
            print('Hello World!')
        if not self._pending_receipt_edus:
            return
        if not force_flush and (not self._rrs_pending_flush):
            return
        for content in self._pending_receipt_edus[:limit]:
            yield Edu(origin=self._server_name, destination=self._destination, edu_type=EduTypes.RECEIPT, content=content)
        self._pending_receipt_edus = self._pending_receipt_edus[limit:]
        if not self._pending_receipt_edus:
            self._rrs_pending_flush = False

    def _pop_pending_edus(self, limit: int) -> List[Edu]:
        if False:
            i = 10
            return i + 15
        pending_edus = self._pending_edus
        (pending_edus, self._pending_edus) = (pending_edus[:limit], pending_edus[limit:])
        return pending_edus

    async def _get_device_update_edus(self, limit: int) -> Tuple[List[Edu], int]:
        last_device_list = self._last_device_list_stream_id
        (now_stream_id, results) = await self._store.get_device_updates_by_remote(self._destination, last_device_list, limit=limit)
        edus = [Edu(origin=self._server_name, destination=self._destination, edu_type=edu_type, content=content) for (edu_type, content) in results]
        assert len(edus) <= limit, 'get_device_updates_by_remote returned too many EDUs'
        return (edus, now_stream_id)

    async def _get_to_device_message_edus(self, limit: int) -> Tuple[List[Edu], int]:
        last_device_stream_id = self._last_device_stream_id
        to_device_stream_id = self._store.get_to_device_stream_token()
        (contents, stream_id) = await self._store.get_new_device_msgs_for_remote(self._destination, last_device_stream_id, to_device_stream_id, limit)
        for content in contents:
            message_id = content.get('message_id')
            if not message_id:
                continue
            set_tag(SynapseTags.TO_DEVICE_EDU_ID, message_id)
        edus = [Edu(origin=self._server_name, destination=self._destination, edu_type=EduTypes.DIRECT_TO_DEVICE, content=content) for content in contents]
        if edus:
            issue9533_logger.debug('Sending %i to-device messages to %s, up to stream id %i', len(edus), self._destination, stream_id)
        return (edus, stream_id)

    def _start_catching_up(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Marks this destination as being in catch-up mode.\n\n        This throws away the PDU queue.\n        '
        self._catching_up = True
        self._pending_pdus = []

@attr.s(slots=True, auto_attribs=True)
class _TransactionQueueManager:
    """A helper async context manager for pulling stuff off the queues and
    tracking what was last successfully sent, etc.
    """
    queue: PerDestinationQueue
    _device_stream_id: Optional[int] = None
    _device_list_id: Optional[int] = None
    _last_stream_ordering: Optional[int] = None
    _pdus: List[EventBase] = attr.Factory(list)

    async def __aenter__(self) -> Tuple[List[EventBase], List[Edu]]:
        pending_edus = []
        if self.queue._pending_presence:
            pending_edus.append(Edu(origin=self.queue._server_name, destination=self.queue._destination, edu_type=EduTypes.PRESENCE, content={'push': [format_user_presence_state(presence, self.queue._clock.time_msec()) for presence in self.queue._pending_presence.values()]}))
            self.queue._pending_presence = {}
        pending_edus.extend(self.queue._get_receipt_edus(force_flush=False, limit=5))
        edu_limit = MAX_EDUS_PER_TRANSACTION - len(pending_edus)
        (to_device_edus, device_stream_id) = await self.queue._get_to_device_message_edus(edu_limit - 10)
        if to_device_edus:
            self._device_stream_id = device_stream_id
        else:
            self.queue._last_device_stream_id = device_stream_id
        pending_edus.extend(to_device_edus)
        edu_limit -= len(to_device_edus)
        (device_update_edus, dev_list_id) = await self.queue._get_device_update_edus(edu_limit)
        if device_update_edus:
            self._device_list_id = dev_list_id
        else:
            self.queue._last_device_list_stream_id = dev_list_id
        pending_edus.extend(device_update_edus)
        edu_limit -= len(device_update_edus)
        other_edus = self.queue._pop_pending_edus(edu_limit)
        pending_edus.extend(other_edus)
        edu_limit -= len(other_edus)
        while edu_limit > 0 and self.queue._pending_edus_keyed:
            (_, val) = self.queue._pending_edus_keyed.popitem()
            pending_edus.append(val)
            edu_limit -= 1
        self._pdus = self.queue._pending_pdus[:50]
        if not self._pdus and (not pending_edus):
            return ([], [])
        if edu_limit:
            pending_edus.extend(self.queue._get_receipt_edus(force_flush=True, limit=edu_limit))
        if self._pdus:
            self._last_stream_ordering = self._pdus[-1].internal_metadata.stream_ordering
            assert self._last_stream_ordering
        return (self._pdus, pending_edus)

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]) -> None:
        if exc_type is not None:
            return
        if self._pdus:
            self.queue._pending_pdus = self.queue._pending_pdus[len(self._pdus):]
        if self._device_stream_id:
            await self.queue._store.delete_device_msgs_for_remote(self.queue._destination, self._device_stream_id)
            self.queue._last_device_stream_id = self._device_stream_id
        if self._device_list_id:
            logger.info('Marking as sent %r %r', self.queue._destination, self._device_list_id)
            await self.queue._store.mark_as_sent_devices_by_remote(self.queue._destination, self._device_list_id)
            self.queue._last_device_list_stream_id = self._device_list_id
        if self._last_stream_ordering:
            await self.queue._store.set_destination_last_successful_stream_ordering(self.queue._destination, self._last_stream_ordering)