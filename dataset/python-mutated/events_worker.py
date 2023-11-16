import logging
import threading
import weakref
from enum import Enum, auto
from itertools import chain
from typing import TYPE_CHECKING, Any, Collection, Dict, Iterable, List, Mapping, MutableMapping, Optional, Set, Tuple, cast, overload
import attr
from prometheus_client import Gauge
from typing_extensions import Literal
from twisted.internet import defer
from synapse.api.constants import Direction, EventTypes
from synapse.api.errors import NotFoundError, SynapseError
from synapse.api.room_versions import KNOWN_ROOM_VERSIONS, EventFormatVersions, RoomVersion, RoomVersions
from synapse.events import EventBase, make_event_from_dict
from synapse.events.snapshot import EventContext
from synapse.events.utils import prune_event
from synapse.logging.context import PreserveLoggingContext, current_context, make_deferred_yieldable
from synapse.logging.opentracing import start_active_span, tag_args, trace
from synapse.metrics.background_process_metrics import run_as_background_process, wrap_as_background_process
from synapse.replication.tcp.streams import BackfillStream, UnPartialStatedEventStream
from synapse.replication.tcp.streams.events import EventsStream
from synapse.replication.tcp.streams.partial_state import UnPartialStatedEventStreamRow
from synapse.storage._base import SQLBaseStore, db_to_json, make_in_list_sql_clause
from synapse.storage.database import DatabasePool, LoggingDatabaseConnection, LoggingTransaction
from synapse.storage.engines import PostgresEngine
from synapse.storage.types import Cursor
from synapse.storage.util.id_generators import AbstractStreamIdGenerator, MultiWriterIdGenerator, StreamIdGenerator
from synapse.storage.util.sequence import build_sequence_generator
from synapse.types import JsonDict, get_domain_from_id
from synapse.types.state import StateFilter
from synapse.util import unwrapFirstError
from synapse.util.async_helpers import ObservableDeferred, delay_cancellation
from synapse.util.caches.descriptors import cached, cachedList
from synapse.util.caches.lrucache import AsyncLruCache
from synapse.util.caches.stream_change_cache import StreamChangeCache
from synapse.util.cancellation import cancellable
from synapse.util.iterutils import batch_iter
from synapse.util.metrics import Measure
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)
EVENT_QUEUE_THREADS = 3
EVENT_QUEUE_ITERATIONS = 3
EVENT_QUEUE_TIMEOUT_S = 0.1
event_fetch_ongoing_gauge = Gauge('synapse_event_fetch_ongoing', 'The number of event fetchers that are running')

class InvalidEventError(Exception):
    """The event retrieved from the database is invalid and cannot be used."""

@attr.s(slots=True, auto_attribs=True)
class EventCacheEntry:
    event: EventBase
    redacted_event: Optional[EventBase]

@attr.s(slots=True, frozen=True, auto_attribs=True)
class _EventRow:
    """
    An event, as pulled from the database.

    Properties:
        event_id: The event ID of the event.

        stream_ordering: stream ordering for this event

        json: json-encoded event structure

        internal_metadata: json-encoded internal metadata dict

        format_version: The format of the event. Hopefully one of EventFormatVersions.
            'None' means the event predates EventFormatVersions (so the event is format V1).

        room_version_id: The version of the room which contains the event. Hopefully
            one of RoomVersions.

           Due to historical reasons, there may be a few events in the database which
           do not have an associated room; in this case None will be returned here.

        rejected_reason: if the event was rejected, the reason why.

        redactions: a list of event-ids which (claim to) redact this event.

        outlier: True if this event is an outlier.
    """
    event_id: str
    stream_ordering: int
    json: str
    internal_metadata: str
    format_version: Optional[int]
    room_version_id: Optional[str]
    rejected_reason: Optional[str]
    redactions: List[str]
    outlier: bool

class EventRedactBehaviour(Enum):
    """
    What to do when retrieving a redacted event from the database.
    """
    as_is = auto()
    redact = auto()
    block = auto()

class EventsWorkerStore(SQLBaseStore):
    USE_DEDICATED_DB_THREADS_FOR_EVENT_FETCHING = True

    def __init__(self, database: DatabasePool, db_conn: LoggingDatabaseConnection, hs: 'HomeServer'):
        if False:
            i = 10
            return i + 15
        super().__init__(database, db_conn, hs)
        self._stream_id_gen: AbstractStreamIdGenerator
        self._backfill_id_gen: AbstractStreamIdGenerator
        if isinstance(database.engine, PostgresEngine):
            self._stream_id_gen = MultiWriterIdGenerator(db_conn=db_conn, db=database, notifier=hs.get_replication_notifier(), stream_name='events', instance_name=hs.get_instance_name(), tables=[('events', 'instance_name', 'stream_ordering')], sequence_name='events_stream_seq', writers=hs.config.worker.writers.events)
            self._backfill_id_gen = MultiWriterIdGenerator(db_conn=db_conn, db=database, notifier=hs.get_replication_notifier(), stream_name='backfill', instance_name=hs.get_instance_name(), tables=[('events', 'instance_name', 'stream_ordering')], sequence_name='events_backfill_stream_seq', positive=False, writers=hs.config.worker.writers.events)
        else:
            self._stream_id_gen = StreamIdGenerator(db_conn, hs.get_replication_notifier(), 'events', 'stream_ordering', is_writer=hs.get_instance_name() in hs.config.worker.writers.events)
            self._backfill_id_gen = StreamIdGenerator(db_conn, hs.get_replication_notifier(), 'events', 'stream_ordering', step=-1, extra_tables=[('ex_outlier_stream', 'event_stream_ordering')], is_writer=hs.get_instance_name() in hs.config.worker.writers.events)
        events_max = self._stream_id_gen.get_current_token()
        (curr_state_delta_prefill, min_curr_state_delta_id) = self.db_pool.get_cache_dict(db_conn, 'current_state_delta_stream', entity_column='room_id', stream_column='stream_id', max_value=events_max, limit=1000)
        self._curr_state_delta_stream_cache: StreamChangeCache = StreamChangeCache('_curr_state_delta_stream_cache', min_curr_state_delta_id, prefilled_cache=curr_state_delta_prefill)
        if hs.config.worker.run_background_tasks:
            self._clock.looping_call(self._cleanup_old_transaction_ids, 5 * 60 * 1000)
        self._get_event_cache: AsyncLruCache[Tuple[str], EventCacheEntry] = AsyncLruCache(cache_name='*getEvent*', max_size=hs.config.caches.event_cache_size)
        self._current_event_fetches: Dict[str, ObservableDeferred[Dict[str, EventCacheEntry]]] = {}
        self._event_ref: MutableMapping[str, EventBase] = weakref.WeakValueDictionary()
        self._event_fetch_lock = threading.Condition()
        self._event_fetch_list: List[Tuple[Iterable[str], 'defer.Deferred[Dict[str, _EventRow]]']] = []
        self._event_fetch_ongoing = 0
        event_fetch_ongoing_gauge.set(self._event_fetch_ongoing)

        def get_chain_id_txn(txn: Cursor) -> int:
            if False:
                for i in range(10):
                    print('nop')
            txn.execute('SELECT COALESCE(max(chain_id), 0) FROM event_auth_chains')
            return cast(Tuple[int], txn.fetchone())[0]
        self.event_chain_id_gen = build_sequence_generator(db_conn, database.engine, get_chain_id_txn, 'event_auth_chain_id', table='event_auth_chains', id_column='chain_id')
        self._un_partial_stated_events_stream_id_gen: AbstractStreamIdGenerator
        if isinstance(database.engine, PostgresEngine):
            self._un_partial_stated_events_stream_id_gen = MultiWriterIdGenerator(db_conn=db_conn, db=database, notifier=hs.get_replication_notifier(), stream_name='un_partial_stated_event_stream', instance_name=hs.get_instance_name(), tables=[('un_partial_stated_event_stream', 'instance_name', 'stream_id')], sequence_name='un_partial_stated_event_stream_sequence', writers=['master'])
        else:
            self._un_partial_stated_events_stream_id_gen = StreamIdGenerator(db_conn, hs.get_replication_notifier(), 'un_partial_stated_event_stream', 'stream_id')

    def get_un_partial_stated_events_token(self, instance_name: str) -> int:
        if False:
            while True:
                i = 10
        return self._un_partial_stated_events_stream_id_gen.get_current_token_for_writer(instance_name)

    async def get_un_partial_stated_events_from_stream(self, instance_name: str, last_id: int, current_id: int, limit: int) -> Tuple[List[Tuple[int, Tuple[str, bool]]], int, bool]:
        """Get updates for the un-partial-stated events replication stream.

        Args:
            instance_name: The writer we want to fetch updates from. Unused
                here since there is only ever one writer.
            last_id: The token to fetch updates from. Exclusive.
            current_id: The token to fetch updates up to. Inclusive.
            limit: The requested limit for the number of rows to return. The
                function may return more or fewer rows.

        Returns:
            A tuple consisting of: the updates, a token to use to fetch
            subsequent updates, and whether we returned fewer rows than exists
            between the requested tokens due to the limit.

            The token returned can be used in a subsequent call to this
            function to get further updatees.

            The updates are a list of 2-tuples of stream ID and the row data
        """
        if last_id == current_id:
            return ([], current_id, False)

        def get_un_partial_stated_events_from_stream_txn(txn: LoggingTransaction) -> Tuple[List[Tuple[int, Tuple[str, bool]]], int, bool]:
            if False:
                i = 10
                return i + 15
            sql = '\n                SELECT stream_id, event_id, rejection_status_changed\n                FROM un_partial_stated_event_stream\n                WHERE ? < stream_id AND stream_id <= ? AND instance_name = ?\n                ORDER BY stream_id ASC\n                LIMIT ?\n            '
            txn.execute(sql, (last_id, current_id, instance_name, limit))
            updates = [(row[0], (row[1], bool(row[2]))) for row in txn]
            limited = False
            upto_token = current_id
            if len(updates) >= limit:
                upto_token = updates[-1][0]
                limited = True
            return (updates, upto_token, limited)
        return await self.db_pool.runInteraction('get_un_partial_stated_events_from_stream', get_un_partial_stated_events_from_stream_txn)

    def process_replication_rows(self, stream_name: str, instance_name: str, token: int, rows: Iterable[Any]) -> None:
        if False:
            i = 10
            return i + 15
        if stream_name == UnPartialStatedEventStream.NAME:
            for row in rows:
                assert isinstance(row, UnPartialStatedEventStreamRow)
                self.is_partial_state_event.invalidate((row.event_id,))
                if row.rejection_status_changed:
                    self._invalidate_local_get_event_cache(row.event_id)
        super().process_replication_rows(stream_name, instance_name, token, rows)

    def process_replication_position(self, stream_name: str, instance_name: str, token: int) -> None:
        if False:
            while True:
                i = 10
        if stream_name == EventsStream.NAME:
            self._stream_id_gen.advance(instance_name, token)
        elif stream_name == BackfillStream.NAME:
            self._backfill_id_gen.advance(instance_name, -token)
        elif stream_name == UnPartialStatedEventStream.NAME:
            self._un_partial_stated_events_stream_id_gen.advance(instance_name, token)
        super().process_replication_position(stream_name, instance_name, token)

    async def have_censored_event(self, event_id: str) -> bool:
        """Check if an event has been censored, i.e. if the content of the event has been erased
        from the database due to a redaction.

        Args:
            event_id: The event ID that was redacted.

        Returns:
            True if the event has been censored, False otherwise.
        """
        censored_redactions_list = await self.db_pool.simple_select_onecol(table='redactions', keyvalues={'redacts': event_id}, retcol='have_censored', desc='get_have_censored')
        return any(censored_redactions_list)

    @overload
    async def get_event(self, event_id: str, redact_behaviour: EventRedactBehaviour=EventRedactBehaviour.redact, get_prev_content: bool=..., allow_rejected: bool=..., allow_none: Literal[False]=..., check_room_id: Optional[str]=...) -> EventBase:
        ...

    @overload
    async def get_event(self, event_id: str, redact_behaviour: EventRedactBehaviour=EventRedactBehaviour.redact, get_prev_content: bool=..., allow_rejected: bool=..., allow_none: Literal[True]=..., check_room_id: Optional[str]=...) -> Optional[EventBase]:
        ...

    @cancellable
    async def get_event(self, event_id: str, redact_behaviour: EventRedactBehaviour=EventRedactBehaviour.redact, get_prev_content: bool=False, allow_rejected: bool=False, allow_none: bool=False, check_room_id: Optional[str]=None) -> Optional[EventBase]:
        """Get an event from the database by event_id.

        Args:
            event_id: The event_id of the event to fetch

            redact_behaviour: Determine what to do with a redacted event. Possible values:
                * as_is - Return the full event body with no redacted content
                * redact - Return the event but with a redacted body
                * block - Do not return redacted events (behave as per allow_none
                    if the event is redacted)

            get_prev_content: If True and event is a state event,
                include the previous states content in the unsigned field.

            allow_rejected: If True, return rejected events. Otherwise,
                behave as per allow_none.

            allow_none: If True, return None if no event found, if
                False throw a NotFoundError

            check_room_id: if not None, check the room of the found event.
                If there is a mismatch, behave as per allow_none.

        Returns:
            The event, or None if the event was not found and allow_none is `True`.
        """
        if not isinstance(event_id, str):
            raise TypeError('Invalid event event_id %r' % (event_id,))
        events = await self.get_events_as_list([event_id], redact_behaviour=redact_behaviour, get_prev_content=get_prev_content, allow_rejected=allow_rejected)
        event = events[0] if events else None
        if event is not None and check_room_id is not None:
            if event.room_id != check_room_id:
                event = None
        if event is None and (not allow_none):
            raise NotFoundError('Could not find event %s' % (event_id,))
        return event

    async def get_events(self, event_ids: Collection[str], redact_behaviour: EventRedactBehaviour=EventRedactBehaviour.redact, get_prev_content: bool=False, allow_rejected: bool=False) -> Dict[str, EventBase]:
        """Get events from the database

        Args:
            event_ids: The event_ids of the events to fetch

            redact_behaviour: Determine what to do with a redacted event. Possible
                values:
                * as_is - Return the full event body with no redacted content
                * redact - Return the event but with a redacted body
                * block - Do not return redacted events (omit them from the response)

            get_prev_content: If True and event is a state event,
                include the previous states content in the unsigned field.

            allow_rejected: If True, return rejected events. Otherwise,
                omits rejected events from the response.

        Returns:
            A mapping from event_id to event.
        """
        events = await self.get_events_as_list(event_ids, redact_behaviour=redact_behaviour, get_prev_content=get_prev_content, allow_rejected=allow_rejected)
        return {e.event_id: e for e in events}

    @trace
    @tag_args
    @cancellable
    async def get_events_as_list(self, event_ids: Collection[str], redact_behaviour: EventRedactBehaviour=EventRedactBehaviour.redact, get_prev_content: bool=False, allow_rejected: bool=False) -> List[EventBase]:
        """Get events from the database and return in a list in the same order
        as given by `event_ids` arg.

        Unknown events will be omitted from the response.

        Args:
            event_ids: The event_ids of the events to fetch

            redact_behaviour: Determine what to do with a redacted event. Possible values:
                * as_is - Return the full event body with no redacted content
                * redact - Return the event but with a redacted body
                * block - Do not return redacted events (omit them from the response)

            get_prev_content: If True and event is a state event,
                include the previous states content in the unsigned field.

            allow_rejected: If True, return rejected events. Otherwise,
                omits rejected events from the response.

        Returns:
            List of events fetched from the database. The events are in the same
            order as `event_ids` arg.

            Note that the returned list may be smaller than the list of event
            IDs if not all events could be fetched.
        """
        if not event_ids:
            return []
        event_entry_map = await self.get_unredacted_events_from_cache_or_db(set(event_ids), allow_rejected=allow_rejected)
        events = []
        for event_id in event_ids:
            entry = event_entry_map.get(event_id, None)
            if not entry:
                continue
            if not allow_rejected:
                assert not entry.event.rejected_reason, 'rejected event returned from _get_events_from_cache_or_db despite allow_rejected=False'
            if not allow_rejected and entry.event.type == EventTypes.Redaction:
                if entry.event.redacts is None:
                    logger.debug("Withholding redaction event %s as we don't have redacts key", event_id)
                    continue
                redacted_event_id = entry.event.redacts
                event_map = await self.get_unredacted_events_from_cache_or_db([redacted_event_id])
                original_event_entry = event_map.get(redacted_event_id)
                if not original_event_entry:
                    logger.debug("Withholding redaction event %s since we don't (yet) have the original %s", event_id, redacted_event_id)
                    continue
                original_event = original_event_entry.event
                if original_event.type == EventTypes.Create:
                    logger.info('Withholding redaction %s of create event %s', event_id, redacted_event_id)
                    continue
                if original_event.room_id != entry.event.room_id:
                    logger.info('Withholding redaction %s of event %s from a different room', event_id, redacted_event_id)
                    continue
                if entry.event.internal_metadata.need_to_check_redaction():
                    original_domain = get_domain_from_id(original_event.sender)
                    redaction_domain = get_domain_from_id(entry.event.sender)
                    if original_domain != redaction_domain:
                        logger.info("Withholding redaction %s whose sender domain %s doesn't match that of redacted event %s %s", event_id, redaction_domain, redacted_event_id, original_domain)
                        continue
                    entry.event.internal_metadata.recheck_redaction = False
            event = entry.event
            if entry.redacted_event:
                if redact_behaviour == EventRedactBehaviour.block:
                    continue
                elif redact_behaviour == EventRedactBehaviour.redact:
                    event = entry.redacted_event
            events.append(event)
            if get_prev_content:
                if 'replaces_state' in event.unsigned:
                    prev = await self.get_event(event.unsigned['replaces_state'], get_prev_content=False, allow_none=True)
                    if prev:
                        event.unsigned = dict(event.unsigned)
                        event.unsigned['prev_content'] = prev.content
                        event.unsigned['prev_sender'] = prev.sender
        return events

    @cancellable
    async def get_unredacted_events_from_cache_or_db(self, event_ids: Iterable[str], allow_rejected: bool=False) -> Dict[str, EventCacheEntry]:
        """Fetch a bunch of events from the cache or the database.

        Note that the events pulled by this function will not have any redactions
        applied, and no guarantee is made about the ordering of the events returned.

        If events are pulled from the database, they will be cached for future lookups.

        Unknown events are omitted from the response.

        Args:

            event_ids: The event_ids of the events to fetch

            allow_rejected: Whether to include rejected events. If False,
                rejected events are omitted from the response.

        Returns:
            map from event id to result
        """
        event_entry_map = self._get_events_from_local_cache(event_ids)
        missing_events_ids = {e for e in event_ids if e not in event_entry_map}
        already_fetching_ids: Set[str] = set()
        already_fetching_deferreds: Set[ObservableDeferred[Dict[str, EventCacheEntry]]] = set()
        for event_id in missing_events_ids:
            deferred = self._current_event_fetches.get(event_id)
            if deferred is not None:
                already_fetching_ids.add(event_id)
                already_fetching_deferreds.add(deferred)
        missing_events_ids.difference_update(already_fetching_ids)
        if missing_events_ids:

            async def get_missing_events_from_cache_or_db() -> Dict[str, EventCacheEntry]:
                """Fetches the events in `missing_event_ids` from the database.

                Also creates entries in `self._current_event_fetches` to allow
                concurrent `_get_events_from_cache_or_db` calls to reuse the same fetch.
                """
                log_ctx = current_context()
                log_ctx.record_event_fetch(len(missing_events_ids))
                fetching_deferred: ObservableDeferred[Dict[str, EventCacheEntry]] = ObservableDeferred(defer.Deferred(), consumeErrors=True)
                for event_id in missing_events_ids:
                    self._current_event_fetches[event_id] = fetching_deferred
                try:
                    missing_events = await self._get_events_from_external_cache(missing_events_ids)
                    db_missing_events = await self._get_events_from_db(missing_events_ids - missing_events.keys())
                    missing_events.update(db_missing_events)
                except Exception as e:
                    with PreserveLoggingContext():
                        fetching_deferred.errback(e)
                    raise e
                finally:
                    for event_id in missing_events_ids:
                        self._current_event_fetches.pop(event_id, None)
                with PreserveLoggingContext():
                    fetching_deferred.callback(missing_events)
                return missing_events
            missing_events: Dict[str, EventCacheEntry] = await delay_cancellation(get_missing_events_from_cache_or_db())
            event_entry_map.update(missing_events)
        if already_fetching_deferreds:
            results = await make_deferred_yieldable(defer.gatherResults((d.observe() for d in already_fetching_deferreds), consumeErrors=True)).addErrback(unwrapFirstError)
            for result in results:
                event_entry_map.update(((event_id, entry) for (event_id, entry) in result.items() if event_id in already_fetching_ids))
        if not allow_rejected:
            event_entry_map = {event_id: entry for (event_id, entry) in event_entry_map.items() if not entry.event.rejected_reason}
        return event_entry_map

    def invalidate_get_event_cache_after_txn(self, txn: LoggingTransaction, event_id: str) -> None:
        if False:
            while True:
                i = 10
        '\n        Prepares a database transaction to invalidate the get event cache for a given\n        event ID when executed successfully. This is achieved by attaching two callbacks\n        to the transaction, one to invalidate the async cache and one for the in memory\n        sync cache (importantly called in that order).\n\n        Arguments:\n            txn: the database transaction to attach the callbacks to\n            event_id: the event ID to be invalidated from caches\n        '
        txn.async_call_after(self._invalidate_async_get_event_cache, event_id)
        txn.call_after(self._invalidate_local_get_event_cache, event_id)

    async def _invalidate_async_get_event_cache(self, event_id: str) -> None:
        """
        Invalidates an event in the asynchronous get event cache, which may be remote.

        Arguments:
            event_id: the event ID to invalidate
        """
        await self._get_event_cache.invalidate((event_id,))

    def _invalidate_local_get_event_cache(self, event_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Invalidates an event in local in-memory get event caches.\n\n        Arguments:\n            event_id: the event ID to invalidate\n        '
        self._get_event_cache.invalidate_local((event_id,))
        self._event_ref.pop(event_id, None)
        self._current_event_fetches.pop(event_id, None)

    def _invalidate_local_get_event_cache_all(self) -> None:
        if False:
            print('Hello World!')
        'Clears the in-memory get event caches.\n\n        Used when we purge room history.\n        '
        self._get_event_cache.clear()
        self._event_ref.clear()
        self._current_event_fetches.clear()

    async def _get_events_from_cache(self, events: Iterable[str], update_metrics: bool=True) -> Dict[str, EventCacheEntry]:
        """Fetch events from the caches, both in memory and any external.

        May return rejected events.

        Args:
            events: list of event_ids to fetch
            update_metrics: Whether to update the cache hit ratio metrics
        """
        event_map = self._get_events_from_local_cache(events, update_metrics=update_metrics)
        missing_event_ids = (e for e in events if e not in event_map)
        event_map.update(await self._get_events_from_external_cache(events=missing_event_ids, update_metrics=update_metrics))
        return event_map

    async def _get_events_from_external_cache(self, events: Iterable[str], update_metrics: bool=True) -> Dict[str, EventCacheEntry]:
        """Fetch events from any configured external cache.

        May return rejected events.

        Args:
            events: list of event_ids to fetch
            update_metrics: Whether to update the cache hit ratio metrics
        """
        event_map = {}
        for event_id in events:
            ret = await self._get_event_cache.get_external((event_id,), None, update_metrics=update_metrics)
            if ret:
                event_map[event_id] = ret
        return event_map

    def _get_events_from_local_cache(self, events: Iterable[str], update_metrics: bool=True) -> Dict[str, EventCacheEntry]:
        if False:
            print('Hello World!')
        'Fetch events from the local, in memory, caches.\n\n        May return rejected events.\n\n        Args:\n            events: list of event_ids to fetch\n            update_metrics: Whether to update the cache hit ratio metrics\n        '
        event_map = {}
        for event_id in events:
            ret = self._get_event_cache.get_local((event_id,), None, update_metrics=update_metrics)
            if ret:
                event_map[event_id] = ret
                continue
            event = self._event_ref.get(event_id)
            if event:
                cache_entry = EventCacheEntry(event=event, redacted_event=None)
                event_map[event_id] = cache_entry
                self._get_event_cache.set_local((event_id,), cache_entry)
        return event_map

    async def get_stripped_room_state_from_event_context(self, context: EventContext, state_keys_to_include: StateFilter, membership_user_id: Optional[str]=None) -> List[JsonDict]:
        """
        Retrieve the stripped state from a room, given an event context to retrieve state
        from as well as the state types to include. Optionally, include the membership
        events from a specific user.

        "Stripped" state means that only the `type`, `state_key`, `content` and `sender` keys
        are included from each state event.

        Args:
            context: The event context to retrieve state of the room from.
            state_keys_to_include: The state events to include, for each event type.
            membership_user_id: An optional user ID to include the stripped membership state
                events of. This is useful when generating the stripped state of a room for
                invites. We want to send membership events of the inviter, so that the
                invitee can display the inviter's profile information if the room lacks any.

        Returns:
            A list of dictionaries, each representing a stripped state event from the room.
        """
        if membership_user_id:
            types = chain(state_keys_to_include.to_types(), [(EventTypes.Member, membership_user_id)])
            filter = StateFilter.from_types(types)
        else:
            filter = state_keys_to_include
        selected_state_ids = await context.get_current_state_ids(filter)
        assert selected_state_ids is not None
        selected_state_ids = filter.filter_state(selected_state_ids)
        state_to_include = await self.get_events(selected_state_ids.values())
        return [{'type': e.type, 'state_key': e.state_key, 'content': e.content, 'sender': e.sender} for e in state_to_include.values()]

    def _maybe_start_fetch_thread(self) -> None:
        if False:
            while True:
                i = 10
        'Starts an event fetch thread if we are not yet at the maximum number.'
        with self._event_fetch_lock:
            if self._event_fetch_list and self._event_fetch_ongoing < EVENT_QUEUE_THREADS:
                self._event_fetch_ongoing += 1
                event_fetch_ongoing_gauge.set(self._event_fetch_ongoing)
                should_start = True
            else:
                should_start = False
        if should_start:
            run_as_background_process('fetch_events', self._fetch_thread)

    async def _fetch_thread(self) -> None:
        """Services requests for events from `_event_fetch_list`."""
        exc = None
        try:
            await self.db_pool.runWithConnection(self._fetch_loop)
        except BaseException as e:
            exc = e
            raise
        finally:
            should_restart = False
            event_fetches_to_fail = []
            with self._event_fetch_lock:
                self._event_fetch_ongoing -= 1
                event_fetch_ongoing_gauge.set(self._event_fetch_ongoing)
                if self._event_fetch_list:
                    if exc is None:
                        should_restart = True
                    elif isinstance(exc, Exception):
                        if self._event_fetch_ongoing == 0:
                            event_fetches_to_fail = self._event_fetch_list
                            self._event_fetch_list = []
                        else:
                            pass
                    else:
                        pass
            if should_restart:
                self._maybe_start_fetch_thread()
            if event_fetches_to_fail:
                assert exc is not None
                with PreserveLoggingContext():
                    for (_, deferred) in event_fetches_to_fail:
                        deferred.errback(exc)

    def _fetch_loop(self, conn: LoggingDatabaseConnection) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Takes a database connection and waits for requests for events from\n        the _event_fetch_list queue.\n        '
        i = 0
        while True:
            with self._event_fetch_lock:
                event_list = self._event_fetch_list
                self._event_fetch_list = []
                if not event_list:
                    single_threaded = self.database_engine.single_threaded
                    if not self.USE_DEDICATED_DB_THREADS_FOR_EVENT_FETCHING or single_threaded or i > EVENT_QUEUE_ITERATIONS:
                        return
                    self._event_fetch_lock.wait(EVENT_QUEUE_TIMEOUT_S)
                    i += 1
                    continue
                i = 0
            self._fetch_event_list(conn, event_list)

    def _fetch_event_list(self, conn: LoggingDatabaseConnection, event_list: List[Tuple[Iterable[str], 'defer.Deferred[Dict[str, _EventRow]]']]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Handle a load of requests from the _event_fetch_list queue\n\n        Args:\n            conn: database connection\n\n            event_list:\n                The fetch requests. Each entry consists of a list of event\n                ids to be fetched, and a deferred to be completed once the\n                events have been fetched.\n\n                The deferreds are callbacked with a dictionary mapping from event id\n                to event row. Note that it may well contain additional events that\n                were not part of this request.\n        '
        with Measure(self._clock, '_fetch_event_list'):
            try:
                events_to_fetch = {event_id for (events, _) in event_list for event_id in events}
                row_dict = self.db_pool.new_transaction(conn, 'do_fetch', [], [], [], self._fetch_event_rows, events_to_fetch)

                def fire() -> None:
                    if False:
                        print('Hello World!')
                    for (_, d) in event_list:
                        d.callback(row_dict)
                with PreserveLoggingContext():
                    self.hs.get_reactor().callFromThread(fire)
            except Exception as e:
                logger.exception('do_fetch')

                def fire_errback(exc: Exception) -> None:
                    if False:
                        print('Hello World!')
                    for (_, d) in event_list:
                        d.errback(exc)
                with PreserveLoggingContext():
                    self.hs.get_reactor().callFromThread(fire_errback, e)

    async def _get_events_from_db(self, event_ids: Collection[str]) -> Dict[str, EventCacheEntry]:
        """Fetch a bunch of events from the database.

        May return rejected events.

        Returned events will be added to the cache for future lookups.

        Unknown events are omitted from the response.

        Args:
            event_ids: The event_ids of the events to fetch

        Returns:
            map from event id to result. May return extra events which
            weren't asked for.
        """
        fetched_event_ids: Set[str] = set()
        fetched_events: Dict[str, _EventRow] = {}

        async def _fetch_event_ids_and_get_outstanding_redactions(event_ids_to_fetch: Collection[str]) -> Collection[str]:
            """
            Fetch all of the given event_ids and return any associated redaction event_ids
            that we still need to fetch in the next iteration.
            """
            row_map = await self._enqueue_events(event_ids_to_fetch)
            redaction_ids: Set[str] = set()
            for event_id in event_ids_to_fetch:
                row = row_map.get(event_id)
                fetched_event_ids.add(event_id)
                if row:
                    fetched_events[event_id] = row
                    redaction_ids.update(row.redactions)
            event_ids_to_fetch = redaction_ids.difference(fetched_event_ids)
            return event_ids_to_fetch
        event_ids_to_fetch = await _fetch_event_ids_and_get_outstanding_redactions(event_ids)
        with start_active_span('recursively fetching redactions'):
            while event_ids_to_fetch:
                logger.debug('Also fetching redaction events %s', event_ids_to_fetch)
                event_ids_to_fetch = await _fetch_event_ids_and_get_outstanding_redactions(event_ids_to_fetch)
        event_map: Dict[str, EventBase] = {}
        for (event_id, row) in fetched_events.items():
            assert row.event_id == event_id
            rejected_reason = row.rejected_reason
            try:
                d = db_to_json(row.json)
            except ValueError:
                logger.error('Unable to parse json from event: %s', event_id)
                continue
            try:
                internal_metadata = db_to_json(row.internal_metadata)
            except ValueError:
                logger.error('Unable to parse internal_metadata from event: %s', event_id)
                continue
            format_version = row.format_version
            if format_version is None:
                format_version = EventFormatVersions.ROOM_V1_V2
            room_version_id = row.room_version_id
            room_version: Optional[RoomVersion]
            if not room_version_id:
                if d['type'] != EventTypes.Member:
                    raise InvalidEventError('Room %s for event %s is unknown' % (d['room_id'], event_id))
                if format_version == EventFormatVersions.ROOM_V1_V2:
                    room_version = RoomVersions.V1
                elif format_version == EventFormatVersions.ROOM_V3:
                    room_version = RoomVersions.V3
                else:
                    room_version = RoomVersions.V5
            else:
                room_version = KNOWN_ROOM_VERSIONS.get(room_version_id)
                if not room_version:
                    logger.warning('Event %s in room %s has unknown room version %s', event_id, d['room_id'], room_version_id)
                    continue
                if room_version.event_format != format_version:
                    logger.error('Event %s in room %s with version %s has wrong format: expected %s, was %s', event_id, d['room_id'], room_version_id, room_version.event_format, format_version)
                    continue
            original_ev = make_event_from_dict(event_dict=d, room_version=room_version, internal_metadata_dict=internal_metadata, rejected_reason=rejected_reason)
            original_ev.internal_metadata.stream_ordering = row.stream_ordering
            original_ev.internal_metadata.outlier = row.outlier
            if original_ev.event_id != event_id:
                raise RuntimeError(f"Database corruption: Event {event_id} in room {d['room_id']} from the database appears to have been modified (calculated event id {original_ev.event_id})")
            event_map[event_id] = original_ev
        result_map: Dict[str, EventCacheEntry] = {}
        for (event_id, original_ev) in event_map.items():
            redactions = fetched_events[event_id].redactions
            redacted_event = self._maybe_redact_event_row(original_ev, redactions, event_map)
            cache_entry = EventCacheEntry(event=original_ev, redacted_event=redacted_event)
            await self._get_event_cache.set((event_id,), cache_entry)
            result_map[event_id] = cache_entry
            if not redacted_event:
                self._event_ref[event_id] = original_ev
        return result_map

    async def _enqueue_events(self, events: Collection[str]) -> Dict[str, _EventRow]:
        """Fetches events from the database using the _event_fetch_list. This
        allows batch and bulk fetching of events - it allows us to fetch events
        without having to create a new transaction for each request for events.

        Args:
            events: events to be fetched.

        Returns:
            A map from event id to row data from the database. May contain events
            that weren't requested.
        """
        events_d: 'defer.Deferred[Dict[str, _EventRow]]' = defer.Deferred()
        with self._event_fetch_lock:
            self._event_fetch_list.append((events, events_d))
            self._event_fetch_lock.notify()
        self._maybe_start_fetch_thread()
        logger.debug('Loading %d events: %s', len(events), events)
        with PreserveLoggingContext():
            row_map = await events_d
        logger.debug('Loaded %d events (%d rows)', len(events), len(row_map))
        return row_map

    def _fetch_event_rows(self, txn: LoggingTransaction, event_ids: Iterable[str]) -> Dict[str, _EventRow]:
        if False:
            while True:
                i = 10
        'Fetch event rows from the database\n\n        Events which are not found are omitted from the result.\n\n        Args:\n            txn: The database transaction.\n            event_ids: event IDs to fetch\n\n        Returns:\n            A map from event id to event info.\n        '
        event_dict = {}
        for evs in batch_iter(event_ids, 200):
            sql = '                SELECT\n                  e.event_id,\n                  e.stream_ordering,\n                  ej.internal_metadata,\n                  ej.json,\n                  ej.format_version,\n                  r.room_version,\n                  rej.reason,\n                  e.outlier\n                FROM events AS e\n                  JOIN event_json AS ej USING (event_id)\n                  LEFT JOIN rooms r ON r.room_id = e.room_id\n                  LEFT JOIN rejections as rej USING (event_id)\n                WHERE '
            (clause, args) = make_in_list_sql_clause(txn.database_engine, 'e.event_id', evs)
            txn.execute(sql + clause, args)
            for row in txn:
                event_id = row[0]
                event_dict[event_id] = _EventRow(event_id=event_id, stream_ordering=row[1], internal_metadata=row[2], json=row[3], format_version=row[4], room_version_id=row[5], rejected_reason=row[6], redactions=[], outlier=row[7])
            redactions_sql = 'SELECT event_id, redacts FROM redactions WHERE '
            (clause, args) = make_in_list_sql_clause(txn.database_engine, 'redacts', evs)
            txn.execute(redactions_sql + clause, args)
            for (redacter, redacted) in txn:
                d = event_dict.get(redacted)
                if d:
                    d.redactions.append(redacter)
        return event_dict

    def _maybe_redact_event_row(self, original_ev: EventBase, redactions: Iterable[str], event_map: Dict[str, EventBase]) -> Optional[EventBase]:
        if False:
            for i in range(10):
                print('nop')
        'Given an event object and a list of possible redacting event ids,\n        determine whether to honour any of those redactions and if so return a redacted\n        event.\n\n        Args:\n             original_ev: The original event.\n             redactions: list of event ids of potential redaction events\n             event_map: other events which have been fetched, in which we can\n                look up the redaaction events. Map from event id to event.\n\n        Returns:\n            If the event should be redacted, a pruned event object. Otherwise, None.\n        '
        if original_ev.type == 'm.room.create':
            return None
        for redaction_id in redactions:
            redaction_event = event_map.get(redaction_id)
            if not redaction_event or redaction_event.rejected_reason:
                logger.debug('%s was redacted by %s but redaction not found/authed', original_ev.event_id, redaction_id)
                continue
            if redaction_event.room_id != original_ev.room_id:
                logger.debug('%s was redacted by %s but redaction was in a different room!', original_ev.event_id, redaction_id)
                continue
            if redaction_event.internal_metadata.need_to_check_redaction():
                expected_domain = get_domain_from_id(original_ev.sender)
                if get_domain_from_id(redaction_event.sender) == expected_domain:
                    redaction_event.internal_metadata.recheck_redaction = False
                else:
                    logger.debug("%s was redacted by %s but the senders don't match", original_ev.event_id, redaction_id)
                    continue
            logger.debug('Redacting %s due to %s', original_ev.event_id, redaction_id)
            redacted_event = prune_event(original_ev)
            redacted_event.unsigned['redacted_by'] = redaction_id
            redacted_event.unsigned['redacted_because'] = redaction_event
            return redacted_event
        return None

    async def have_events_in_timeline(self, event_ids: Iterable[str]) -> Set[str]:
        """Given a list of event ids, check if we have already processed and
        stored them as non outliers.
        """
        rows = cast(List[Tuple[str]], await self.db_pool.simple_select_many_batch(table='events', retcols=('event_id',), column='event_id', iterable=list(event_ids), keyvalues={'outlier': False}, desc='have_events_in_timeline'))
        return {r[0] for r in rows}

    @trace
    @tag_args
    async def have_seen_events(self, room_id: str, event_ids: Iterable[str]) -> Set[str]:
        """Given a list of event ids, check if we have already processed them.

        The room_id is only used to structure the cache (so that it can later be
        invalidated by room_id) - there is no guarantee that the events are actually
        in the room in question.

        Args:
            room_id: Room we are polling
            event_ids: events we are looking for

        Returns:
            The set of events we have already seen.
        """
        results: Set[str] = set()
        for event_ids_chunk in batch_iter(event_ids, 500):
            events_seen_dict = await self._have_seen_events_dict(room_id, event_ids_chunk)
            results.update((eid for (eid, have_event) in events_seen_dict.items() if have_event))
        return results

    @cachedList(cached_method_name='have_seen_event', list_name='event_ids')
    async def _have_seen_events_dict(self, room_id: str, event_ids: Collection[str]) -> Mapping[str, bool]:
        """Helper for have_seen_events

        Returns:
             a dict {event_id -> bool}
        """

        def have_seen_events_txn(txn: LoggingTransaction) -> Dict[str, bool]:
            if False:
                for i in range(10):
                    print('nop')
            sql = 'SELECT event_id FROM events AS e WHERE '
            (clause, args) = make_in_list_sql_clause(txn.database_engine, 'e.event_id', event_ids)
            txn.execute(sql + clause, args)
            found_events = {eid for (eid,) in txn}
            return {eid: eid in found_events for eid in event_ids}
        return await self.db_pool.runInteraction('have_seen_events', have_seen_events_txn)

    @cached(max_entries=100000, tree=True)
    async def have_seen_event(self, room_id: str, event_id: str) -> bool:
        res = await self._have_seen_events_dict(room_id, [event_id])
        return res[event_id]

    def _get_current_state_event_counts_txn(self, txn: LoggingTransaction, room_id: str) -> int:
        if False:
            i = 10
            return i + 15
        '\n        See get_current_state_event_counts.\n        '
        sql = 'SELECT COUNT(*) FROM current_state_events WHERE room_id=?'
        txn.execute(sql, (room_id,))
        row = txn.fetchone()
        return row[0] if row else 0

    async def get_current_state_event_counts(self, room_id: str) -> int:
        """
        Gets the current number of state events in a room.

        Args:
            room_id: The room ID to query.

        Returns:
            The current number of state events.
        """
        return await self.db_pool.runInteraction('get_current_state_event_counts', self._get_current_state_event_counts_txn, room_id)

    async def get_room_complexity(self, room_id: str) -> Dict[str, float]:
        """
        Get a rough approximation of the complexity of the room. This is used by
        remote servers to decide whether they wish to join the room or not.
        Higher complexity value indicates that being in the room will consume
        more resources.

        Args:
            room_id: The room ID to query.

        Returns:
            Map of complexity version to complexity.
        """
        state_events = await self.get_current_state_event_counts(room_id)
        complexity_v1 = round(state_events / 500, 2)
        return {'v1': complexity_v1}

    async def get_all_new_forward_event_rows(self, instance_name: str, last_id: int, current_id: int, limit: int) -> List[Tuple[int, str, str, str, str, str, str, str, bool, bool]]:
        """Returns new events, for the Events replication stream

        Args:
            last_id: the last stream_id from the previous batch.
            current_id: the maximum stream_id to return up to
            limit: the maximum number of rows to return

        Returns:
            a list of events stream rows. Each tuple consists of a stream id as
            the first element, followed by fields suitable for casting into an
            EventsStreamRow.
        """

        def get_all_new_forward_event_rows(txn: LoggingTransaction) -> List[Tuple[int, str, str, str, str, str, str, str, bool, bool]]:
            if False:
                for i in range(10):
                    print('nop')
            sql = 'SELECT e.stream_ordering, e.event_id, e.room_id, e.type, se.state_key, redacts, relates_to_id, membership, rejections.reason IS NOT NULL, e.outlier FROM events AS e LEFT JOIN redactions USING (event_id) LEFT JOIN state_events AS se USING (event_id) LEFT JOIN event_relations USING (event_id) LEFT JOIN room_memberships USING (event_id) LEFT JOIN rejections USING (event_id) WHERE ? < stream_ordering AND stream_ordering <= ? AND instance_name = ? ORDER BY stream_ordering ASC LIMIT ?'
            txn.execute(sql, (last_id, current_id, instance_name, limit))
            return cast(List[Tuple[int, str, str, str, str, str, str, str, bool, bool]], txn.fetchall())
        return await self.db_pool.runInteraction('get_all_new_forward_event_rows', get_all_new_forward_event_rows)

    async def get_ex_outlier_stream_rows(self, instance_name: str, last_id: int, current_id: int) -> List[Tuple[int, str, str, str, str, str, str, str, bool, bool]]:
        """Returns de-outliered events, for the Events replication stream

        Args:
            last_id: the last stream_id from the previous batch.
            current_id: the maximum stream_id to return up to

        Returns:
            a list of events stream rows. Each tuple consists of a stream id as
            the first element, followed by fields suitable for casting into an
            EventsStreamRow.
        """

        def get_ex_outlier_stream_rows_txn(txn: LoggingTransaction) -> List[Tuple[int, str, str, str, str, str, str, str, bool, bool]]:
            if False:
                print('Hello World!')
            sql = 'SELECT out.event_stream_ordering, e.event_id, e.room_id, e.type, se.state_key, redacts, relates_to_id, membership, rejections.reason IS NOT NULL, e.outlier FROM events AS e INNER JOIN ex_outlier_stream AS out USING (event_id) LEFT JOIN redactions USING (event_id) LEFT JOIN state_events AS se USING (event_id) LEFT JOIN event_relations USING (event_id) LEFT JOIN room_memberships USING (event_id) LEFT JOIN rejections USING (event_id) WHERE ? < out.event_stream_ordering AND out.event_stream_ordering <= ? AND out.instance_name = ? ORDER BY out.event_stream_ordering ASC'
            txn.execute(sql, (last_id, current_id, instance_name))
            return cast(List[Tuple[int, str, str, str, str, str, str, str, bool, bool]], txn.fetchall())
        return await self.db_pool.runInteraction('get_ex_outlier_stream_rows', get_ex_outlier_stream_rows_txn)

    async def get_all_new_backfill_event_rows(self, instance_name: str, last_id: int, current_id: int, limit: int) -> Tuple[List[Tuple[int, Tuple[str, str, str, str, str, str]]], int, bool]:
        """Get updates for backfill replication stream, including all new
        backfilled events and events that have gone from being outliers to not.

        NOTE: The IDs given here are from replication, and so should be
        *positive*.

        Args:
            instance_name: The writer we want to fetch updates from. Unused
                here since there is only ever one writer.
            last_id: The token to fetch updates from. Exclusive.
            current_id: The token to fetch updates up to. Inclusive.
            limit: The requested limit for the number of rows to return. The
                function may return more or fewer rows.

        Returns:
            A tuple consisting of: the updates, a token to use to fetch
            subsequent updates, and whether we returned fewer rows than exists
            between the requested tokens due to the limit.

            The token returned can be used in a subsequent call to this
            function to get further updatees.

            The updates are a list of 2-tuples of stream ID and the row data
        """
        if last_id == current_id:
            return ([], current_id, False)

        def get_all_new_backfill_event_rows(txn: LoggingTransaction) -> Tuple[List[Tuple[int, Tuple[str, str, str, str, str, str]]], int, bool]:
            if False:
                for i in range(10):
                    print('nop')
            sql = 'SELECT -e.stream_ordering, e.event_id, e.room_id, e.type, se.state_key, redacts, relates_to_id FROM events AS e LEFT JOIN redactions USING (event_id) LEFT JOIN state_events AS se USING (event_id) LEFT JOIN event_relations USING (event_id) WHERE ? > stream_ordering AND stream_ordering >= ?  AND instance_name = ? ORDER BY stream_ordering ASC LIMIT ?'
            txn.execute(sql, (-last_id, -current_id, instance_name, limit))
            new_event_updates: List[Tuple[int, Tuple[str, str, str, str, str, str]]] = []
            row: Tuple[int, str, str, str, str, str, str]
            for row in txn:
                new_event_updates.append((row[0], row[1:]))
            limited = False
            if len(new_event_updates) == limit:
                upper_bound = new_event_updates[-1][0]
                limited = True
            else:
                upper_bound = current_id
            sql = 'SELECT -event_stream_ordering, e.event_id, e.room_id, e.type, se.state_key, redacts, relates_to_id FROM events AS e INNER JOIN ex_outlier_stream AS out USING (event_id) LEFT JOIN redactions USING (event_id) LEFT JOIN state_events AS se USING (event_id) LEFT JOIN event_relations USING (event_id) WHERE ? > event_stream_ordering AND event_stream_ordering >= ? AND out.instance_name = ? ORDER BY event_stream_ordering DESC'
            txn.execute(sql, (-last_id, -upper_bound, instance_name))
            for row in txn:
                new_event_updates.append((row[0], row[1:]))
            if len(new_event_updates) >= limit:
                upper_bound = new_event_updates[-1][0]
                limited = True
            return (new_event_updates, upper_bound, limited)
        return await self.db_pool.runInteraction('get_all_new_backfill_event_rows', get_all_new_backfill_event_rows)

    async def get_all_updated_current_state_deltas(self, instance_name: str, from_token: int, to_token: int, target_row_count: int) -> Tuple[List[Tuple[int, str, str, str, str]], int, bool]:
        """Fetch updates from current_state_delta_stream

        Args:
            from_token: The previous stream token. Updates from this stream id will
                be excluded.

            to_token: The current stream token (ie the upper limit). Updates up to this
                stream id will be included (modulo the 'limit' param)

            target_row_count: The number of rows to try to return. If more rows are
                available, we will set 'limited' in the result. In the event of a large
                batch, we may return more rows than this.
        Returns:
            A triplet `(updates, new_last_token, limited)`, where:
               * `updates` is a list of database tuples.
               * `new_last_token` is the new position in stream.
               * `limited` is whether there are more updates to fetch.
        """

        def get_all_updated_current_state_deltas_txn(txn: LoggingTransaction) -> List[Tuple[int, str, str, str, str]]:
            if False:
                while True:
                    i = 10
            sql = '\n                SELECT stream_id, room_id, type, state_key, event_id\n                FROM current_state_delta_stream\n                WHERE ? < stream_id AND stream_id <= ?\n                    AND instance_name = ?\n                ORDER BY stream_id ASC LIMIT ?\n            '
            txn.execute(sql, (from_token, to_token, instance_name, target_row_count))
            return cast(List[Tuple[int, str, str, str, str]], txn.fetchall())

        def get_deltas_for_stream_id_txn(txn: LoggingTransaction, stream_id: int) -> List[Tuple[int, str, str, str, str]]:
            if False:
                print('Hello World!')
            sql = '\n                SELECT stream_id, room_id, type, state_key, event_id\n                FROM current_state_delta_stream\n                WHERE stream_id = ?\n            '
            txn.execute(sql, [stream_id])
            return cast(List[Tuple[int, str, str, str, str]], txn.fetchall())
        rows: List[Tuple[int, str, str, str, str]] = await self.db_pool.runInteraction('get_all_updated_current_state_deltas', get_all_updated_current_state_deltas_txn)
        if len(rows) < target_row_count:
            return (rows, to_token, False)
        assert rows[-1][0] <= to_token
        to_token = rows[-1][0] - 1
        for idx in range(len(rows) - 1, 0, -1):
            if rows[idx - 1][0] <= to_token:
                return (rows[:idx], to_token, True)
        to_token += 1
        rows = await self.db_pool.runInteraction('get_deltas_for_stream_id', get_deltas_for_stream_id_txn, to_token)
        return (rows, to_token, True)

    @cached(max_entries=5000)
    async def get_event_ordering(self, event_id: str) -> Tuple[int, int]:
        res = await self.db_pool.simple_select_one(table='events', retcols=['topological_ordering', 'stream_ordering'], keyvalues={'event_id': event_id}, allow_none=True)
        if not res:
            raise SynapseError(404, 'Could not find event %s' % (event_id,))
        return (int(res[0]), int(res[1]))

    async def get_next_event_to_expire(self) -> Optional[Tuple[str, int]]:
        """Retrieve the entry with the lowest expiry timestamp in the event_expiry
        table, or None if there's no more event to expire.

        Returns:
            A tuple containing the event ID as its first element and an expiry timestamp
            as its second one, if there's at least one row in the event_expiry table.
            None otherwise.
        """

        def get_next_event_to_expire_txn(txn: LoggingTransaction) -> Optional[Tuple[str, int]]:
            if False:
                return 10
            txn.execute('\n                SELECT event_id, expiry_ts FROM event_expiry\n                ORDER BY expiry_ts ASC LIMIT 1\n                ')
            return cast(Optional[Tuple[str, int]], txn.fetchone())
        return await self.db_pool.runInteraction(desc='get_next_event_to_expire', func=get_next_event_to_expire_txn)

    async def get_event_id_from_transaction_id_and_device_id(self, room_id: str, user_id: str, device_id: str, txn_id: str) -> Optional[str]:
        """Look up if we have already persisted an event for the transaction ID,
        returning the event ID if so.
        """
        return await self.db_pool.simple_select_one_onecol(table='event_txn_id_device_id', keyvalues={'room_id': room_id, 'user_id': user_id, 'device_id': device_id, 'txn_id': txn_id}, retcol='event_id', allow_none=True, desc='get_event_id_from_transaction_id_and_device_id')

    async def get_already_persisted_events(self, events: Iterable[EventBase]) -> Dict[str, str]:
        """Look up if we have already persisted an event for the transaction ID,
        returning a mapping from event ID in the given list to the event ID of
        an existing event.

        Also checks if there are duplicates in the given events, if there are
        will map duplicates to the *first* event.
        """
        mapping = {}
        txn_id_to_event: Dict[Tuple[str, str, str, str], str] = {}
        for event in events:
            device_id = getattr(event.internal_metadata, 'device_id', None)
            txn_id = getattr(event.internal_metadata, 'txn_id', None)
            if device_id and txn_id:
                existing = txn_id_to_event.get((event.room_id, event.sender, device_id, txn_id))
                if existing:
                    mapping[event.event_id] = existing
                    continue
                existing = await self.get_event_id_from_transaction_id_and_device_id(event.room_id, event.sender, device_id, txn_id)
                if existing:
                    mapping[event.event_id] = existing
                    txn_id_to_event[event.room_id, event.sender, device_id, txn_id] = existing
                else:
                    txn_id_to_event[event.room_id, event.sender, device_id, txn_id] = event.event_id
        return mapping

    @wrap_as_background_process('_cleanup_old_transaction_ids')
    async def _cleanup_old_transaction_ids(self) -> None:
        """Cleans out transaction id mappings older than 24hrs."""

        def _cleanup_old_transaction_ids_txn(txn: LoggingTransaction) -> None:
            if False:
                while True:
                    i = 10
            one_day_ago = self._clock.time_msec() - 24 * 60 * 60 * 1000
            sql = '\n                DELETE FROM event_txn_id_device_id\n                WHERE inserted_ts < ?\n            '
            txn.execute(sql, (one_day_ago,))
        return await self.db_pool.runInteraction('_cleanup_old_transaction_ids', _cleanup_old_transaction_ids_txn)

    async def is_event_next_to_backward_gap(self, event: EventBase) -> bool:
        """Check if the given event is next to a backward gap of missing events.
        <latest messages> A(False)--->B(False)--->C(True)--->  <gap, unknown events> <oldest messages>

        Args:
            room_id: room where the event lives
            event: event to check (can't be an `outlier`)

        Returns:
            Boolean indicating whether it's an extremity
        """
        assert not event.internal_metadata.is_outlier(), "is_event_next_to_backward_gap(...) can't be used with `outlier` events. This function relies on `event_backward_extremities` which won't be filled in for `outliers`."

        def is_event_next_to_backward_gap_txn(txn: LoggingTransaction) -> bool:
            if False:
                print('Hello World!')
            backward_extremity_query = '\n                SELECT 1 FROM event_backward_extremities\n                WHERE\n                    room_id = ?\n                    AND %s\n                LIMIT 1\n            '
            (clause, args) = make_in_list_sql_clause(self.database_engine, 'event_id', [event.event_id] + list(event.prev_event_ids()))
            txn.execute(backward_extremity_query % (clause,), [event.room_id] + args)
            backward_extremities = txn.fetchall()
            if len(backward_extremities):
                return True
            return False
        return await self.db_pool.runInteraction('is_event_next_to_backward_gap_txn', is_event_next_to_backward_gap_txn)

    async def is_event_next_to_forward_gap(self, event: EventBase) -> bool:
        """Check if the given event is next to a forward gap of missing events.
        The gap in front of the latest events is not considered a gap.
        <latest messages> A(False)--->B(False)--->C(False)--->  <gap, unknown events> <oldest messages>
        <latest messages> A(False)--->B(False)--->  <gap, unknown events>  --->D(True)--->E(False) <oldest messages>

        Args:
            room_id: room where the event lives
            event: event to check (can't be an `outlier`)

        Returns:
            Boolean indicating whether it's an extremity
        """
        assert not event.internal_metadata.is_outlier(), "is_event_next_to_forward_gap(...) can't be used with `outlier` events. This function relies on `event_edges` and `event_forward_extremities` which won't be filled in for `outliers`."

        def is_event_next_to_gap_txn(txn: LoggingTransaction) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            forward_extremity_query = '\n                SELECT 1 FROM event_forward_extremities\n                WHERE\n                    room_id = ?\n                    AND event_id = ?\n                LIMIT 1\n            '
            txn.execute(forward_extremity_query, (event.room_id, event.event_id))
            if txn.fetchone():
                return False
            forward_edge_query = "\n                SELECT 1 FROM event_edges\n                /* Check to make sure the event referencing our event in question is not rejected */\n                LEFT JOIN rejections ON event_edges.event_id = rejections.event_id\n                WHERE\n                    event_edges.prev_event_id = ?\n                    /* It's not a valid edge if the event referencing our event in\n                     * question is rejected.\n                     */\n                    AND rejections.event_id IS NULL\n                LIMIT 1\n            "
            txn.execute(forward_edge_query, (event.event_id,))
            if not txn.fetchone():
                return True
            return False
        return await self.db_pool.runInteraction('is_event_next_to_gap_txn', is_event_next_to_gap_txn)

    async def get_event_id_for_timestamp(self, room_id: str, timestamp: int, direction: Direction) -> Optional[str]:
        """Find the closest event to the given timestamp in the given direction.

        Args:
            room_id: Room to fetch the event from
            timestamp: The point in time (inclusive) we should navigate from in
                the given direction to find the closest event.
            direction: indicates whether we should navigate forward
                or backward from the given timestamp to find the closest event.

        Returns:
            The closest event_id otherwise None if we can't find any event in
            the given direction.
        """
        if direction == Direction.BACKWARDS:
            comparison_operator = '<='
            order = 'DESC'
        else:
            comparison_operator = '>='
            order = 'ASC'
        sql_template = f"""\n            SELECT event_id FROM events\n            LEFT JOIN rejections USING (event_id)\n            WHERE\n                room_id = ?\n                AND origin_server_ts {comparison_operator} ?\n                /**\n                 * Make sure the event isn't an `outlier` because we have no way\n                 * to later check whether it's next to a gap. `outliers` do not\n                 * have entries in the `event_edges`, `event_forward_extremeties`,\n                 * and `event_backward_extremities` tables to check against\n                 * (used by `is_event_next_to_backward_gap` and `is_event_next_to_forward_gap`).\n                 */\n                AND NOT outlier\n                /* Make sure event is not rejected */\n                AND rejections.event_id IS NULL\n            /**\n             * First sort by the message timestamp. If the message timestamps are the\n             * same, we want the message that logically comes "next" (before/after\n             * the given timestamp) based on the DAG and its topological order (`depth`).\n             * Finally, we can tie-break based on when it was received on the server\n             * (`stream_ordering`).\n             */\n            ORDER BY origin_server_ts {order}, depth {order}, stream_ordering {order}\n            LIMIT 1;\n        """

        def get_event_id_for_timestamp_txn(txn: LoggingTransaction) -> Optional[str]:
            if False:
                return 10
            txn.execute(sql_template, (room_id, timestamp))
            row = txn.fetchone()
            if row:
                (event_id,) = row
                return event_id
            return None
        return await self.db_pool.runInteraction('get_event_id_for_timestamp_txn', get_event_id_for_timestamp_txn)

    @cachedList(cached_method_name='is_partial_state_event', list_name='event_ids')
    async def get_partial_state_events(self, event_ids: Collection[str]) -> Mapping[str, bool]:
        """Checks which of the given events have partial state

        Args:
            event_ids: the events we want to check for partial state.

        Returns:
            a dict mapping from event id to partial-stateness. We return True for
            any of the events which are unknown (or are outliers).
        """
        result = cast(List[Tuple[str]], await self.db_pool.simple_select_many_batch(table='partial_state_events', column='event_id', iterable=event_ids, retcols=['event_id'], desc='get_partial_state_events'))
        partial = {r[0] for r in result}
        return {e_id: e_id in partial for e_id in event_ids}

    @cached()
    async def is_partial_state_event(self, event_id: str) -> bool:
        """Checks if the given event has partial state"""
        result = await self.db_pool.simple_select_one_onecol(table='partial_state_events', keyvalues={'event_id': event_id}, retcol='1', allow_none=True, desc='is_partial_state_event')
        return result is not None

    async def get_partial_state_events_batch(self, room_id: str) -> List[str]:
        """
        Get a list of events in the given room that:
        - have partial state; and
        - are ready to be resynced (because they have no prev_events that are
          partial-stated)

        See the docstring on `_get_partial_state_events_batch_txn` for more
        information.
        """
        return await self.db_pool.runInteraction('get_partial_state_events_batch', self._get_partial_state_events_batch_txn, room_id)

    @staticmethod
    def _get_partial_state_events_batch_txn(txn: LoggingTransaction, room_id: str) -> List[str]:
        if False:
            print('Hello World!')
        txn.execute('\n            SELECT event_id FROM partial_state_events AS pse\n                JOIN events USING (event_id)\n            WHERE pse.room_id = ? AND\n               NOT EXISTS(\n                  SELECT 1 FROM event_edges AS ee\n                     JOIN partial_state_events AS prev_pse ON (prev_pse.event_id=ee.prev_event_id)\n                     WHERE ee.event_id=pse.event_id\n               )\n            ORDER BY events.stream_ordering\n            LIMIT 100\n            ', (room_id,))
        return [row[0] for row in txn]

    def mark_event_rejected_txn(self, txn: LoggingTransaction, event_id: str, rejection_reason: Optional[str]) -> None:
        if False:
            return 10
        "Mark an event that was previously accepted as rejected, or vice versa\n\n        This can happen, for example, when resyncing state during a faster join.\n\n        It is the caller's responsibility to ensure that other workers are\n        sent a notification so that they call `_invalidate_local_get_event_cache()`.\n\n        Args:\n            txn:\n            event_id: ID of event to update\n            rejection_reason: reason it has been rejected, or None if it is now accepted\n        "
        if rejection_reason is None:
            logger.info('Marking previously-processed event %s as accepted', event_id)
            self.db_pool.simple_delete_txn(txn, 'rejections', keyvalues={'event_id': event_id})
        else:
            logger.info('Marking previously-processed event %s as rejected(%s)', event_id, rejection_reason)
            self.db_pool.simple_upsert_txn(txn, table='rejections', keyvalues={'event_id': event_id}, values={'reason': rejection_reason, 'last_check': self._clock.time_msec()})
        self.db_pool.simple_update_txn(txn, table='events', keyvalues={'event_id': event_id}, updatevalues={'rejection_reason': rejection_reason})
        self.invalidate_get_event_cache_after_txn(txn, event_id)