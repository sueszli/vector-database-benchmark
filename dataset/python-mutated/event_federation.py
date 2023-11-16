import datetime
import itertools
import logging
from queue import Empty, PriorityQueue
from typing import TYPE_CHECKING, Collection, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple, cast
import attr
from prometheus_client import Counter, Gauge
from synapse.api.constants import MAX_DEPTH
from synapse.api.errors import StoreError
from synapse.api.room_versions import EventFormatVersions, RoomVersion
from synapse.events import EventBase, make_event_from_dict
from synapse.logging.opentracing import tag_args, trace
from synapse.metrics.background_process_metrics import wrap_as_background_process
from synapse.storage._base import SQLBaseStore, db_to_json, make_in_list_sql_clause
from synapse.storage.background_updates import ForeignKeyConstraint
from synapse.storage.database import DatabasePool, LoggingDatabaseConnection, LoggingTransaction
from synapse.storage.databases.main.events_worker import EventsWorkerStore
from synapse.storage.databases.main.signatures import SignatureWorkerStore
from synapse.storage.engines import PostgresEngine, Sqlite3Engine
from synapse.types import JsonDict, StrCollection
from synapse.util import json_encoder
from synapse.util.caches.descriptors import cached
from synapse.util.caches.lrucache import LruCache
from synapse.util.cancellation import cancellable
from synapse.util.iterutils import batch_iter
if TYPE_CHECKING:
    from synapse.server import HomeServer
oldest_pdu_in_federation_staging = Gauge('synapse_federation_server_oldest_inbound_pdu_in_staging', 'The age in seconds since we received the oldest pdu in the federation staging area')
number_pdus_in_federation_queue = Gauge('synapse_federation_server_number_inbound_pdu_in_staging', 'The total number of events in the inbound federation staging')
pdus_pruned_from_federation_queue = Counter('synapse_federation_server_number_inbound_pdu_pruned', 'The number of events in the inbound federation staging that have been pruned due to the queue getting too long')
logger = logging.getLogger(__name__)
BACKFILL_EVENT_EXPONENTIAL_BACKOFF_MAXIMUM_DOUBLING_STEPS = 8
BACKFILL_EVENT_EXPONENTIAL_BACKOFF_STEP_MILLISECONDS = int(datetime.timedelta(hours=1).total_seconds() * 1000)
_LONGEST_BACKOFF_PERIOD_MILLISECONDS = 2 ** BACKFILL_EVENT_EXPONENTIAL_BACKOFF_MAXIMUM_DOUBLING_STEPS * BACKFILL_EVENT_EXPONENTIAL_BACKOFF_STEP_MILLISECONDS
assert 0 < _LONGEST_BACKOFF_PERIOD_MILLISECONDS <= 2 ** 31 - 1

@attr.s(frozen=True, slots=True, auto_attribs=True)
class BackfillQueueNavigationItem:
    depth: int
    stream_ordering: int
    event_id: str
    type: str

class _NoChainCoverIndex(Exception):

    def __init__(self, room_id: str):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('Unexpectedly no chain cover for events in %s' % (room_id,))

class EventFederationWorkerStore(SignatureWorkerStore, EventsWorkerStore, SQLBaseStore):
    stream_ordering_month_ago: Optional[int]

    def __init__(self, database: DatabasePool, db_conn: LoggingDatabaseConnection, hs: 'HomeServer'):
        if False:
            print('Hello World!')
        super().__init__(database, db_conn, hs)
        self.hs = hs
        if hs.config.worker.run_background_tasks:
            hs.get_clock().looping_call(self._delete_old_forward_extrem_cache, 60 * 60 * 1000)
        self._event_auth_cache: LruCache[str, List[Tuple[str, int]]] = LruCache(500000, '_event_auth_cache', size_callback=len)
        self._clock.looping_call(self._get_stats_for_federation_staging, 30 * 1000)
        if isinstance(self.database_engine, PostgresEngine):
            self.db_pool.updates.register_background_validate_constraint_and_delete_rows(update_name='event_forward_extremities_event_id_foreign_key_constraint_update', table='event_forward_extremities', constraint_name='event_forward_extremities_event_id', constraint=ForeignKeyConstraint('events', [('event_id', 'event_id')], deferred=True), unique_columns=('event_id', 'room_id'))

    async def get_auth_chain(self, room_id: str, event_ids: Collection[str], include_given: bool=False) -> List[EventBase]:
        """Get auth events for given event_ids. The events *must* be state events.

        Args:
            room_id: The room the event is in.
            event_ids: state events
            include_given: include the given events in result

        Returns:
            list of events
        """
        event_ids = await self.get_auth_chain_ids(room_id, event_ids, include_given=include_given)
        return await self.get_events_as_list(event_ids)

    @trace
    @tag_args
    async def get_auth_chain_ids(self, room_id: str, event_ids: Collection[str], include_given: bool=False) -> Set[str]:
        """Get auth events for given event_ids. The events *must* be state events.

        Args:
            room_id: The room the event is in.
            event_ids: state events
            include_given: include the given events in result

        Returns:
            set of event_ids
        """
        room = await self.get_room(room_id)
        if room[1]:
            try:
                return await self.db_pool.runInteraction('get_auth_chain_ids_chains', self._get_auth_chain_ids_using_cover_index_txn, room_id, event_ids, include_given)
            except _NoChainCoverIndex:
                pass
        return await self.db_pool.runInteraction('get_auth_chain_ids', self._get_auth_chain_ids_txn, event_ids, include_given)

    def _get_auth_chain_ids_using_cover_index_txn(self, txn: LoggingTransaction, room_id: str, event_ids: Collection[str], include_given: bool) -> Set[str]:
        if False:
            for i in range(10):
                print('nop')
        'Calculates the auth chain IDs using the chain index.'
        initial_events = set(event_ids)
        seen_events: Set[str] = set()
        event_chains: Dict[int, int] = {}
        sql = '\n            SELECT event_id, chain_id, sequence_number\n            FROM event_auth_chains\n            WHERE %s\n        '
        for batch in batch_iter(initial_events, 1000):
            (clause, args) = make_in_list_sql_clause(txn.database_engine, 'event_id', batch)
            txn.execute(sql % (clause,), args)
            for (event_id, chain_id, sequence_number) in txn:
                seen_events.add(event_id)
                event_chains[chain_id] = max(sequence_number, event_chains.get(chain_id, 0))
        events_missing_chain_info = initial_events.difference(seen_events)
        if events_missing_chain_info:
            logger.info("Unexpectedly found that events don't have chain IDs in room %s: %s", room_id, events_missing_chain_info)
            raise _NoChainCoverIndex(room_id)
        sql = '\n            SELECT\n                origin_chain_id, origin_sequence_number,\n                target_chain_id, target_sequence_number\n            FROM event_auth_chain_links\n            WHERE %s\n        '
        chains: Dict[int, int] = {}
        for batch2 in batch_iter(event_chains, 1000):
            (clause, args) = make_in_list_sql_clause(txn.database_engine, 'origin_chain_id', batch2)
            txn.execute(sql % (clause,), args)
            for (origin_chain_id, origin_sequence_number, target_chain_id, target_sequence_number) in txn:
                if origin_sequence_number <= event_chains.get(origin_chain_id, 0):
                    chains[target_chain_id] = max(target_sequence_number, chains.get(target_chain_id, 0))
        for (chain_id, seq_no) in event_chains.items():
            chains[chain_id] = max(seq_no - 1, chains.get(chain_id, 0))
        if include_given:
            results = initial_events
        else:
            results = set()
        if isinstance(self.database_engine, PostgresEngine):
            sql = '\n                SELECT event_id\n                FROM event_auth_chains AS c, (VALUES ?) AS l(chain_id, max_seq)\n                WHERE\n                    c.chain_id = l.chain_id\n                    AND sequence_number <= max_seq\n            '
            rows = txn.execute_values(sql, chains.items())
            results.update((r for (r,) in rows))
        else:
            sql = '\n                SELECT event_id FROM event_auth_chains\n                WHERE chain_id = ? AND sequence_number <= ?\n            '
            for (chain_id, max_no) in chains.items():
                txn.execute(sql, (chain_id, max_no))
                results.update((r for (r,) in txn))
        return results

    def _get_auth_chain_ids_txn(self, txn: LoggingTransaction, event_ids: Collection[str], include_given: bool) -> Set[str]:
        if False:
            return 10
        "Calculates the auth chain IDs.\n\n        This is used when we don't have a cover index for the room.\n        "
        if include_given:
            results = set(event_ids)
        else:
            results = set()
        base_sql = '\n            SELECT a.event_id, auth_id, depth\n            FROM event_auth AS a\n            INNER JOIN events AS e ON (e.event_id = a.auth_id)\n            WHERE\n        '
        front = set(event_ids)
        while front:
            new_front: Set[str] = set()
            for chunk in batch_iter(front, 100):
                to_fetch: List[str] = []
                for event_id in chunk:
                    res = self._event_auth_cache.get(event_id)
                    if res is None:
                        to_fetch.append(event_id)
                    else:
                        new_front.update((auth_id for (auth_id, depth) in res))
                if to_fetch:
                    (clause, args) = make_in_list_sql_clause(txn.database_engine, 'a.event_id', to_fetch)
                    txn.execute(base_sql + clause, args)
                    to_cache: Dict[str, List[Tuple[str, int]]] = {}
                    for (event_id, auth_event_id, auth_event_depth) in txn:
                        to_cache.setdefault(event_id, []).append((auth_event_id, auth_event_depth))
                        new_front.add(auth_event_id)
                    for (event_id, auth_events) in to_cache.items():
                        self._event_auth_cache.set(event_id, auth_events)
            new_front -= results
            front = new_front
            results.update(front)
        return results

    async def get_auth_chain_difference(self, room_id: str, state_sets: List[Set[str]]) -> Set[str]:
        """Given sets of state events figure out the auth chain difference (as
        per state res v2 algorithm).

        This equivalent to fetching the full auth chain for each set of state
        and returning the events that don't appear in each and every auth
        chain.

        Returns:
            The set of the difference in auth chains.
        """
        room = await self.get_room(room_id)
        if room[1]:
            try:
                return await self.db_pool.runInteraction('get_auth_chain_difference_chains', self._get_auth_chain_difference_using_cover_index_txn, room_id, state_sets)
            except _NoChainCoverIndex:
                pass
        return await self.db_pool.runInteraction('get_auth_chain_difference', self._get_auth_chain_difference_txn, state_sets)

    def _get_auth_chain_difference_using_cover_index_txn(self, txn: LoggingTransaction, room_id: str, state_sets: List[Set[str]]) -> Set[str]:
        if False:
            print('Hello World!')
        'Calculates the auth chain difference using the chain index.\n\n        See docs/auth_chain_difference_algorithm.md for details\n        '
        initial_events = set(state_sets[0]).union(*state_sets[1:])
        chain_info: Dict[str, Tuple[int, int]] = {}
        chain_to_event: Dict[int, Dict[int, str]] = {}
        seen_chains: Set[int] = set()

        def fetch_chain_info(events_to_fetch: Collection[str]) -> None:
            if False:
                print('Hello World!')
            sql = '\n                SELECT event_id, chain_id, sequence_number\n                FROM event_auth_chains\n                WHERE %s\n            '
            for batch in batch_iter(events_to_fetch, 1000):
                (clause, args) = make_in_list_sql_clause(txn.database_engine, 'event_id', batch)
                txn.execute(sql % (clause,), args)
                for (event_id, chain_id, sequence_number) in txn:
                    chain_info[event_id] = (chain_id, sequence_number)
                    seen_chains.add(chain_id)
                    chain_to_event.setdefault(chain_id, {})[sequence_number] = event_id
        fetch_chain_info(initial_events)
        events_missing_chain_info = initial_events.difference(chain_info)
        result: Set[str] = set()
        if events_missing_chain_info:
            result = self._fixup_auth_chain_difference_sets(txn, room_id, state_sets=state_sets, events_missing_chain_info=events_missing_chain_info, events_that_have_chain_index=chain_info)
            new_events_to_fetch = {event_id for state_set in state_sets for event_id in state_set if event_id not in initial_events}
            fetch_chain_info(new_events_to_fetch)
        set_to_chain: List[Dict[int, int]] = []
        for state_set in state_sets:
            chains: Dict[int, int] = {}
            set_to_chain.append(chains)
            for state_id in state_set:
                (chain_id, seq_no) = chain_info[state_id]
                chains[chain_id] = max(seq_no, chains.get(chain_id, 0))
        sql = '\n            SELECT\n                origin_chain_id, origin_sequence_number,\n                target_chain_id, target_sequence_number\n            FROM event_auth_chain_links\n            WHERE %s\n        '
        for batch2 in batch_iter(set(seen_chains), 1000):
            (clause, args) = make_in_list_sql_clause(txn.database_engine, 'origin_chain_id', batch2)
            txn.execute(sql % (clause,), args)
            for (origin_chain_id, origin_sequence_number, target_chain_id, target_sequence_number) in txn:
                for chains in set_to_chain:
                    if origin_sequence_number <= chains.get(origin_chain_id, 0):
                        chains[target_chain_id] = max(target_sequence_number, chains.get(target_chain_id, 0))
                seen_chains.add(target_chain_id)
        chain_to_gap: Dict[int, Tuple[int, int]] = {}
        for chain_id in seen_chains:
            min_seq_no = min((chains.get(chain_id, 0) for chains in set_to_chain))
            max_seq_no = max((chains.get(chain_id, 0) for chains in set_to_chain))
            if min_seq_no < max_seq_no:
                for seq_no in range(min_seq_no + 1, max_seq_no + 1):
                    event_id = chain_to_event.get(chain_id, {}).get(seq_no)
                    if event_id:
                        result.add(event_id)
                    else:
                        chain_to_gap[chain_id] = (min_seq_no, max_seq_no)
                        break
        if not chain_to_gap:
            return result
        if isinstance(self.database_engine, PostgresEngine):
            sql = '\n                SELECT event_id\n                FROM event_auth_chains AS c, (VALUES ?) AS l(chain_id, min_seq, max_seq)\n                WHERE\n                    c.chain_id = l.chain_id\n                    AND min_seq < sequence_number AND sequence_number <= max_seq\n            '
            args = [(chain_id, min_no, max_no) for (chain_id, (min_no, max_no)) in chain_to_gap.items()]
            rows = txn.execute_values(sql, args)
            result.update((r for (r,) in rows))
        else:
            sql = '\n                SELECT event_id FROM event_auth_chains\n                WHERE chain_id = ? AND ? < sequence_number AND sequence_number <= ?\n            '
            for (chain_id, (min_no, max_no)) in chain_to_gap.items():
                txn.execute(sql, (chain_id, min_no, max_no))
                result.update((r for (r,) in txn))
        return result

    def _fixup_auth_chain_difference_sets(self, txn: LoggingTransaction, room_id: str, state_sets: List[Set[str]], events_missing_chain_info: Set[str], events_that_have_chain_index: Collection[str]) -> Set[str]:
        if False:
            print('Hello World!')
        "Helper for `_get_auth_chain_difference_using_cover_index_txn` to\n        handle the case where we haven't calculated the chain cover index for\n        all events.\n\n        This modifies `state_sets` so that they only include events that have a\n        chain cover index, and returns a set of event IDs that are part of the\n        auth difference.\n        "
        sql = '\n            SELECT tc.event_id, ea.auth_id, eac.chain_id IS NOT NULL\n            FROM event_auth_chain_to_calculate AS tc\n            LEFT JOIN event_auth AS ea USING (event_id)\n            LEFT JOIN event_auth_chains AS eac ON (ea.auth_id = eac.event_id)\n            WHERE tc.room_id = ?\n        '
        txn.execute(sql, (room_id,))
        event_to_auth_ids: Dict[str, Set[str]] = {}
        events_that_have_chain_index = set(events_that_have_chain_index)
        for (event_id, auth_id, auth_id_has_chain) in txn:
            s = event_to_auth_ids.setdefault(event_id, set())
            if auth_id is not None:
                s.add(auth_id)
                if auth_id_has_chain:
                    events_that_have_chain_index.add(auth_id)
        if events_missing_chain_info - event_to_auth_ids.keys():
            logger.info("Unexpectedly found that events don't have chain IDs in room %s: %s", room_id, events_missing_chain_info - event_to_auth_ids.keys())
            raise _NoChainCoverIndex(room_id)
        event_id_to_partial_auth_chain: Dict[str, Set[str]] = {}
        for (event_id, auth_ids) in event_to_auth_ids.items():
            if not any((event_id in state_set for state_set in state_sets)):
                continue
            processing = set(auth_ids)
            to_add = set()
            while processing:
                auth_id = processing.pop()
                to_add.add(auth_id)
                sub_auth_ids = event_to_auth_ids.get(auth_id)
                if sub_auth_ids is None:
                    continue
                processing.update(sub_auth_ids - to_add)
            event_id_to_partial_auth_chain[event_id] = to_add
        unindexed_state_sets: List[Set[str]] = []
        for state_set in state_sets:
            unindexed_state_set = set()
            for (event_id, auth_chain) in event_id_to_partial_auth_chain.items():
                if event_id not in state_set:
                    continue
                unindexed_state_set.add(event_id)
                state_set.discard(event_id)
                state_set.difference_update(auth_chain)
                for auth_id in auth_chain:
                    if auth_id in events_that_have_chain_index:
                        state_set.add(auth_id)
                    else:
                        unindexed_state_set.add(auth_id)
            unindexed_state_sets.append(unindexed_state_set)
        union = unindexed_state_sets[0].union(*unindexed_state_sets[1:])
        intersection = unindexed_state_sets[0].intersection(*unindexed_state_sets[1:])
        return union - intersection

    def _get_auth_chain_difference_txn(self, txn: LoggingTransaction, state_sets: List[Set[str]]) -> Set[str]:
        if False:
            print('Hello World!')
        "Calculates the auth chain difference using a breadth first search.\n\n        This is used when we don't have a cover index for the room.\n        "
        initial_events = set(state_sets[0]).union(*state_sets[1:])
        event_to_missing_sets = {event_id: {i for (i, a) in enumerate(state_sets) if event_id not in a} for event_id in initial_events}
        search: List[Tuple[int, str]] = []
        sql = '\n            SELECT depth, event_id FROM events\n            WHERE %s\n        '
        for batch in batch_iter(initial_events, 1000):
            (clause, args) = make_in_list_sql_clause(txn.database_engine, 'event_id', batch)
            txn.execute(sql % (clause,), args)
            search.extend(cast(List[Tuple[int, str]], txn.fetchall()))
        search.sort()
        event_to_auth_events: Dict[str, Set[str]] = {}
        base_sql = '\n            SELECT a.event_id, auth_id, depth\n            FROM event_auth AS a\n            INNER JOIN events AS e ON (e.event_id = a.auth_id)\n            WHERE\n        '
        while search:
            if all((not event_to_missing_sets[eid] for (_, eid) in search)):
                break
            (search, chunk) = (search[:-100], search[-100:])
            found: List[Tuple[str, str, int]] = []
            to_fetch: List[str] = []
            for (_, event_id) in chunk:
                res = self._event_auth_cache.get(event_id)
                if res is None:
                    to_fetch.append(event_id)
                else:
                    found.extend(((event_id, auth_id, depth) for (auth_id, depth) in res))
            if to_fetch:
                (clause, args) = make_in_list_sql_clause(txn.database_engine, 'a.event_id', to_fetch)
                txn.execute(base_sql + clause, args)
                to_cache: Dict[str, List[Tuple[str, int]]] = {}
                for (event_id, auth_event_id, auth_event_depth) in txn:
                    to_cache.setdefault(event_id, []).append((auth_event_id, auth_event_depth))
                    found.append((event_id, auth_event_id, auth_event_depth))
                for (event_id, auth_events) in to_cache.items():
                    self._event_auth_cache.set(event_id, auth_events)
            for (event_id, auth_event_id, auth_event_depth) in found:
                event_to_auth_events.setdefault(event_id, set()).add(auth_event_id)
                sets = event_to_missing_sets.get(auth_event_id)
                if sets is None:
                    search.append((auth_event_depth, auth_event_id))
                    sets = event_to_missing_sets[auth_event_id] = set(range(len(state_sets)))
                else:
                    a_ids = event_to_auth_events.get(auth_event_id)
                    while a_ids:
                        new_aids = set()
                        for a_id in a_ids:
                            event_to_missing_sets[a_id].intersection_update(event_to_missing_sets[event_id])
                            b = event_to_auth_events.get(a_id)
                            if b:
                                new_aids.update(b)
                        a_ids = new_aids
                sets.intersection_update(event_to_missing_sets[event_id])
            search.sort()
        return {eid for (eid, n) in event_to_missing_sets.items() if n}

    @trace
    @tag_args
    async def get_backfill_points_in_room(self, room_id: str, current_depth: int, limit: int) -> List[Tuple[str, int]]:
        """
        Get the backward extremities to backfill from in the room along with the
        approximate depth.

        Only returns events that are at a depth lower than or
        equal to the `current_depth`. Sorted by depth, highest to lowest (descending)
        so the closest events to the `current_depth` are first in the list.

        We ignore extremities that are newer than the user's current scroll position
        (ie, those with depth greater than `current_depth`) as:
            1. we don't really care about getting events that have happened
               after our current position; and
            2. by the nature of paginating and scrolling back, we have likely
               previously tried and failed to backfill from that extremity, so
               to avoid getting "stuck" requesting the same backfill repeatedly
               we drop those extremities.

        Args:
            room_id: Room where we want to find the oldest events
            current_depth: The depth at the user's current scrollback position
            limit: The max number of backfill points to return

        Returns:
            List of (event_id, depth) tuples. Sorted by depth, highest to lowest
            (descending) so the closest events to the `current_depth` are first
            in the list.
        """

        def get_backfill_points_in_room_txn(txn: LoggingTransaction, room_id: str) -> List[Tuple[str, int]]:
            if False:
                print('Hello World!')
            if isinstance(self.database_engine, PostgresEngine):
                least_function = 'LEAST'
            elif isinstance(self.database_engine, Sqlite3Engine):
                least_function = 'MIN'
            else:
                raise RuntimeError('Unknown database engine')
            sql = f"""\n                SELECT backward_extrem.event_id, event.depth FROM events AS event\n                /**\n                 * Get the edge connections from the event_edges table\n                 * so we can see whether this event's prev_events points\n                 * to a backward extremity in the next join.\n                 */\n                INNER JOIN event_edges AS edge\n                ON edge.event_id = event.event_id\n                /**\n                 * We find the "oldest" events in the room by looking for\n                 * events connected to backwards extremeties (oldest events\n                 * in the room that we know of so far).\n                 */\n                INNER JOIN event_backward_extremities AS backward_extrem\n                ON edge.prev_event_id = backward_extrem.event_id\n                /**\n                 * We use this info to make sure we don't retry to use a backfill point\n                 * if we've already attempted to backfill from it recently.\n                 */\n                LEFT JOIN event_failed_pull_attempts AS failed_backfill_attempt_info\n                ON\n                    failed_backfill_attempt_info.room_id = backward_extrem.room_id\n                    AND failed_backfill_attempt_info.event_id = backward_extrem.event_id\n                WHERE\n                    backward_extrem.room_id = ?\n                    /* We only care about non-state edges because we used to use\n                     * `event_edges` for two different sorts of "edges" (the current\n                     * event DAG, but also a link to the previous state, for state\n                     * events). These legacy state event edges can be distinguished by\n                     * `is_state` and are removed from the codebase and schema but\n                     * because the schema change is in a background update, it's not\n                     * necessarily safe to assume that it will have been completed.\n                     */\n                    AND edge.is_state is FALSE\n                    /**\n                     * We only want backwards extremities that are older than or at\n                     * the same position of the given `current_depth` (where older\n                     * means less than the given depth) because we're looking backwards\n                     * from the `current_depth` when backfilling.\n                     *\n                     *                         current_depth (ignore events that come after this, ignore 2-4)\n                     *                         |\n                     *                         ▼\n                     * <oldest-in-time> [0]<--[1]<--[2]<--[3]<--[4] <newest-in-time>\n                     */\n                    AND event.depth <= ? /* current_depth */\n                    /**\n                     * Exponential back-off (up to the upper bound) so we don't retry the\n                     * same backfill point over and over. ex. 2hr, 4hr, 8hr, 16hr, etc.\n                     *\n                     * We use `1 << n` as a power of 2 equivalent for compatibility\n                     * with older SQLites. The left shift equivalent only works with\n                     * powers of 2 because left shift is a binary operation (base-2).\n                     * Otherwise, we would use `power(2, n)` or the power operator, `2^n`.\n                     */\n                    AND (\n                        failed_backfill_attempt_info.event_id IS NULL\n                        OR ? /* current_time */ >= failed_backfill_attempt_info.last_attempt_ts + (\n                            (1 << {least_function}(failed_backfill_attempt_info.num_attempts, ? /* max doubling steps */))\n                            * ? /* step */\n                        )\n                    )\n                /**\n                 * Sort from highest (closest to the `current_depth`) to the lowest depth\n                 * because the closest are most relevant to backfill from first.\n                 * Then tie-break on alphabetical order of the event_ids so we get a\n                 * consistent ordering which is nice when asserting things in tests.\n                 */\n                ORDER BY event.depth DESC, backward_extrem.event_id DESC\n                LIMIT ?\n            """
            txn.execute(sql, (room_id, current_depth, self._clock.time_msec(), BACKFILL_EVENT_EXPONENTIAL_BACKOFF_MAXIMUM_DOUBLING_STEPS, BACKFILL_EVENT_EXPONENTIAL_BACKOFF_STEP_MILLISECONDS, limit))
            return cast(List[Tuple[str, int]], txn.fetchall())
        return await self.db_pool.runInteraction('get_backfill_points_in_room', get_backfill_points_in_room_txn, room_id)

    async def get_max_depth_of(self, event_ids: Collection[str]) -> Tuple[Optional[str], int]:
        """Returns the event ID and depth for the event that has the max depth from a set of event IDs

        Args:
            event_ids: The event IDs to calculate the max depth of.
        """
        rows = cast(List[Tuple[str, int]], await self.db_pool.simple_select_many_batch(table='events', column='event_id', iterable=event_ids, retcols=('event_id', 'depth'), desc='get_max_depth_of'))
        if not rows:
            return (None, 0)
        else:
            max_depth_event_id = ''
            current_max_depth = 0
            for (event_id, depth) in rows:
                if depth > current_max_depth:
                    max_depth_event_id = event_id
                    current_max_depth = depth
            return (max_depth_event_id, current_max_depth)

    async def get_min_depth_of(self, event_ids: List[str]) -> Tuple[Optional[str], int]:
        """Returns the event ID and depth for the event that has the min depth from a set of event IDs

        Args:
            event_ids: The event IDs to calculate the max depth of.
        """
        rows = cast(List[Tuple[str, int]], await self.db_pool.simple_select_many_batch(table='events', column='event_id', iterable=event_ids, retcols=('event_id', 'depth'), desc='get_min_depth_of'))
        if not rows:
            return (None, 0)
        else:
            min_depth_event_id = ''
            current_min_depth = MAX_DEPTH
            for (event_id, depth) in rows:
                if depth < current_min_depth:
                    min_depth_event_id = event_id
                    current_min_depth = depth
            return (min_depth_event_id, current_min_depth)

    async def get_prev_events_for_room(self, room_id: str) -> List[str]:
        """
        Gets a subset of the current forward extremities in the given room.

        Limits the result to 10 extremities, so that we can avoid creating
        events which refer to hundreds of prev_events.

        Args:
            room_id: room_id

        Returns:
            The event ids of the forward extremities.

        """
        return await self.db_pool.runInteraction('get_prev_events_for_room', self._get_prev_events_for_room_txn, room_id)

    def _get_prev_events_for_room_txn(self, txn: LoggingTransaction, room_id: str) -> List[str]:
        if False:
            print('Hello World!')
        sql = '\n            SELECT e.event_id FROM event_forward_extremities AS f\n            INNER JOIN events AS e USING (event_id)\n            WHERE f.room_id = ?\n            ORDER BY e.depth DESC\n            LIMIT 10\n        '
        txn.execute(sql, (room_id,))
        return [row[0] for row in txn]

    async def get_rooms_with_many_extremities(self, min_count: int, limit: int, room_id_filter: Iterable[str]) -> List[str]:
        """Get the top rooms with at least N extremities.

        Args:
            min_count: The minimum number of extremities
            limit: The maximum number of rooms to return.
            room_id_filter: room_ids to exclude from the results

        Returns:
            At most `limit` room IDs that have at least `min_count` extremities,
            sorted by extremity count.
        """

        def _get_rooms_with_many_extremities_txn(txn: LoggingTransaction) -> List[str]:
            if False:
                return 10
            where_clause = '1=1'
            if room_id_filter:
                where_clause = 'room_id NOT IN (%s)' % (','.join(('?' for _ in room_id_filter)),)
            sql = '\n                SELECT room_id FROM event_forward_extremities\n                WHERE %s\n                GROUP BY room_id\n                HAVING count(*) > ?\n                ORDER BY count(*) DESC\n                LIMIT ?\n            ' % (where_clause,)
            query_args = list(itertools.chain(room_id_filter, [min_count, limit]))
            txn.execute(sql, query_args)
            return [room_id for (room_id,) in txn]
        return await self.db_pool.runInteraction('get_rooms_with_many_extremities', _get_rooms_with_many_extremities_txn)

    @cached(max_entries=5000, iterable=True)
    async def get_latest_event_ids_in_room(self, room_id: str) -> FrozenSet[str]:
        event_ids = await self.db_pool.simple_select_onecol(table='event_forward_extremities', keyvalues={'room_id': room_id}, retcol='event_id', desc='get_latest_event_ids_in_room')
        return frozenset(event_ids)

    async def get_min_depth(self, room_id: str) -> Optional[int]:
        """For the given room, get the minimum depth we have seen for it."""
        return await self.db_pool.runInteraction('get_min_depth', self._get_min_depth_interaction, room_id)

    def _get_min_depth_interaction(self, txn: LoggingTransaction, room_id: str) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        min_depth = self.db_pool.simple_select_one_onecol_txn(txn, table='room_depth', keyvalues={'room_id': room_id}, retcol='min_depth', allow_none=True)
        return int(min_depth) if min_depth is not None else None

    async def have_room_forward_extremities_changed_since(self, room_id: str, stream_ordering: int) -> bool:
        """Check if the forward extremities in a room have changed since the
        given stream ordering

        Throws a StoreError if we have since purged the index for
        stream_orderings from that point.
        """
        assert self.stream_ordering_month_ago is not None
        if stream_ordering <= self.stream_ordering_month_ago:
            raise StoreError(400, f'stream_ordering too old {stream_ordering}')
        sql = '\n            SELECT 1 FROM stream_ordering_to_exterm\n            WHERE stream_ordering > ? AND room_id = ?\n            LIMIT 1\n        '

        def have_room_forward_extremities_changed_since_txn(txn: LoggingTransaction) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            txn.execute(sql, (stream_ordering, room_id))
            return txn.fetchone() is not None
        return await self.db_pool.runInteraction('have_room_forward_extremities_changed_since', have_room_forward_extremities_changed_since_txn)

    @cancellable
    async def get_forward_extremities_for_room_at_stream_ordering(self, room_id: str, stream_ordering: int) -> Sequence[str]:
        """For a given room_id and stream_ordering, return the forward
        extremeties of the room at that point in "time".

        Throws a StoreError if we have since purged the index for
        stream_orderings from that point.

        Args:
            room_id:
            stream_ordering:

        Returns:
            A list of event_ids
        """
        last_change = self._events_stream_cache.get_max_pos_of_last_change(room_id)
        last_change = max(self._stream_order_on_start, last_change)
        assert self.stream_ordering_month_ago is not None
        if last_change > self.stream_ordering_month_ago:
            stream_ordering = min(last_change, stream_ordering)
        return await self._get_forward_extremeties_for_room(room_id, stream_ordering)

    @cached(max_entries=5000, num_args=2)
    async def _get_forward_extremeties_for_room(self, room_id: str, stream_ordering: int) -> Sequence[str]:
        """For a given room_id and stream_ordering, return the forward
        extremeties of the room at that point in "time".

        Throws a StoreError if we have since purged the index for
        stream_orderings from that point.
        """
        assert self.stream_ordering_month_ago is not None
        if stream_ordering <= self.stream_ordering_month_ago:
            raise StoreError(400, 'stream_ordering too old %s' % (stream_ordering,))
        sql = '\n                SELECT event_id FROM stream_ordering_to_exterm\n                INNER JOIN (\n                    SELECT room_id, MAX(stream_ordering) AS stream_ordering\n                    FROM stream_ordering_to_exterm\n                    WHERE stream_ordering <= ? GROUP BY room_id\n                ) AS rms USING (room_id, stream_ordering)\n                WHERE room_id = ?\n        '

        def get_forward_extremeties_for_room_txn(txn: LoggingTransaction) -> List[str]:
            if False:
                print('Hello World!')
            txn.execute(sql, (stream_ordering, room_id))
            return [event_id for (event_id,) in txn]
        event_ids = await self.db_pool.runInteraction('get_forward_extremeties_for_room', get_forward_extremeties_for_room_txn)
        if not event_ids:
            raise StoreError(400, 'stream_ordering too old %s' % (stream_ordering,))
        return event_ids

    def _get_connected_prev_event_backfill_results_txn(self, txn: LoggingTransaction, event_id: str, limit: int) -> List[BackfillQueueNavigationItem]:
        if False:
            return 10
        "\n        Find any events connected by prev_event the specified event_id.\n\n        Args:\n            txn: The database transaction to use\n            event_id: The event ID to navigate from\n            limit: Max number of event ID's to query for and return\n\n        Returns:\n            List of prev events that the backfill queue can process\n        "
        connected_prev_event_query = "\n            SELECT depth, stream_ordering, prev_event_id, events.type FROM event_edges\n            /* Get the depth and stream_ordering of the prev_event_id from the events table */\n            INNER JOIN events\n            ON prev_event_id = events.event_id\n\n            /* exclude outliers from the results (we don't have the state, so cannot\n             * verify if the requesting server can see them).\n             */\n            WHERE NOT events.outlier\n\n            /* Look for an edge which matches the given event_id */\n            AND event_edges.event_id = ? AND NOT event_edges.is_state\n\n            /* Because we can have many events at the same depth,\n            * we want to also tie-break and sort on stream_ordering */\n            ORDER BY depth DESC, stream_ordering DESC\n            LIMIT ?\n        "
        txn.execute(connected_prev_event_query, (event_id, limit))
        return [BackfillQueueNavigationItem(depth=row[0], stream_ordering=row[1], event_id=row[2], type=row[3]) for row in txn]

    async def get_backfill_events(self, room_id: str, seed_event_id_list: List[str], limit: int) -> List[EventBase]:
        """Get a list of Events for a given topic that occurred before (and
        including) the events in seed_event_id_list. Return a list of max size `limit`

        Args:
            room_id
            seed_event_id_list
            limit
        """
        event_ids = await self.db_pool.runInteraction('get_backfill_events', self._get_backfill_events, room_id, seed_event_id_list, limit)
        events = await self.get_events_as_list(event_ids)
        return sorted(events, key=lambda e: (-e.depth, -e.internal_metadata.stream_ordering))

    def _get_backfill_events(self, txn: LoggingTransaction, room_id: str, seed_event_id_list: List[str], limit: int) -> Set[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        We want to make sure that we do a breadth-first, "depth" ordered search.\n        We also handle navigating historical branches of history connected by\n        insertion and batch events.\n        '
        logger.debug('_get_backfill_events(room_id=%s): seeding backfill with seed_event_id_list=%s limit=%s', room_id, seed_event_id_list, limit)
        event_id_results: Set[str] = set()
        queue: 'PriorityQueue[Tuple[int, int, str, str]]' = PriorityQueue()
        for seed_event_id in seed_event_id_list:
            event_lookup_result = self.db_pool.simple_select_one_txn(txn, table='events', keyvalues={'event_id': seed_event_id, 'room_id': room_id}, retcols=('type', 'depth', 'stream_ordering'), allow_none=True)
            if event_lookup_result is not None:
                (event_type, depth, stream_ordering) = event_lookup_result
                logger.debug('_get_backfill_events(room_id=%s): seed_event_id=%s depth=%s stream_ordering=%s type=%s', room_id, seed_event_id, depth, stream_ordering, event_type)
                if depth:
                    queue.put((-depth, -stream_ordering, seed_event_id, event_type))
        while not queue.empty() and len(event_id_results) < limit:
            try:
                (_, _, event_id, event_type) = queue.get_nowait()
            except Empty:
                break
            if event_id in event_id_results:
                continue
            event_id_results.add(event_id)
            connected_prev_event_backfill_results = self._get_connected_prev_event_backfill_results_txn(txn, event_id, limit - len(event_id_results))
            logger.debug('_get_backfill_events(room_id=%s): connected_prev_event_backfill_results=%s', room_id, connected_prev_event_backfill_results)
            for connected_prev_event_backfill_item in connected_prev_event_backfill_results:
                if connected_prev_event_backfill_item.event_id not in event_id_results:
                    queue.put((-connected_prev_event_backfill_item.depth, -connected_prev_event_backfill_item.stream_ordering, connected_prev_event_backfill_item.event_id, connected_prev_event_backfill_item.type))
        return event_id_results

    @trace
    async def record_event_failed_pull_attempt(self, room_id: str, event_id: str, cause: str) -> None:
        """
        Record when we fail to pull an event over federation.

        This information allows us to be more intelligent when we decide to
        retry (we don't need to fail over and over) and we can process that
        event in the background so we don't block on it each time.

        Args:
            room_id: The room where the event failed to pull from
            event_id: The event that failed to be fetched or processed
            cause: The error message or reason that we failed to pull the event
        """
        logger.debug('record_event_failed_pull_attempt room_id=%s, event_id=%s, cause=%s', room_id, event_id, cause)
        await self.db_pool.runInteraction('record_event_failed_pull_attempt', self._record_event_failed_pull_attempt_upsert_txn, room_id, event_id, cause, db_autocommit=True)

    def _record_event_failed_pull_attempt_upsert_txn(self, txn: LoggingTransaction, room_id: str, event_id: str, cause: str) -> None:
        if False:
            print('Hello World!')
        sql = '\n            INSERT INTO event_failed_pull_attempts (\n                room_id, event_id, num_attempts, last_attempt_ts, last_cause\n            )\n                VALUES (?, ?, ?, ?, ?)\n            ON CONFLICT (room_id, event_id) DO UPDATE SET\n                num_attempts=event_failed_pull_attempts.num_attempts + 1,\n                last_attempt_ts=EXCLUDED.last_attempt_ts,\n                last_cause=EXCLUDED.last_cause;\n        '
        txn.execute(sql, (room_id, event_id, 1, self._clock.time_msec(), cause))

    @trace
    async def get_event_ids_with_failed_pull_attempts(self, event_ids: StrCollection) -> Set[str]:
        """
        Filter the given list of `event_ids` and return events which have any failed
        pull attempts.

        Args:
            event_ids: A list of events to filter down.

        Returns:
            A filtered down list of `event_ids` that have previous failed pull attempts.
        """
        rows = cast(List[Tuple[str]], await self.db_pool.simple_select_many_batch(table='event_failed_pull_attempts', column='event_id', iterable=event_ids, keyvalues={}, retcols=('event_id',), desc='get_event_ids_with_failed_pull_attempts'))
        return {row[0] for row in rows}

    @trace
    async def get_event_ids_to_not_pull_from_backoff(self, room_id: str, event_ids: Collection[str]) -> Dict[str, int]:
        """
        Filter down the events to ones that we've failed to pull before recently. Uses
        exponential backoff.

        Args:
            room_id: The room that the events belong to
            event_ids: A list of events to filter down

        Returns:
            A dictionary of event_ids that should not be attempted to be pulled and the
            next timestamp at which we may try pulling them again.
        """
        event_failed_pull_attempts = cast(List[Tuple[str, int, int]], await self.db_pool.simple_select_many_batch(table='event_failed_pull_attempts', column='event_id', iterable=event_ids, keyvalues={}, retcols=('event_id', 'last_attempt_ts', 'num_attempts'), desc='get_event_ids_to_not_pull_from_backoff'))
        current_time = self._clock.time_msec()
        event_ids_with_backoff = {}
        for (event_id, last_attempt_ts, num_attempts) in event_failed_pull_attempts:
            backoff_end_time = last_attempt_ts + 2 ** min(num_attempts, BACKFILL_EVENT_EXPONENTIAL_BACKOFF_MAXIMUM_DOUBLING_STEPS) * BACKFILL_EVENT_EXPONENTIAL_BACKOFF_STEP_MILLISECONDS
            if current_time < backoff_end_time:
                event_ids_with_backoff[event_id] = backoff_end_time
        return event_ids_with_backoff

    async def get_missing_events(self, room_id: str, earliest_events: List[str], latest_events: List[str], limit: int) -> List[EventBase]:
        ids = await self.db_pool.runInteraction('get_missing_events', self._get_missing_events, room_id, earliest_events, latest_events, limit)
        return await self.get_events_as_list(ids)

    def _get_missing_events(self, txn: LoggingTransaction, room_id: str, earliest_events: List[str], latest_events: List[str], limit: int) -> List[str]:
        if False:
            i = 10
            return i + 15
        seen_events = set(earliest_events)
        front = set(latest_events) - seen_events
        event_results: List[str] = []
        query = 'SELECT prev_event_id FROM event_edges WHERE event_id = ? AND NOT is_state LIMIT ?'
        while front and len(event_results) < limit:
            new_front = set()
            for event_id in front:
                txn.execute(query, (event_id, limit - len(event_results)))
                new_results = {t[0] for t in txn} - seen_events
                new_front |= new_results
                seen_events |= new_results
                event_results.extend(new_results)
            front = new_front
        event_results.reverse()
        return event_results

    @trace
    @tag_args
    async def get_successor_events(self, event_id: str) -> List[str]:
        """Fetch all events that have the given event as a prev event

        Args:
            event_id: The event to search for as a prev_event.
        """
        return await self.db_pool.simple_select_onecol(table='event_edges', keyvalues={'prev_event_id': event_id}, retcol='event_id', desc='get_successor_events')

    @wrap_as_background_process('delete_old_forward_extrem_cache')
    async def _delete_old_forward_extrem_cache(self) -> None:

        def _delete_old_forward_extrem_cache_txn(txn: LoggingTransaction) -> None:
            if False:
                return 10
            sql = '\n                DELETE FROM stream_ordering_to_exterm\n                WHERE stream_ordering < ?\n            '
            txn.execute(sql, (self.stream_ordering_month_ago,))
        await self.db_pool.runInteraction('_delete_old_forward_extrem_cache', _delete_old_forward_extrem_cache_txn)

    async def insert_received_event_to_staging(self, origin: str, event: EventBase) -> None:
        """Insert a newly received event from federation into the staging area."""
        await self.db_pool.simple_upsert(table='federation_inbound_events_staging', keyvalues={'origin': origin, 'event_id': event.event_id}, values={}, insertion_values={'room_id': event.room_id, 'received_ts': self._clock.time_msec(), 'event_json': json_encoder.encode(event.get_dict()), 'internal_metadata': json_encoder.encode(event.internal_metadata.get_dict())}, desc='insert_received_event_to_staging')

    async def remove_received_event_from_staging(self, origin: str, event_id: str) -> Optional[int]:
        """Remove the given event from the staging area.

        Returns:
            The received_ts of the row that was deleted, if any.
        """
        if self.db_pool.engine.supports_returning:

            def _remove_received_event_from_staging_txn(txn: LoggingTransaction) -> Optional[int]:
                if False:
                    i = 10
                    return i + 15
                sql = '\n                    DELETE FROM federation_inbound_events_staging\n                    WHERE origin = ? AND event_id = ?\n                    RETURNING received_ts\n                '
                txn.execute(sql, (origin, event_id))
                row = cast(Optional[Tuple[int]], txn.fetchone())
                if row is None:
                    return None
                return row[0]
            return await self.db_pool.runInteraction('remove_received_event_from_staging', _remove_received_event_from_staging_txn, db_autocommit=True)
        else:

            def _remove_received_event_from_staging_txn(txn: LoggingTransaction) -> Optional[int]:
                if False:
                    i = 10
                    return i + 15
                received_ts = self.db_pool.simple_select_one_onecol_txn(txn, table='federation_inbound_events_staging', keyvalues={'origin': origin, 'event_id': event_id}, retcol='received_ts', allow_none=True)
                self.db_pool.simple_delete_txn(txn, table='federation_inbound_events_staging', keyvalues={'origin': origin, 'event_id': event_id})
                return received_ts
            return await self.db_pool.runInteraction('remove_received_event_from_staging', _remove_received_event_from_staging_txn)

    async def get_next_staged_event_id_for_room(self, room_id: str) -> Optional[Tuple[str, str]]:
        """
        Get the next event ID in the staging area for the given room.

        Returns:
            Tuple of the `origin` and `event_id`
        """

        def _get_next_staged_event_id_for_room_txn(txn: LoggingTransaction) -> Optional[Tuple[str, str]]:
            if False:
                print('Hello World!')
            sql = '\n                SELECT origin, event_id\n                FROM federation_inbound_events_staging\n                WHERE room_id = ?\n                ORDER BY received_ts ASC\n                LIMIT 1\n            '
            txn.execute(sql, (room_id,))
            return cast(Optional[Tuple[str, str]], txn.fetchone())
        return await self.db_pool.runInteraction('get_next_staged_event_id_for_room', _get_next_staged_event_id_for_room_txn)

    async def get_next_staged_event_for_room(self, room_id: str, room_version: RoomVersion) -> Optional[Tuple[str, EventBase]]:
        """Get the next event in the staging area for the given room."""

        def _get_next_staged_event_for_room_txn(txn: LoggingTransaction) -> Optional[Tuple[str, str, str]]:
            if False:
                while True:
                    i = 10
            sql = '\n                SELECT event_json, internal_metadata, origin\n                FROM federation_inbound_events_staging\n                WHERE room_id = ?\n                ORDER BY received_ts ASC\n                LIMIT 1\n            '
            txn.execute(sql, (room_id,))
            return cast(Optional[Tuple[str, str, str]], txn.fetchone())
        row = await self.db_pool.runInteraction('get_next_staged_event_for_room', _get_next_staged_event_for_room_txn)
        if not row:
            return None
        event_d = db_to_json(row[0])
        internal_metadata_d = db_to_json(row[1])
        origin = row[2]
        event = make_event_from_dict(event_dict=event_d, room_version=room_version, internal_metadata_dict=internal_metadata_d)
        return (origin, event)

    async def prune_staged_events_in_room(self, room_id: str, room_version: RoomVersion) -> bool:
        """Checks if there are lots of staged events for the room, and if so
        prune them down.

        Returns:
            Whether any events were pruned
        """
        count = await self.db_pool.simple_select_one_onecol(table='federation_inbound_events_staging', keyvalues={'room_id': room_id}, retcol='COUNT(*)', desc='prune_staged_events_in_room_count')
        if count < 100:
            return False
        rows = cast(List[Tuple[str, str]], await self.db_pool.simple_select_list(table='federation_inbound_events_staging', keyvalues={'room_id': room_id}, retcols=('event_id', 'event_json'), desc='prune_staged_events_in_room_fetch'))
        referenced_events: Set[str] = set()
        seen_events: Set[str] = set()
        for (event_id, event_json) in rows:
            seen_events.add(event_id)
            event_d = db_to_json(event_json)
            prev_events = event_d.get('prev_events', [])
            if not isinstance(prev_events, list):
                logger.info('Invalid prev_events for %s', event_id)
                continue
            if room_version.event_format == EventFormatVersions.ROOM_V1_V2:
                for prev_event_tuple in prev_events:
                    if not isinstance(prev_event_tuple, list) or len(prev_event_tuple) != 2:
                        logger.info('Invalid prev_events for %s', event_id)
                        break
                    prev_event_id = prev_event_tuple[0]
                    if not isinstance(prev_event_id, str):
                        logger.info('Invalid prev_events for %s', event_id)
                        break
                    referenced_events.add(prev_event_id)
            else:
                for prev_event_id in prev_events:
                    if not isinstance(prev_event_id, str):
                        logger.info('Invalid prev_events for %s', event_id)
                        break
                    referenced_events.add(prev_event_id)
        to_delete = referenced_events & seen_events
        if not to_delete:
            return False
        pdus_pruned_from_federation_queue.inc(len(to_delete))
        logger.info('Pruning %d events in room %s from federation queue', len(to_delete), room_id)
        await self.db_pool.simple_delete_many(table='federation_inbound_events_staging', keyvalues={'room_id': room_id}, iterable=to_delete, column='event_id', desc='prune_staged_events_in_room_delete')
        return True

    async def get_all_rooms_with_staged_incoming_events(self) -> List[str]:
        """Get the room IDs of all events currently staged."""
        return await self.db_pool.simple_select_onecol(table='federation_inbound_events_staging', keyvalues={}, retcol='DISTINCT room_id', desc='get_all_rooms_with_staged_incoming_events')

    @wrap_as_background_process('_get_stats_for_federation_staging')
    async def _get_stats_for_federation_staging(self) -> None:
        """Update the prometheus metrics for the inbound federation staging area."""

        def _get_stats_for_federation_staging_txn(txn: LoggingTransaction) -> Tuple[int, int]:
            if False:
                for i in range(10):
                    print('nop')
            txn.execute('SELECT count(*) FROM federation_inbound_events_staging')
            (count,) = cast(Tuple[int], txn.fetchone())
            txn.execute('SELECT min(received_ts) FROM federation_inbound_events_staging')
            (received_ts,) = cast(Tuple[Optional[int]], txn.fetchone())
            age = 0
            if received_ts is not None:
                age = self._clock.time_msec() - received_ts
            return (count, age)
        (count, age) = await self.db_pool.runInteraction('_get_stats_for_federation_staging', _get_stats_for_federation_staging_txn)
        number_pdus_in_federation_queue.set(count)
        oldest_pdu_in_federation_staging.set(age)

class EventFederationStore(EventFederationWorkerStore):
    """Responsible for storing and serving up the various graphs associated
    with an event. Including the main event graph and the auth chains for an
    event.

    Also has methods for getting the front (latest) and back (oldest) edges
    of the event graphs. These are used to generate the parents for new events
    and backfilling from another server respectively.
    """
    EVENT_AUTH_STATE_ONLY = 'event_auth_state_only'

    def __init__(self, database: DatabasePool, db_conn: LoggingDatabaseConnection, hs: 'HomeServer'):
        if False:
            i = 10
            return i + 15
        super().__init__(database, db_conn, hs)
        self.db_pool.updates.register_background_update_handler(self.EVENT_AUTH_STATE_ONLY, self._background_delete_non_state_event_auth)

    async def clean_room_for_join(self, room_id: str) -> None:
        await self.db_pool.runInteraction('clean_room_for_join', self._clean_room_for_join_txn, room_id)

    def _clean_room_for_join_txn(self, txn: LoggingTransaction, room_id: str) -> None:
        if False:
            return 10
        query = 'DELETE FROM event_forward_extremities WHERE room_id = ?'
        txn.execute(query, (room_id,))
        txn.call_after(self.get_latest_event_ids_in_room.invalidate, (room_id,))

    async def _background_delete_non_state_event_auth(self, progress: JsonDict, batch_size: int) -> int:

        def delete_event_auth(txn: LoggingTransaction) -> bool:
            if False:
                i = 10
                return i + 15
            target_min_stream_id = progress.get('target_min_stream_id_inclusive')
            max_stream_id = progress.get('max_stream_id_exclusive')
            if not target_min_stream_id or not max_stream_id:
                txn.execute('SELECT COALESCE(MIN(stream_ordering), 0) FROM events')
                rows = txn.fetchall()
                target_min_stream_id = rows[0][0]
                txn.execute('SELECT COALESCE(MAX(stream_ordering), 0) FROM events')
                rows = txn.fetchall()
                max_stream_id = rows[0][0]
            min_stream_id = max_stream_id - batch_size
            sql = '\n                DELETE FROM event_auth\n                WHERE event_id IN (\n                    SELECT event_id FROM events\n                    LEFT JOIN state_events AS se USING (room_id, event_id)\n                    WHERE ? <= stream_ordering AND stream_ordering < ?\n                        AND se.state_key IS null\n                )\n            '
            txn.execute(sql, (min_stream_id, max_stream_id))
            new_progress = {'target_min_stream_id_inclusive': target_min_stream_id, 'max_stream_id_exclusive': min_stream_id}
            self.db_pool.updates._background_update_progress_txn(txn, self.EVENT_AUTH_STATE_ONLY, new_progress)
            return min_stream_id >= target_min_stream_id
        result = await self.db_pool.runInteraction(self.EVENT_AUTH_STATE_ONLY, delete_event_auth)
        if not result:
            await self.db_pool.updates._end_background_update(self.EVENT_AUTH_STATE_ONLY)
        return batch_size