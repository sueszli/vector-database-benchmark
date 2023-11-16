import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, cast
import attr
from synapse.api.constants import EventContentFields, RelationTypes
from synapse.api.room_versions import KNOWN_ROOM_VERSIONS
from synapse.events import make_event_from_dict
from synapse.storage._base import SQLBaseStore, db_to_json, make_in_list_sql_clause
from synapse.storage.database import DatabasePool, LoggingDatabaseConnection, LoggingTransaction, make_tuple_comparison_clause
from synapse.storage.databases.main.events import PersistEventsStore
from synapse.storage.types import Cursor
from synapse.types import JsonDict, StrCollection
if TYPE_CHECKING:
    from synapse.server import HomeServer
logger = logging.getLogger(__name__)
_REPLACE_STREAM_ORDERING_SQL_COMMANDS = ('UPDATE events SET stream_ordering2 = stream_ordering WHERE stream_ordering2 IS NULL', 'DROP RULE populate_stream_ordering2 ON events', 'ALTER TABLE events DROP COLUMN stream_ordering', 'ALTER TABLE events RENAME COLUMN stream_ordering2 TO stream_ordering', 'ALTER INDEX event_contains_url_index2 RENAME TO event_contains_url_index', 'ALTER INDEX events_order_room2 RENAME TO events_order_room', 'ALTER INDEX events_room_stream2 RENAME TO events_room_stream', 'ALTER INDEX events_ts2 RENAME TO events_ts')

class _BackgroundUpdates:
    EVENT_ORIGIN_SERVER_TS_NAME = 'event_origin_server_ts'
    EVENT_FIELDS_SENDER_URL_UPDATE_NAME = 'event_fields_sender_url'
    DELETE_SOFT_FAILED_EXTREMITIES = 'delete_soft_failed_extremities'
    POPULATE_STREAM_ORDERING2 = 'populate_stream_ordering2'
    INDEX_STREAM_ORDERING2 = 'index_stream_ordering2'
    INDEX_STREAM_ORDERING2_CONTAINS_URL = 'index_stream_ordering2_contains_url'
    INDEX_STREAM_ORDERING2_ROOM_ORDER = 'index_stream_ordering2_room_order'
    INDEX_STREAM_ORDERING2_ROOM_STREAM = 'index_stream_ordering2_room_stream'
    INDEX_STREAM_ORDERING2_TS = 'index_stream_ordering2_ts'
    REPLACE_STREAM_ORDERING_COLUMN = 'replace_stream_ordering_column'
    EVENT_EDGES_DROP_INVALID_ROWS = 'event_edges_drop_invalid_rows'
    EVENT_EDGES_REPLACE_INDEX = 'event_edges_replace_index'
    EVENTS_POPULATE_STATE_KEY_REJECTIONS = 'events_populate_state_key_rejections'
    EVENTS_JUMP_TO_DATE_INDEX = 'events_jump_to_date_index'

@attr.s(slots=True, frozen=True, auto_attribs=True)
class _CalculateChainCover:
    """Return value for _calculate_chain_cover_txn."""
    room_id: str
    depth: int
    stream: int
    processed_count: int
    finished_room_map: Dict[str, Tuple[int, int]]

class EventsBackgroundUpdatesStore(SQLBaseStore):

    def __init__(self, database: DatabasePool, db_conn: LoggingDatabaseConnection, hs: 'HomeServer'):
        if False:
            while True:
                i = 10
        super().__init__(database, db_conn, hs)
        self.db_pool.updates.register_background_update_handler(_BackgroundUpdates.EVENT_ORIGIN_SERVER_TS_NAME, self._background_reindex_origin_server_ts)
        self.db_pool.updates.register_background_update_handler(_BackgroundUpdates.EVENT_FIELDS_SENDER_URL_UPDATE_NAME, self._background_reindex_fields_sender)
        self.db_pool.updates.register_background_index_update('event_contains_url_index', index_name='event_contains_url_index', table='events', columns=['room_id', 'topological_ordering', 'stream_ordering'], where_clause='contains_url = true AND outlier = false')
        self.db_pool.updates.register_background_index_update('event_search_event_id_idx', index_name='event_search_event_id_idx', table='event_search', columns=['event_id'], unique=True, psql_only=True)
        self.db_pool.updates.register_background_update_handler(_BackgroundUpdates.DELETE_SOFT_FAILED_EXTREMITIES, self._cleanup_extremities_bg_update)
        self.db_pool.updates.register_background_update_handler('redactions_received_ts', self._redactions_received_ts)
        self.db_pool.updates.register_background_index_update('event_fix_redactions_bytes_create_index', index_name='redactions_censored_redacts', table='redactions', columns=['redacts'], where_clause='have_censored')
        self.db_pool.updates.register_background_update_handler('event_fix_redactions_bytes', self._event_fix_redactions_bytes)
        self.db_pool.updates.register_background_update_handler('event_store_labels', self._event_store_labels)
        self.db_pool.updates.register_background_index_update('redactions_have_censored_ts_idx', index_name='redactions_have_censored_ts', table='redactions', columns=['received_ts'], where_clause='NOT have_censored')
        self.db_pool.updates.register_background_index_update('users_have_local_media', index_name='users_have_local_media', table='local_media_repository', columns=['user_id', 'created_ts'])
        self.db_pool.updates.register_background_update_handler('rejected_events_metadata', self._rejected_events_metadata)
        self.db_pool.updates.register_background_update_handler('chain_cover', self._chain_cover_index)
        self.db_pool.updates.register_background_update_handler('purged_chain_cover', self._purged_chain_cover_index)
        self.db_pool.updates.register_background_update_handler('event_arbitrary_relations', self._event_arbitrary_relations)
        self.db_pool.updates.register_background_update_handler(_BackgroundUpdates.POPULATE_STREAM_ORDERING2, self._background_populate_stream_ordering2)
        self.db_pool.updates.register_background_index_update(_BackgroundUpdates.INDEX_STREAM_ORDERING2, index_name='events_stream_ordering', table='events', columns=['stream_ordering2'], unique=True)
        self.db_pool.updates.register_background_index_update(_BackgroundUpdates.INDEX_STREAM_ORDERING2_CONTAINS_URL, index_name='event_contains_url_index2', table='events', columns=['room_id', 'topological_ordering', 'stream_ordering2'], where_clause='contains_url = true AND outlier = false')
        self.db_pool.updates.register_background_index_update(_BackgroundUpdates.INDEX_STREAM_ORDERING2_ROOM_ORDER, index_name='events_order_room2', table='events', columns=['room_id', 'topological_ordering', 'stream_ordering2'])
        self.db_pool.updates.register_background_index_update(_BackgroundUpdates.INDEX_STREAM_ORDERING2_ROOM_STREAM, index_name='events_room_stream2', table='events', columns=['room_id', 'stream_ordering2'])
        self.db_pool.updates.register_background_index_update(_BackgroundUpdates.INDEX_STREAM_ORDERING2_TS, index_name='events_ts2', table='events', columns=['origin_server_ts', 'stream_ordering2'])
        self.db_pool.updates.register_background_update_handler(_BackgroundUpdates.REPLACE_STREAM_ORDERING_COLUMN, self._background_replace_stream_ordering_column)
        self.db_pool.updates.register_background_update_handler(_BackgroundUpdates.EVENT_EDGES_DROP_INVALID_ROWS, self._background_drop_invalid_event_edges_rows)
        self.db_pool.updates.register_background_index_update(_BackgroundUpdates.EVENT_EDGES_REPLACE_INDEX, index_name='event_edges_event_id_prev_event_id_idx', table='event_edges', columns=['event_id', 'prev_event_id'], unique=True, replaces_index='ev_edges_id')
        self.db_pool.updates.register_background_update_handler(_BackgroundUpdates.EVENTS_POPULATE_STATE_KEY_REJECTIONS, self._background_events_populate_state_key_rejections)
        self.db_pool.updates.register_background_index_update(_BackgroundUpdates.EVENTS_JUMP_TO_DATE_INDEX, index_name='events_jump_to_date_idx', table='events', columns=['room_id', 'origin_server_ts'], where_clause='NOT outlier')

    async def _background_reindex_fields_sender(self, progress: JsonDict, batch_size: int) -> int:
        target_min_stream_id = progress['target_min_stream_id_inclusive']
        max_stream_id = progress['max_stream_id_exclusive']
        rows_inserted = progress.get('rows_inserted', 0)

        def reindex_txn(txn: LoggingTransaction) -> int:
            if False:
                i = 10
                return i + 15
            sql = 'SELECT stream_ordering, event_id, json FROM events INNER JOIN event_json USING (event_id) WHERE ? <= stream_ordering AND stream_ordering < ? ORDER BY stream_ordering DESC LIMIT ?'
            txn.execute(sql, (target_min_stream_id, max_stream_id, batch_size))
            rows = txn.fetchall()
            if not rows:
                return 0
            min_stream_id = rows[-1][0]
            update_rows = []
            for row in rows:
                try:
                    event_id = row[1]
                    event_json = db_to_json(row[2])
                    sender = event_json['sender']
                    content = event_json['content']
                    contains_url = 'url' in content
                    if contains_url:
                        contains_url &= isinstance(content['url'], str)
                except (KeyError, AttributeError):
                    continue
                update_rows.append((sender, contains_url, event_id))
            sql = 'UPDATE events SET sender = ?, contains_url = ? WHERE event_id = ?'
            txn.execute_batch(sql, update_rows)
            progress = {'target_min_stream_id_inclusive': target_min_stream_id, 'max_stream_id_exclusive': min_stream_id, 'rows_inserted': rows_inserted + len(rows)}
            self.db_pool.updates._background_update_progress_txn(txn, _BackgroundUpdates.EVENT_FIELDS_SENDER_URL_UPDATE_NAME, progress)
            return len(rows)
        result = await self.db_pool.runInteraction(_BackgroundUpdates.EVENT_FIELDS_SENDER_URL_UPDATE_NAME, reindex_txn)
        if not result:
            await self.db_pool.updates._end_background_update(_BackgroundUpdates.EVENT_FIELDS_SENDER_URL_UPDATE_NAME)
        return result

    async def _background_reindex_origin_server_ts(self, progress: JsonDict, batch_size: int) -> int:
        target_min_stream_id = progress['target_min_stream_id_inclusive']
        max_stream_id = progress['max_stream_id_exclusive']
        rows_inserted = progress.get('rows_inserted', 0)

        def reindex_search_txn(txn: LoggingTransaction) -> int:
            if False:
                return 10
            sql = 'SELECT stream_ordering, event_id FROM events WHERE ? <= stream_ordering AND stream_ordering < ? ORDER BY stream_ordering DESC LIMIT ?'
            txn.execute(sql, (target_min_stream_id, max_stream_id, batch_size))
            rows = txn.fetchall()
            if not rows:
                return 0
            min_stream_id = rows[-1][0]
            event_ids = [row[1] for row in rows]
            rows_to_update = []
            chunks = [event_ids[i:i + 100] for i in range(0, len(event_ids), 100)]
            for chunk in chunks:
                ev_rows = cast(List[Tuple[str, str]], self.db_pool.simple_select_many_txn(txn, table='event_json', column='event_id', iterable=chunk, retcols=['event_id', 'json'], keyvalues={}))
                for (event_id, json) in ev_rows:
                    event_json = db_to_json(json)
                    try:
                        origin_server_ts = event_json['origin_server_ts']
                    except (KeyError, AttributeError):
                        continue
                    rows_to_update.append((origin_server_ts, event_id))
            sql = 'UPDATE events SET origin_server_ts = ? WHERE event_id = ?'
            txn.execute_batch(sql, rows_to_update)
            progress = {'target_min_stream_id_inclusive': target_min_stream_id, 'max_stream_id_exclusive': min_stream_id, 'rows_inserted': rows_inserted + len(rows_to_update)}
            self.db_pool.updates._background_update_progress_txn(txn, _BackgroundUpdates.EVENT_ORIGIN_SERVER_TS_NAME, progress)
            return len(rows_to_update)
        result = await self.db_pool.runInteraction(_BackgroundUpdates.EVENT_ORIGIN_SERVER_TS_NAME, reindex_search_txn)
        if not result:
            await self.db_pool.updates._end_background_update(_BackgroundUpdates.EVENT_ORIGIN_SERVER_TS_NAME)
        return result

    async def _cleanup_extremities_bg_update(self, progress: JsonDict, batch_size: int) -> int:
        """Background update to clean out extremities that should have been
        deleted previously.

        Mainly used to deal with the aftermath of https://github.com/matrix-org/synapse/issues/5269.
        """

        def _cleanup_extremities_bg_update_txn(txn: LoggingTransaction) -> int:
            if False:
                i = 10
                return i + 15
            original_set = set()
            graph: Dict[str, Set[str]] = {}
            non_rejected_leaves = set()
            soft_failed_events_to_lookup = set()
            txn.execute('SELECT prev_event_id, event_id, internal_metadata,\n                    rejections.event_id IS NOT NULL, events.outlier\n                FROM (\n                    SELECT event_id AS prev_event_id\n                    FROM _extremities_to_check\n                    LIMIT ?\n                ) AS f\n                LEFT JOIN event_edges USING (prev_event_id)\n                LEFT JOIN events USING (event_id)\n                LEFT JOIN event_json USING (event_id)\n                LEFT JOIN rejections USING (event_id)\n                ', (batch_size,))
            for (prev_event_id, event_id, metadata, rejected, outlier) in txn:
                original_set.add(prev_event_id)
                if not event_id or outlier:
                    continue
                graph.setdefault(event_id, set()).add(prev_event_id)
                soft_failed = False
                if metadata:
                    soft_failed = db_to_json(metadata).get('soft_failed')
                if soft_failed or rejected:
                    soft_failed_events_to_lookup.add(event_id)
                else:
                    non_rejected_leaves.add(event_id)
            while soft_failed_events_to_lookup:
                batch = list(soft_failed_events_to_lookup)
                (to_check, to_defer) = (batch[:100], batch[100:])
                soft_failed_events_to_lookup = set(to_defer)
                sql = 'SELECT prev_event_id, event_id, internal_metadata,\n                    rejections.event_id IS NOT NULL\n                    FROM event_edges\n                    INNER JOIN events USING (event_id)\n                    INNER JOIN event_json USING (event_id)\n                    LEFT JOIN rejections USING (event_id)\n                    WHERE\n                        NOT events.outlier\n                        AND\n                '
                (clause, args) = make_in_list_sql_clause(self.database_engine, 'prev_event_id', to_check)
                txn.execute(sql + clause, list(args))
                for (prev_event_id, event_id, metadata, rejected) in txn:
                    if event_id in graph:
                        graph[event_id].add(prev_event_id)
                        continue
                    graph[event_id] = {prev_event_id}
                    soft_failed = db_to_json(metadata).get('soft_failed')
                    if soft_failed or rejected:
                        soft_failed_events_to_lookup.add(event_id)
                    else:
                        non_rejected_leaves.add(event_id)
            to_delete = set()
            while non_rejected_leaves:
                event_id = non_rejected_leaves.pop()
                prev_event_ids = graph.get(event_id, set())
                non_rejected_leaves.update(prev_event_ids)
                to_delete.update(prev_event_ids)
            to_delete.intersection_update(original_set)
            deleted = self.db_pool.simple_delete_many_txn(txn=txn, table='event_forward_extremities', column='event_id', values=to_delete, keyvalues={})
            logger.info('Deleted %d forward extremities of %d checked, to clean up matrix-org/synapse#5269', deleted, len(original_set))
            if deleted:
                rows = cast(List[Tuple[str]], self.db_pool.simple_select_many_txn(txn, table='events', column='event_id', iterable=to_delete, keyvalues={}, retcols=('room_id',)))
                room_ids = {row[0] for row in rows}
                for room_id in room_ids:
                    txn.call_after(self.get_latest_event_ids_in_room.invalidate, (room_id,))
            self.db_pool.simple_delete_many_txn(txn=txn, table='_extremities_to_check', column='event_id', values=original_set, keyvalues={})
            return len(original_set)
        num_handled = await self.db_pool.runInteraction('_cleanup_extremities_bg_update', _cleanup_extremities_bg_update_txn)
        if not num_handled:
            await self.db_pool.updates._end_background_update(_BackgroundUpdates.DELETE_SOFT_FAILED_EXTREMITIES)

            def _drop_table_txn(txn: LoggingTransaction) -> None:
                if False:
                    i = 10
                    return i + 15
                txn.execute('DROP TABLE _extremities_to_check')
            await self.db_pool.runInteraction('_cleanup_extremities_bg_update_drop_table', _drop_table_txn)
        return num_handled

    async def _redactions_received_ts(self, progress: JsonDict, batch_size: int) -> int:
        """Handles filling out the `received_ts` column in redactions."""
        last_event_id = progress.get('last_event_id', '')

        def _redactions_received_ts_txn(txn: LoggingTransaction) -> int:
            if False:
                print('Hello World!')
            sql = '\n                SELECT event_id FROM redactions\n                WHERE event_id > ?\n                ORDER BY event_id ASC\n                LIMIT ?\n            '
            txn.execute(sql, (last_event_id, batch_size))
            rows = txn.fetchall()
            if not rows:
                return 0
            (upper_event_id,) = rows[-1]
            sql = '\n                UPDATE redactions\n                SET received_ts = (\n                    SELECT COALESCE(received_ts, origin_server_ts, ?) FROM events\n                    WHERE events.event_id = redactions.event_id\n                )\n                WHERE ? <= event_id AND event_id <= ?\n            '
            txn.execute(sql, (self._clock.time_msec(), last_event_id, upper_event_id))
            self.db_pool.updates._background_update_progress_txn(txn, 'redactions_received_ts', {'last_event_id': upper_event_id})
            return len(rows)
        count = await self.db_pool.runInteraction('_redactions_received_ts', _redactions_received_ts_txn)
        if not count:
            await self.db_pool.updates._end_background_update('redactions_received_ts')
        return count

    async def _event_fix_redactions_bytes(self, progress: JsonDict, batch_size: int) -> int:
        """Undoes hex encoded censored redacted event JSON."""

        def _event_fix_redactions_bytes_txn(txn: LoggingTransaction) -> None:
            if False:
                i = 10
                return i + 15
            txn.execute("\n                UPDATE event_json\n                SET\n                    json = convert_from(json::bytea, 'utf8')\n                FROM redactions\n                WHERE\n                    redactions.have_censored\n                    AND event_json.event_id = redactions.redacts\n                    AND json NOT LIKE '{%';\n                ")
            txn.execute('DROP INDEX redactions_censored_redacts')
        await self.db_pool.runInteraction('_event_fix_redactions_bytes', _event_fix_redactions_bytes_txn)
        await self.db_pool.updates._end_background_update('event_fix_redactions_bytes')
        return 1

    async def _event_store_labels(self, progress: JsonDict, batch_size: int) -> int:
        """Background update handler which will store labels for existing events."""
        last_event_id = progress.get('last_event_id', '')

        def _event_store_labels_txn(txn: LoggingTransaction) -> int:
            if False:
                i = 10
                return i + 15
            txn.execute('\n                SELECT event_id, json FROM event_json\n                LEFT JOIN event_labels USING (event_id)\n                WHERE event_id > ? AND label IS NULL\n                ORDER BY event_id LIMIT ?\n                ', (last_event_id, batch_size))
            results = list(txn)
            nbrows = 0
            last_row_event_id = ''
            for (event_id, event_json_raw) in results:
                try:
                    event_json = db_to_json(event_json_raw)
                    self.db_pool.simple_insert_many_txn(txn=txn, table='event_labels', keys=('event_id', 'label', 'room_id', 'topological_ordering'), values=[(event_id, label, event_json['room_id'], event_json['depth']) for label in event_json['content'].get(EventContentFields.LABELS, []) if isinstance(label, str)])
                except Exception as e:
                    logger.warning('Unable to load event %s (no labels will be imported): %s', event_id, e)
                nbrows += 1
                last_row_event_id = event_id
            self.db_pool.updates._background_update_progress_txn(txn, 'event_store_labels', {'last_event_id': last_row_event_id})
            return nbrows
        num_rows = await self.db_pool.runInteraction(desc='event_store_labels', func=_event_store_labels_txn)
        if not num_rows:
            await self.db_pool.updates._end_background_update('event_store_labels')
        return num_rows

    async def _rejected_events_metadata(self, progress: dict, batch_size: int) -> int:
        """Adds rejected events to the `state_events` and `event_auth` metadata
        tables.
        """
        last_event_id = progress.get('last_event_id', '')

        def get_rejected_events(txn: Cursor) -> List[Tuple[str, str, JsonDict, bool, bool]]:
            if False:
                while True:
                    i = 10
            sql = "\n                SELECT DISTINCT\n                    event_id,\n                    COALESCE(room_version, '1'),\n                    json,\n                    state_events.event_id IS NOT NULL,\n                    event_auth.event_id IS NOT NULL\n                FROM rejections\n                INNER JOIN event_json USING (event_id)\n                LEFT JOIN rooms USING (room_id)\n                LEFT JOIN state_events USING (event_id)\n                LEFT JOIN event_auth USING (event_id)\n                WHERE event_id > ?\n                ORDER BY event_id\n                LIMIT ?\n            "
            txn.execute(sql, (last_event_id, batch_size))
            return cast(List[Tuple[str, str, JsonDict, bool, bool]], [(row[0], row[1], db_to_json(row[2]), row[3], row[4]) for row in txn])
        results = await self.db_pool.runInteraction(desc='_rejected_events_metadata_get', func=get_rejected_events)
        if not results:
            await self.db_pool.updates._end_background_update('rejected_events_metadata')
            return 0
        state_events = []
        auth_events = []
        for (event_id, room_version, event_json, has_state, has_event_auth) in results:
            last_event_id = event_id
            if has_state and has_event_auth:
                continue
            room_version_obj = KNOWN_ROOM_VERSIONS.get(room_version)
            if not room_version_obj:
                logger.info('Ignoring event with unknown room version %r: %r', room_version, event_id)
                continue
            event = make_event_from_dict(event_json, room_version_obj)
            if not event.is_state():
                continue
            if not has_state:
                state_events.append((event.event_id, event.room_id, event.type, event.state_key))
            if not has_event_auth:
                for auth_id in set(event.auth_event_ids()):
                    auth_events.append((event.event_id, event.room_id, auth_id))
        if state_events:
            await self.db_pool.simple_insert_many(table='state_events', keys=('event_id', 'room_id', 'type', 'state_key'), values=state_events, desc='_rejected_events_metadata_state_events')
        if auth_events:
            await self.db_pool.simple_insert_many(table='event_auth', keys=('event_id', 'room_id', 'auth_id'), values=auth_events, desc='_rejected_events_metadata_event_auth')
        await self.db_pool.updates._background_update_progress('rejected_events_metadata', {'last_event_id': last_event_id})
        if len(results) < batch_size:
            await self.db_pool.updates._end_background_update('rejected_events_metadata')
        return len(results)

    async def _chain_cover_index(self, progress: dict, batch_size: int) -> int:
        """A background updates that iterates over all rooms and generates the
        chain cover index for them.
        """
        current_room_id = progress.get('current_room_id', '')
        last_depth = progress.get('last_depth', -1)
        last_stream = progress.get('last_stream', -1)
        result = await self.db_pool.runInteraction('_chain_cover_index', self._calculate_chain_cover_txn, current_room_id, last_depth, last_stream, batch_size, single_room=False)
        finished = result.processed_count == 0
        total_rows_processed = result.processed_count
        current_room_id = result.room_id
        last_depth = result.depth
        last_stream = result.stream
        for (room_id, (depth, stream)) in result.finished_room_map.items():
            await self.db_pool.simple_update(table='rooms', keyvalues={'room_id': room_id}, updatevalues={'has_auth_chain_index': True}, desc='_chain_cover_index')
            result = await self.db_pool.runInteraction('_chain_cover_index', self._calculate_chain_cover_txn, room_id, depth, stream, batch_size=None, single_room=True)
            total_rows_processed += result.processed_count
        if finished:
            await self.db_pool.updates._end_background_update('chain_cover')
            return total_rows_processed
        await self.db_pool.updates._background_update_progress('chain_cover', {'current_room_id': current_room_id, 'last_depth': last_depth, 'last_stream': last_stream})
        return total_rows_processed

    def _calculate_chain_cover_txn(self, txn: LoggingTransaction, last_room_id: str, last_depth: int, last_stream: int, batch_size: Optional[int], single_room: bool) -> _CalculateChainCover:
        if False:
            print('Hello World!')
        'Calculate the chain cover for `batch_size` events, ordered by\n        `(room_id, depth, stream)`.\n\n        Args:\n            txn,\n            last_room_id, last_depth, last_stream: The `(room_id, depth, stream)`\n                tuple to fetch results after.\n            batch_size: The maximum number of events to process. If None then\n                no limit.\n            single_room: Whether to calculate the index for just the given\n                room.\n        '
        (tuple_clause, tuple_args) = make_tuple_comparison_clause([('events.room_id', last_room_id), ('topological_ordering', last_depth), ('stream_ordering', last_stream)])
        extra_clause = ''
        if single_room:
            extra_clause = 'AND events.room_id = ?'
            tuple_args.append(last_room_id)
        sql = '\n            SELECT\n                event_id, state_events.type, state_events.state_key,\n                topological_ordering, stream_ordering,\n                events.room_id\n            FROM events\n            INNER JOIN state_events USING (event_id)\n            LEFT JOIN event_auth_chains USING (event_id)\n            LEFT JOIN event_auth_chain_to_calculate USING (event_id)\n            WHERE event_auth_chains.event_id IS NULL\n                AND event_auth_chain_to_calculate.event_id IS NULL\n                AND %(tuple_cmp)s\n                %(extra)s\n            ORDER BY events.room_id, topological_ordering, stream_ordering\n            %(limit)s\n        ' % {'tuple_cmp': tuple_clause, 'limit': 'LIMIT ?' if batch_size is not None else '', 'extra': extra_clause}
        if batch_size is not None:
            tuple_args.append(batch_size)
        txn.execute(sql, tuple_args)
        rows = txn.fetchall()
        event_to_room_id = {row[0]: row[5] for row in rows}
        event_to_types = {row[0]: (row[1], row[2]) for row in rows}
        new_last_depth: int = rows[-1][3] if rows else last_depth
        new_last_stream: int = rows[-1][4] if rows else last_stream
        new_last_room_id: str = rows[-1][5] if rows else ''
        finished_rooms = {row[5]: (row[3], row[4]) for row in rows if row[5] != new_last_room_id}
        if last_room_id not in finished_rooms and last_room_id != new_last_room_id:
            finished_rooms[last_room_id] = (last_depth, last_stream)
        count = len(rows)
        auth_events = cast(List[Tuple[str, str]], self.db_pool.simple_select_many_txn(txn, table='event_auth', column='event_id', iterable=event_to_room_id, keyvalues={}, retcols=('event_id', 'auth_id')))
        event_to_auth_chain: Dict[str, List[str]] = {}
        for (event_id, auth_id) in auth_events:
            event_to_auth_chain.setdefault(event_id, []).append(auth_id)
        PersistEventsStore._add_chain_cover_index(txn, self.db_pool, self.event_chain_id_gen, event_to_room_id, event_to_types, cast(Dict[str, StrCollection], event_to_auth_chain))
        return _CalculateChainCover(room_id=new_last_room_id, depth=new_last_depth, stream=new_last_stream, processed_count=count, finished_room_map=finished_rooms)

    async def _purged_chain_cover_index(self, progress: dict, batch_size: int) -> int:
        """
        A background updates that iterates over the chain cover and deletes the
        chain cover for events that have been purged.

        This may be due to fully purging a room or via setting a retention policy.
        """
        current_event_id = progress.get('current_event_id', '')

        def purged_chain_cover_txn(txn: LoggingTransaction) -> int:
            if False:
                for i in range(10):
                    print('nop')
            sql = '\n                SELECT event_id, chain_id, sequence_number, e.event_id IS NOT NULL\n                FROM event_auth_chains\n                LEFT JOIN events AS e USING (event_id)\n                WHERE event_id > ? ORDER BY event_auth_chains.event_id ASC LIMIT ?\n            '
            txn.execute(sql, (current_event_id, batch_size))
            rows = txn.fetchall()
            if not rows:
                return 0
            unreferenced_event_ids = []
            unreferenced_chain_id_tuples = []
            event_id = ''
            for (event_id, chain_id, sequence_number, has_event) in rows:
                if not has_event:
                    unreferenced_event_ids.append((event_id,))
                    unreferenced_chain_id_tuples.append((chain_id, sequence_number))
            txn.executemany('\n                DELETE FROM event_auth_chains WHERE event_id = ?\n                ', unreferenced_event_ids)
            txn.executemany('\n                DELETE FROM event_auth_chain_links WHERE\n                origin_chain_id = ? AND origin_sequence_number = ?\n                ', unreferenced_chain_id_tuples)
            progress = {'current_event_id': event_id}
            self.db_pool.updates._background_update_progress_txn(txn, 'purged_chain_cover', progress)
            return len(rows)
        result = await self.db_pool.runInteraction('_purged_chain_cover_index', purged_chain_cover_txn)
        if not result:
            await self.db_pool.updates._end_background_update('purged_chain_cover')
        return result

    async def _event_arbitrary_relations(self, progress: JsonDict, batch_size: int) -> int:
        """Background update handler which will store previously unknown relations for existing events."""
        last_event_id = progress.get('last_event_id', '')

        def _event_arbitrary_relations_txn(txn: LoggingTransaction) -> int:
            if False:
                i = 10
                return i + 15
            txn.execute('\n                SELECT event_id, json FROM event_json\n                WHERE event_id > ?\n                ORDER BY event_id LIMIT ?\n                ', (last_event_id, batch_size))
            results = list(txn)
            relations_to_insert: List[Tuple[str, str, str]] = []
            for (event_id, event_json_raw) in results:
                try:
                    event_json = db_to_json(event_json_raw)
                except Exception as e:
                    logger.warning('Unable to load event %s (no relations will be updated): %s', event_id, e)
                    continue
                relates_to = event_json['content'].get('m.relates_to')
                if not relates_to or not isinstance(relates_to, dict):
                    continue
                rel_type = relates_to.get('rel_type')
                if not isinstance(rel_type, str) or rel_type in (RelationTypes.ANNOTATION, RelationTypes.REFERENCE, RelationTypes.REPLACE):
                    continue
                parent_id = relates_to.get('event_id')
                if not isinstance(parent_id, str):
                    continue
                relations_to_insert.append((event_id, parent_id, rel_type))
            if relations_to_insert:
                self.db_pool.simple_upsert_many_txn(txn=txn, table='event_relations', key_names=('event_id',), key_values=[(r[0],) for r in relations_to_insert], value_names=('relates_to_id', 'relation_type'), value_values=[r[1:] for r in relations_to_insert])
                cache_tuples = {(r[1],) for r in relations_to_insert}
                self._invalidate_cache_and_stream_bulk(txn, self.get_relations_for_event, cache_tuples)
                self._invalidate_cache_and_stream_bulk(txn, self.get_thread_summary, cache_tuples)
            if results:
                latest_event_id = results[-1][0]
                self.db_pool.updates._background_update_progress_txn(txn, 'event_arbitrary_relations', {'last_event_id': latest_event_id})
            return len(results)
        num_rows = await self.db_pool.runInteraction(desc='event_arbitrary_relations', func=_event_arbitrary_relations_txn)
        if not num_rows:
            await self.db_pool.updates._end_background_update('event_arbitrary_relations')
        return num_rows

    async def _background_populate_stream_ordering2(self, progress: JsonDict, batch_size: int) -> int:
        """Populate events.stream_ordering2, then replace stream_ordering

        This is to deal with the fact that stream_ordering was initially created as a
        32-bit integer field.
        """
        batch_size = max(batch_size, 1)

        def process(txn: LoggingTransaction) -> int:
            if False:
                while True:
                    i = 10
            last_stream = progress.get('last_stream', -(1 << 31))
            txn.execute('\n                UPDATE events SET stream_ordering2=stream_ordering\n                WHERE stream_ordering IN (\n                   SELECT stream_ordering FROM events WHERE stream_ordering > ?\n                   ORDER BY stream_ordering LIMIT ?\n                )\n                RETURNING stream_ordering;\n                ', (last_stream, batch_size))
            row_count = txn.rowcount
            if row_count == 0:
                return 0
            last_stream = max((row[0] for row in txn))
            logger.info('populated stream_ordering2 up to %i', last_stream)
            self.db_pool.updates._background_update_progress_txn(txn, _BackgroundUpdates.POPULATE_STREAM_ORDERING2, {'last_stream': last_stream})
            return row_count
        result = await self.db_pool.runInteraction('_background_populate_stream_ordering2', process)
        if result != 0:
            return result
        await self.db_pool.updates._end_background_update(_BackgroundUpdates.POPULATE_STREAM_ORDERING2)
        return 0

    async def _background_replace_stream_ordering_column(self, progress: JsonDict, batch_size: int) -> int:
        """Drop the old 'stream_ordering' column and rename 'stream_ordering2' into its place."""

        def process(txn: Cursor) -> None:
            if False:
                i = 10
                return i + 15
            for sql in _REPLACE_STREAM_ORDERING_SQL_COMMANDS:
                logger.info('completing stream_ordering migration: %s', sql)
                txn.execute(sql)
        await self.db_pool.runInteraction('background_analyze_new_stream_ordering_column', lambda txn: txn.execute('ANALYZE events(stream_ordering2)'))
        await self.db_pool.runInteraction('_background_replace_stream_ordering_column', process)
        await self.db_pool.updates._end_background_update(_BackgroundUpdates.REPLACE_STREAM_ORDERING_COLUMN)
        return 0

    async def _background_drop_invalid_event_edges_rows(self, progress: JsonDict, batch_size: int) -> int:
        """Drop invalid rows from event_edges

        This only runs for postgres. For SQLite, it all happens synchronously.

        Firstly, drop any rows with is_state=True. These may have been added a long time
        ago, but they are no longer used.

        We also drop rows that do not correspond to entries in `events`, and add a
        foreign key.
        """
        last_event_id = progress.get('last_event_id', '')

        def drop_invalid_event_edges_txn(txn: LoggingTransaction) -> bool:
            if False:
                i = 10
                return i + 15
            "Returns True if we're done."
            txn.execute('\n                SELECT event_id FROM event_edges\n                WHERE event_id > ?\n                ORDER BY event_id\n                LIMIT 1 OFFSET ?\n                ', (last_event_id, batch_size))
            endpoint = None
            row = txn.fetchone()
            if row:
                endpoint = row[0]
            where_clause = 'ee.event_id > ?'
            args = [last_event_id]
            if endpoint:
                where_clause += ' AND ee.event_id <= ?'
                args.append(endpoint)
            txn.execute(f'\n                DELETE FROM event_edges\n                WHERE event_id IN (\n                   SELECT ee.event_id\n                   FROM event_edges ee\n                     LEFT JOIN events ev USING (event_id)\n                   WHERE ({where_clause}) AND\n                     (is_state OR ev.event_id IS NULL)\n                )', args)
            logger.info('cleaned up event_edges up to %s: removed %i/%i rows', endpoint, txn.rowcount, batch_size)
            if endpoint is not None:
                self.db_pool.updates._background_update_progress_txn(txn, _BackgroundUpdates.EVENT_EDGES_DROP_INVALID_ROWS, {'last_event_id': endpoint})
                return False
            logger.info('cleaned up event_edges; enabling foreign key')
            txn.execute('ALTER TABLE event_edges VALIDATE CONSTRAINT event_edges_event_id_fkey')
            return True
        done = await self.db_pool.runInteraction(desc='drop_invalid_event_edges', func=drop_invalid_event_edges_txn)
        if done:
            await self.db_pool.updates._end_background_update(_BackgroundUpdates.EVENT_EDGES_DROP_INVALID_ROWS)
        return batch_size

    async def _background_events_populate_state_key_rejections(self, progress: JsonDict, batch_size: int) -> int:
        """Back-populate `events.state_key` and `events.rejection_reason"""
        min_stream_ordering_exclusive = progress['min_stream_ordering_exclusive']
        max_stream_ordering_inclusive = progress['max_stream_ordering_inclusive']

        def _populate_txn(txn: LoggingTransaction) -> bool:
            if False:
                i = 10
                return i + 15
            "Returns True if we're done."
            txn.execute('\n                SELECT stream_ordering FROM events\n                WHERE stream_ordering > ? AND stream_ordering <= ?\n                ORDER BY stream_ordering\n                LIMIT 1 OFFSET ?\n                ', (min_stream_ordering_exclusive, max_stream_ordering_inclusive, batch_size - 1))
            row = txn.fetchone()
            if row:
                endpoint = row[0]
            else:
                endpoint = max_stream_ordering_inclusive
            where_clause = 'stream_ordering > ? AND stream_ordering <= ?'
            args = [min_stream_ordering_exclusive, endpoint]
            txn.execute(f'\n                UPDATE events\n                SET state_key = (SELECT state_key FROM state_events se WHERE se.event_id = events.event_id),\n                    rejection_reason = (SELECT reason FROM rejections rej WHERE rej.event_id = events.event_id)\n                WHERE ({where_clause})\n                ', args)
            logger.info('populated new `events` columns up to %i/%i: updated %i rows', endpoint, max_stream_ordering_inclusive, txn.rowcount)
            if endpoint >= max_stream_ordering_inclusive:
                return True
            progress['min_stream_ordering_exclusive'] = endpoint
            self.db_pool.updates._background_update_progress_txn(txn, _BackgroundUpdates.EVENTS_POPULATE_STATE_KEY_REJECTIONS, progress)
            return False
        done = await self.db_pool.runInteraction(desc='events_populate_state_key_rejections', func=_populate_txn)
        if done:
            await self.db_pool.updates._end_background_update(_BackgroundUpdates.EVENTS_POPULATE_STATE_KEY_REJECTIONS)
        return batch_size