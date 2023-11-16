import contextlib
import glob
import logging
import os
import re
import sqlite3
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ContextManager, Iterator, Optional, Sequence, Union
import sqlalchemy as db
import sqlalchemy.exc as db_exc
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.pool import NullPool
from tqdm import tqdm
from watchdog.events import FileSystemEvent, PatternMatchingEventHandler
from watchdog.observers import Observer
import dagster._check as check
import dagster._seven as seven
from dagster._config import StringSource
from dagster._config.config_schema import UserConfigSchema
from dagster._core.definitions.events import AssetKey
from dagster._core.errors import DagsterInvariantViolationError
from dagster._core.event_api import EventHandlerFn, EventRecordsResult, RunStatusChangeRecordsFilter
from dagster._core.events import ASSET_CHECK_EVENTS, ASSET_EVENTS, EVENT_TYPE_TO_PIPELINE_RUN_STATUS, DagsterEventType
from dagster._core.events.log import EventLogEntry
from dagster._core.storage.dagster_run import DagsterRunStatus, RunsFilter
from dagster._core.storage.event_log.base import EventLogCursor, EventLogRecord, EventRecordsFilter
from dagster._core.storage.sql import AlembicVersion, check_alembic_revision, create_engine, get_alembic_config, run_alembic_upgrade, stamp_alembic_rev
from dagster._core.storage.sqlalchemy_compat import db_select
from dagster._core.storage.sqlite import create_db_conn_string
from dagster._serdes import ConfigurableClass, ConfigurableClassData
from dagster._serdes.errors import DeserializationError
from dagster._serdes.serdes import deserialize_value
from dagster._utils import mkdir_p
from ..schema import SqlEventLogStorageMetadata, SqlEventLogStorageTable
from ..sql_event_log import RunShardedEventsCursor, SqlEventLogStorage
if TYPE_CHECKING:
    from dagster._core.storage.sqlite_storage import SqliteStorageConfig
INDEX_SHARD_NAME = 'index'

class SqliteEventLogStorage(SqlEventLogStorage, ConfigurableClass):
    """SQLite-backed event log storage.

    Users should not directly instantiate this class; it is instantiated by internal machinery when
    ``dagster-webserver`` and ``dagster-graphql`` load, based on the values in the ``dagster.yaml`` file insqliteve
    ``$DAGSTER_HOME``. Configuration of this class should be done by setting values in that file.

    This is the default event log storage when none is specified in the ``dagster.yaml``.

    To explicitly specify SQLite for event log storage, you can add a block such as the following
    to your ``dagster.yaml``:

    .. code-block:: YAML

        event_log_storage:
          module: dagster._core.storage.event_log
          class: SqliteEventLogStorage
          config:
            base_dir: /path/to/dir

    The ``base_dir`` param tells the event log storage where on disk to store the databases. To
    improve concurrent performance, event logs are stored in a separate SQLite database for each
    run.
    """

    def __init__(self, base_dir: str, inst_data: Optional[ConfigurableClassData]=None):
        if False:
            i = 10
            return i + 15
        'Note that idempotent initialization of the SQLite database is done on a per-run_id\n        basis in the body of connect, since each run is stored in a separate database.\n        '
        self._base_dir = os.path.abspath(check.str_param(base_dir, 'base_dir'))
        mkdir_p(self._base_dir)
        self._obs = None
        self._watchers = defaultdict(dict)
        self._inst_data = check.opt_inst_param(inst_data, 'inst_data', ConfigurableClassData)
        self._initialized_dbs = set()
        self._db_lock = threading.Lock()
        if not os.path.exists(self.path_for_shard(INDEX_SHARD_NAME)):
            conn_string = self.conn_string_for_shard(INDEX_SHARD_NAME)
            engine = create_engine(conn_string, poolclass=NullPool)
            self._initdb(engine)
            self.reindex_events()
            self.reindex_assets()
        super().__init__()

    def upgrade(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        all_run_ids = self.get_all_run_ids()
        print(f'Updating event log storage for {len(all_run_ids)} runs on disk...')
        alembic_config = get_alembic_config(__file__)
        if all_run_ids:
            for run_id in tqdm(all_run_ids):
                with self.run_connection(run_id) as conn:
                    run_alembic_upgrade(alembic_config, conn, run_id)
        print('Updating event log storage for index db on disk...')
        with self.index_connection() as conn:
            run_alembic_upgrade(alembic_config, conn, 'index')
        self._initialized_dbs = set()

    @property
    def inst_data(self) -> Optional[ConfigurableClassData]:
        if False:
            i = 10
            return i + 15
        return self._inst_data

    @classmethod
    def config_type(cls) -> UserConfigSchema:
        if False:
            return 10
        return {'base_dir': StringSource}

    @classmethod
    def from_config_value(cls, inst_data: Optional[ConfigurableClassData], config_value: 'SqliteStorageConfig') -> 'SqliteEventLogStorage':
        if False:
            while True:
                i = 10
        return SqliteEventLogStorage(inst_data=inst_data, **config_value)

    def get_all_run_ids(self) -> Sequence[str]:
        if False:
            i = 10
            return i + 15
        all_filenames = glob.glob(os.path.join(self._base_dir, '*.db'))
        return [os.path.splitext(os.path.basename(filename))[0] for filename in all_filenames if os.path.splitext(os.path.basename(filename))[0] != INDEX_SHARD_NAME]

    def has_table(self, table_name: str) -> bool:
        if False:
            while True:
                i = 10
        conn_string = self.conn_string_for_shard(INDEX_SHARD_NAME)
        engine = create_engine(conn_string, poolclass=NullPool)
        with engine.connect() as conn:
            return bool(engine.dialect.has_table(conn, table_name))

    def path_for_shard(self, run_id: str) -> str:
        if False:
            return 10
        return os.path.join(self._base_dir, f'{run_id}.db')

    def conn_string_for_shard(self, shard_name: str) -> str:
        if False:
            return 10
        check.str_param(shard_name, 'shard_name')
        return create_db_conn_string(self._base_dir, shard_name)

    def _initdb(self, engine: Engine) -> None:
        if False:
            while True:
                i = 10
        alembic_config = get_alembic_config(__file__)
        retry_limit = 10
        while True:
            try:
                with engine.connect() as connection:
                    (db_revision, head_revision) = check_alembic_revision(alembic_config, connection)
                    if not (db_revision and head_revision):
                        SqlEventLogStorageMetadata.create_all(engine)
                        connection.execute(db.text('PRAGMA journal_mode=WAL;'))
                        stamp_alembic_rev(alembic_config, connection)
                break
            except (db_exc.DatabaseError, sqlite3.DatabaseError, sqlite3.OperationalError) as exc:
                err_msg = str(exc)
                if not (re.search('table [A-Za-z_]* already exists', err_msg) or 'database is locked' in err_msg or 'UNIQUE constraint failed: alembic_version.version_num' in err_msg):
                    raise
                if retry_limit == 0:
                    raise
                else:
                    logging.info('SqliteEventLogStorage._initdb: Encountered apparent concurrent init, retrying (%s retries left). Exception: %s', retry_limit, err_msg)
                    time.sleep(0.2)
                    retry_limit -= 1

    @contextmanager
    def _connect(self, shard: str) -> Iterator[Connection]:
        if False:
            while True:
                i = 10
        with self._db_lock:
            check.str_param(shard, 'shard')
            conn_string = self.conn_string_for_shard(shard)
            engine = create_engine(conn_string, poolclass=NullPool)
            if shard not in self._initialized_dbs:
                self._initdb(engine)
                self._initialized_dbs.add(shard)
            with engine.connect() as conn:
                with conn.begin():
                    yield conn
            engine.dispose()

    def run_connection(self, run_id: Optional[str]=None) -> Any:
        if False:
            while True:
                i = 10
        return self._connect(run_id)

    def index_connection(self) -> ContextManager[Connection]:
        if False:
            for i in range(10):
                print('nop')
        return self._connect(INDEX_SHARD_NAME)

    def store_event(self, event: EventLogEntry) -> None:
        if False:
            while True:
                i = 10
        'Overridden method to replicate asset events in a central assets.db sqlite shard, enabling\n        cross-run asset queries.\n\n        Args:\n            event (EventLogEntry): The event to store.\n        '
        check.inst_param(event, 'event', EventLogEntry)
        insert_event_statement = self.prepare_insert_event(event)
        run_id = event.run_id
        with self.run_connection(run_id) as conn:
            conn.execute(insert_event_statement)
        if event.is_dagster_event and event.dagster_event.asset_key:
            check.invariant(event.dagster_event_type in ASSET_EVENTS, 'Can only store asset materializations, materialization_planned, and observations in index database')
            event_id = None
            with self.index_connection() as conn:
                result = conn.execute(insert_event_statement)
                event_id = result.inserted_primary_key[0]
            self.store_asset_event(event, event_id)
            if event_id is None:
                raise DagsterInvariantViolationError('Cannot store asset event tags for null event id.')
            self.store_asset_event_tags(event, event_id)
        if event.is_dagster_event and event.dagster_event_type in ASSET_CHECK_EVENTS:
            self.store_asset_check_event(event, None)
        if event.is_dagster_event and event.dagster_event_type in EVENT_TYPE_TO_PIPELINE_RUN_STATUS:
            with self.index_connection() as conn:
                conn.execute(insert_event_statement)

    def get_event_records(self, event_records_filter: EventRecordsFilter, limit: Optional[int]=None, ascending: bool=False) -> Sequence[EventLogRecord]:
        if False:
            while True:
                i = 10
        'Overridden method to enable cross-run event queries in sqlite.\n\n        The record id in sqlite does not auto increment cross runs, so instead of fetching events\n        after record id, we only fetch events whose runs updated after update_timestamp.\n        '
        check.opt_inst_param(event_records_filter, 'event_records_filter', EventRecordsFilter)
        check.opt_int_param(limit, 'limit')
        check.bool_param(ascending, 'ascending')
        is_asset_query = event_records_filter and event_records_filter.event_type in ASSET_EVENTS
        if is_asset_query:
            return super(SqliteEventLogStorage, self).get_event_records(event_records_filter=event_records_filter, limit=limit, ascending=ascending)
        return self._get_run_sharded_event_records(event_records_filter=event_records_filter, limit=limit, ascending=ascending)

    def _get_run_sharded_event_records(self, event_records_filter: EventRecordsFilter, limit: Optional[int]=None, ascending: bool=False) -> Sequence[EventLogRecord]:
        if False:
            return 10
        query = db_select([SqlEventLogStorageTable.c.id, SqlEventLogStorageTable.c.event])
        if event_records_filter.asset_key:
            asset_details = next(iter(self._get_assets_details([event_records_filter.asset_key])))
        else:
            asset_details = None
        if event_records_filter.after_cursor is not None and (not isinstance(event_records_filter.after_cursor, RunShardedEventsCursor)):
            raise Exception('\n                Called `get_event_records` on a run-sharded event log storage with a cursor that\n                is not run-aware. Add a RunShardedEventsCursor to your query filter\n                or switch your instance configuration to use a non-run-sharded event log storage\n                (e.g. PostgresEventLogStorage, ConsolidatedSqliteEventLogStorage)\n            ')
        query = self._apply_filter_to_query(query=query, event_records_filter=event_records_filter, asset_details=asset_details, apply_cursor_filters=False)
        if limit:
            query = query.limit(limit)
        if ascending:
            query = query.order_by(SqlEventLogStorageTable.c.timestamp.asc())
        else:
            query = query.order_by(SqlEventLogStorageTable.c.timestamp.desc())
        run_updated_after = event_records_filter.after_cursor.run_updated_after if isinstance(event_records_filter.after_cursor, RunShardedEventsCursor) else None
        run_records = self._instance.get_run_records(filters=RunsFilter(updated_after=run_updated_after), order_by='update_timestamp', ascending=ascending)
        event_records = []
        for run_record in run_records:
            run_id = run_record.dagster_run.run_id
            with self.run_connection(run_id) as conn:
                results = conn.execute(query).fetchall()
            for (row_id, json_str) in results:
                try:
                    event_record = deserialize_value(json_str, EventLogEntry)
                    event_records.append(EventLogRecord(storage_id=row_id, event_log_entry=event_record))
                    if limit and len(event_records) >= limit:
                        break
                except DeserializationError:
                    logging.warning('Could not resolve event record as EventLogEntry for id `%s`.', row_id)
                except seven.JSONDecodeError:
                    logging.warning('Could not parse event record id `%s`.', row_id)
            if limit and len(event_records) >= limit:
                break
        return event_records[:limit]

    def fetch_run_status_changes(self, records_filter: Union[DagsterEventType, RunStatusChangeRecordsFilter], limit: int, cursor: Optional[str]=None, ascending: bool=False) -> EventRecordsResult:
        if False:
            while True:
                i = 10
        event_type = records_filter if isinstance(records_filter, DagsterEventType) else records_filter.event_type
        if event_type not in EVENT_TYPE_TO_PIPELINE_RUN_STATUS:
            expected = ', '.join(EVENT_TYPE_TO_PIPELINE_RUN_STATUS.keys())
            check.failed(f'Expected one of {expected}, received {event_type.value}')
        (before_cursor, after_cursor) = EventRecordsFilter.get_cursor_params(cursor, ascending)
        event_records_filter = records_filter.to_event_records_filter(cursor, ascending) if isinstance(records_filter, RunStatusChangeRecordsFilter) else EventRecordsFilter(event_type, before_cursor=before_cursor, after_cursor=after_cursor)
        records = super(SqliteEventLogStorage, self).get_event_records(event_records_filter=event_records_filter, limit=limit, ascending=ascending)
        if records:
            new_cursor = EventLogCursor.from_storage_id(records[-1].storage_id).to_string()
        elif cursor:
            new_cursor = cursor
        else:
            new_cursor = EventLogCursor.from_storage_id(-1).to_string()
        has_more = len(records) == limit
        return EventRecordsResult(records, cursor=new_cursor, has_more=has_more)

    def supports_event_consumer_queries(self) -> bool:
        if False:
            return 10
        return False

    def delete_events(self, run_id: str) -> None:
        if False:
            while True:
                i = 10
        with self.run_connection(run_id) as conn:
            self.delete_events_for_run(conn, run_id)
        with self.index_connection() as conn:
            self.delete_events_for_run(conn, run_id)

    def wipe(self) -> None:
        if False:
            while True:
                i = 10
        for filename in glob.glob(os.path.join(self._base_dir, '*.db')) + glob.glob(os.path.join(self._base_dir, '*.db-wal')) + glob.glob(os.path.join(self._base_dir, '*.db-shm')):
            if not filename.endswith(f'{INDEX_SHARD_NAME}.db') and (not filename.endswith(f'{INDEX_SHARD_NAME}.db-wal')) and (not filename.endswith(f'{INDEX_SHARD_NAME}.db-shm')):
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(filename)
        self._initialized_dbs = set()
        self._wipe_index()

    def _delete_mirrored_events_for_asset_key(self, asset_key: AssetKey) -> None:
        if False:
            i = 10
            return i + 15
        with self.index_connection() as conn:
            conn.execute(SqlEventLogStorageTable.delete().where(SqlEventLogStorageTable.c.asset_key == asset_key.to_string()))

    def wipe_asset(self, asset_key: AssetKey) -> None:
        if False:
            return 10
        super(SqliteEventLogStorage, self).wipe_asset(asset_key)
        self._delete_mirrored_events_for_asset_key(asset_key)

    def watch(self, run_id: str, cursor: Optional[str], callback: EventHandlerFn) -> None:
        if False:
            i = 10
            return i + 15
        if not self._obs:
            self._obs = Observer()
            self._obs.start()
        watchdog = SqliteEventLogStorageWatchdog(self, run_id, callback, cursor)
        self._watchers[run_id][callback] = (watchdog, self._obs.schedule(watchdog, self._base_dir, True))

    def end_watch(self, run_id: str, handler: EventHandlerFn) -> None:
        if False:
            i = 10
            return i + 15
        if handler in self._watchers[run_id]:
            (event_handler, watch) = self._watchers[run_id][handler]
            self._obs.remove_handler_for_watch(event_handler, watch)
            del self._watchers[run_id][handler]

    def dispose(self) -> None:
        if False:
            while True:
                i = 10
        if self._obs:
            self._obs.stop()
            self._obs.join(timeout=15)

    def alembic_version(self) -> AlembicVersion:
        if False:
            i = 10
            return i + 15
        alembic_config = get_alembic_config(__file__)
        with self.index_connection() as conn:
            return check_alembic_revision(alembic_config, conn)

    @property
    def is_run_sharded(self) -> bool:
        if False:
            i = 10
            return i + 15
        return True

    @property
    def supports_global_concurrency_limits(self) -> bool:
        if False:
            return 10
        return False

class SqliteEventLogStorageWatchdog(PatternMatchingEventHandler):

    def __init__(self, event_log_storage: SqliteEventLogStorage, run_id: str, callback: EventHandlerFn, cursor: Optional[str], **kwargs: Any):
        if False:
            return 10
        self._event_log_storage = check.inst_param(event_log_storage, 'event_log_storage', SqliteEventLogStorage)
        self._run_id = check.str_param(run_id, 'run_id')
        self._cb = check.callable_param(callback, 'callback')
        self._log_path = event_log_storage.path_for_shard(run_id)
        self._cursor = cursor
        super(SqliteEventLogStorageWatchdog, self).__init__(patterns=[self._log_path], **kwargs)

    def _process_log(self) -> None:
        if False:
            while True:
                i = 10
        connection = self._event_log_storage.get_records_for_run(self._run_id, self._cursor)
        if connection.cursor:
            self._cursor = connection.cursor
        for record in connection.records:
            status = None
            try:
                status = self._cb(record.event_log_entry, str(EventLogCursor.from_storage_id(record.storage_id)))
            except Exception:
                logging.exception('Exception in callback for event watch on run %s.', self._run_id)
            if status == DagsterRunStatus.SUCCESS or status == DagsterRunStatus.FAILURE or status == DagsterRunStatus.CANCELED:
                self._event_log_storage.end_watch(self._run_id, self._cb)

    def on_modified(self, event: FileSystemEvent) -> None:
        if False:
            return 10
        check.invariant(event.src_path == self._log_path)
        self._process_log()