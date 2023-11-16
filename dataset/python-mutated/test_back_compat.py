import datetime
import os
import subprocess
import tempfile
from urllib.parse import urlparse
import pytest
import sqlalchemy as db
from dagster import AssetKey, AssetMaterialization, AssetObservation, DagsterEventType, EventRecordsFilter, Output, job, op
from dagster._core.errors import DagsterInvalidInvocationError
from dagster._core.instance import DagsterInstance
from dagster._core.storage.event_log.migration import ASSET_KEY_INDEX_COLS
from dagster._core.storage.migration.bigint_migration import run_bigint_migration
from dagster._core.storage.sqlalchemy_compat import db_select
from dagster._daemon.types import DaemonHeartbeat
from dagster._utils import file_relative_path

def get_columns(instance, table_name: str):
    if False:
        i = 10
        return i + 15
    with instance.run_storage.connect() as conn:
        return set((c['name'] for c in db.inspect(conn).get_columns(table_name)))

def get_indexes(instance, table_name: str):
    if False:
        i = 10
        return i + 15
    with instance.run_storage.connect() as conn:
        return set((i['name'] for i in db.inspect(conn).get_indexes(table_name)))

def get_tables(instance):
    if False:
        return 10
    with instance.run_storage.connect() as conn:
        return db.inspect(conn).get_table_names()

def _reconstruct_from_file(conn_string, path, _username='root', _password='test'):
    if False:
        i = 10
        return i + 15
    parse_result = urlparse(conn_string)
    hostname = parse_result.hostname
    port = parse_result.port
    engine = db.create_engine(conn_string)
    with engine.connect() as conn:
        with conn.begin():
            conn.execute(db.text('drop schema test;'))
            conn.execute(db.text('create schema test;'))
    env = os.environ.copy()
    env['MYSQL_PWD'] = 'test'
    subprocess.check_call(f'mysql -uroot -h{hostname} -P{port} -ptest test < {path}', shell=True, env=env)
    return (hostname, port)

def test_0_13_17_mysql_convert_float_cols(conn_string):
    if False:
        return 10
    (hostname, port) = _reconstruct_from_file(conn_string, file_relative_path(__file__, 'snapshot_0_13_18_start_end_timestamp.sql'))
    with tempfile.TemporaryDirectory() as tempdir:
        with open(file_relative_path(__file__, 'dagster.yaml'), 'r', encoding='utf8') as template_fd:
            with open(os.path.join(tempdir, 'dagster.yaml'), 'w', encoding='utf8') as target_fd:
                template = template_fd.read().format(hostname=hostname, port=port)
                target_fd.write(template)
        instance = DagsterInstance.from_config(tempdir)
        record = instance.get_run_records(limit=1)[0]
        assert int(record.start_time) == 1643760000
        assert int(record.end_time) == 1643760000
        instance.upgrade()
        record = instance.get_run_records(limit=1)[0]
        assert record.start_time is None
        assert record.end_time is None
        instance.reindex()
        record = instance.get_run_records(limit=1)[0]
        assert int(record.start_time) == 1643788829
        assert int(record.end_time) == 1643788834

def test_instigators_table_backcompat(conn_string):
    if False:
        return 10
    (hostname, port) = _reconstruct_from_file(conn_string, file_relative_path(__file__, 'snapshot_0_14_6_instigators_table.sql'))
    with tempfile.TemporaryDirectory() as tempdir:
        with open(file_relative_path(__file__, 'dagster.yaml'), 'r', encoding='utf8') as template_fd:
            with open(os.path.join(tempdir, 'dagster.yaml'), 'w', encoding='utf8') as target_fd:
                template = template_fd.read().format(hostname=hostname, port=port)
                target_fd.write(template)
        instance = DagsterInstance.from_config(tempdir)
        assert not instance.schedule_storage.has_instigators_table()
        instance.upgrade()
        assert instance.schedule_storage.has_instigators_table()

def test_asset_observation_backcompat(conn_string):
    if False:
        while True:
            i = 10
    (hostname, port) = _reconstruct_from_file(conn_string, file_relative_path(__file__, 'snapshot_0_11_16_pre_add_asset_key_index_cols.sql'))

    @op
    def asset_op(_):
        if False:
            for i in range(10):
                print('nop')
        yield AssetObservation(asset_key=AssetKey(['a']))
        yield Output(1)

    @job
    def asset_job():
        if False:
            return 10
        asset_op()
    with tempfile.TemporaryDirectory() as tempdir:
        with open(file_relative_path(__file__, 'dagster.yaml'), 'r', encoding='utf8') as template_fd:
            with open(os.path.join(tempdir, 'dagster.yaml'), 'w', encoding='utf8') as target_fd:
                template = template_fd.read().format(hostname=hostname, port=port)
                target_fd.write(template)
        with DagsterInstance.from_config(tempdir) as instance:
            storage = instance._event_storage
            assert not instance.event_log_storage.has_secondary_index(ASSET_KEY_INDEX_COLS)
            asset_job.execute_in_process(instance=instance)
            assert storage.has_asset_key(AssetKey(['a']))

def test_jobs_selector_id_migration(conn_string):
    if False:
        return 10
    import sqlalchemy as db
    from dagster._core.storage.schedules.migration import SCHEDULE_JOBS_SELECTOR_ID
    from dagster._core.storage.schedules.schema import InstigatorsTable, JobTable, JobTickTable
    (hostname, port) = _reconstruct_from_file(conn_string, file_relative_path(__file__, 'snapshot_0_14_6_post_schema_pre_data_migration.sql'))
    with tempfile.TemporaryDirectory() as tempdir:
        with open(file_relative_path(__file__, 'dagster.yaml'), 'r', encoding='utf8') as template_fd:
            with open(os.path.join(tempdir, 'dagster.yaml'), 'w', encoding='utf8') as target_fd:
                template = template_fd.read().format(hostname=hostname, port=port)
                target_fd.write(template)
        with DagsterInstance.from_config(tempdir) as instance:
            instance.upgrade()
            assert instance.schedule_storage.has_built_index(SCHEDULE_JOBS_SELECTOR_ID)
            legacy_count = len(instance.all_instigator_state())
            migrated_instigator_count = instance.schedule_storage.execute(db_select([db.func.count()]).select_from(InstigatorsTable))[0][0]
            assert migrated_instigator_count == legacy_count
            migrated_job_count = instance.schedule_storage.execute(db_select([db.func.count()]).select_from(JobTable).where(JobTable.c.selector_id.isnot(None)))[0][0]
            assert migrated_job_count == legacy_count
            legacy_tick_count = instance.schedule_storage.execute(db_select([db.func.count()]).select_from(JobTickTable))[0][0]
            assert legacy_tick_count > 0
            migrated_tick_count = instance.schedule_storage.execute(db_select([db.func.count()]).select_from(JobTickTable).where(JobTickTable.c.selector_id.isnot(None)))[0][0]
            assert migrated_tick_count == 0
            instance.reindex()
            migrated_tick_count = instance.schedule_storage.execute(db_select([db.func.count()]).select_from(JobTickTable).where(JobTickTable.c.selector_id.isnot(None)))[0][0]
            assert migrated_tick_count == legacy_tick_count

def test_add_bulk_actions_columns(conn_string):
    if False:
        i = 10
        return i + 15
    new_columns = {'selector_id', 'action_type'}
    new_indexes = {'idx_bulk_actions_action_type', 'idx_bulk_actions_selector_id'}
    (hostname, port) = _reconstruct_from_file(conn_string, file_relative_path(__file__, 'snapshot_0_14_6_post_schema_pre_data_migration.sql'))
    with tempfile.TemporaryDirectory() as tempdir:
        with open(file_relative_path(__file__, 'dagster.yaml'), 'r', encoding='utf8') as template_fd:
            with open(os.path.join(tempdir, 'dagster.yaml'), 'w', encoding='utf8') as target_fd:
                template = template_fd.read().format(hostname=hostname, port=port)
                target_fd.write(template)
        with DagsterInstance.from_config(tempdir) as instance:
            assert get_columns(instance, 'bulk_actions') & new_columns == set()
            assert get_indexes(instance, 'bulk_actions') & new_indexes == set()
            instance.upgrade()
            assert new_columns <= get_columns(instance, 'bulk_actions')
            assert new_indexes <= get_indexes(instance, 'bulk_actions')

def test_add_kvs_table(conn_string):
    if False:
        while True:
            i = 10
    (hostname, port) = _reconstruct_from_file(conn_string, file_relative_path(__file__, 'snapshot_0_14_6_post_schema_pre_data_migration.sql'))
    with tempfile.TemporaryDirectory() as tempdir:
        with open(file_relative_path(__file__, 'dagster.yaml'), 'r', encoding='utf8') as template_fd:
            with open(os.path.join(tempdir, 'dagster.yaml'), 'w', encoding='utf8') as target_fd:
                template = template_fd.read().format(hostname=hostname, port=port)
                target_fd.write(template)
        with DagsterInstance.from_config(tempdir) as instance:
            assert 'kvs' not in get_tables(instance)
            instance.upgrade()
            assert 'kvs' in get_tables(instance)
            assert 'idx_kvs_keys_unique' in get_indexes(instance, 'kvs')

def test_add_asset_event_tags_table(conn_string):
    if False:
        return 10

    @op
    def yields_materialization_w_tags(_):
        if False:
            return 10
        yield AssetMaterialization(asset_key=AssetKey(['a']), tags={'dagster/foo': 'bar'})
        yield Output(1)

    @job
    def asset_job():
        if False:
            print('Hello World!')
        yields_materialization_w_tags()
    (hostname, port) = _reconstruct_from_file(conn_string, file_relative_path(__file__, 'snapshot_1_0_12_pre_add_asset_event_tags_table.sql'))
    with tempfile.TemporaryDirectory() as tempdir:
        with open(file_relative_path(__file__, 'dagster.yaml'), 'r', encoding='utf8') as template_fd:
            with open(os.path.join(tempdir, 'dagster.yaml'), 'w', encoding='utf8') as target_fd:
                template = template_fd.read().format(hostname=hostname, port=port)
                target_fd.write(template)
        with DagsterInstance.from_config(tempdir) as instance:
            assert 'asset_event_tags' not in get_tables(instance)
            asset_job.execute_in_process(instance=instance)
            with pytest.raises(DagsterInvalidInvocationError, match='In order to search for asset event tags'):
                instance._event_storage.get_event_tags_for_asset(asset_key=AssetKey(['a']))
            assert len(instance.get_event_records(EventRecordsFilter(event_type=DagsterEventType.ASSET_MATERIALIZATION, asset_key=AssetKey('a'), tags={'dagster/foo': 'bar'}))) == 1
            mysql_version = instance._event_storage._mysql_version
            try:
                instance._event_storage._mysql_version = '8.0.30'
                assert len(instance.get_event_records(EventRecordsFilter(event_type=DagsterEventType.ASSET_MATERIALIZATION, asset_key=AssetKey('a'), tags={'dagster/foo': 'bar'}))) == 1
            finally:
                instance._event_storage._mysql_version = mysql_version
            instance.upgrade()
            assert 'asset_event_tags' in get_tables(instance)
            asset_job.execute_in_process(instance=instance)
            assert instance._event_storage.get_event_tags_for_asset(asset_key=AssetKey(['a'])) == [{'dagster/foo': 'bar'}]
            indexes = get_indexes(instance, 'asset_event_tags')
            assert 'idx_asset_event_tags' in indexes
            assert 'idx_asset_event_tags_event_id' in indexes

def test_add_cached_status_data_column(conn_string):
    if False:
        return 10
    new_columns = {'cached_status_data'}
    (hostname, port) = _reconstruct_from_file(conn_string, file_relative_path(__file__, 'snapshot_1_0_17_add_cached_status_data_column.sql'))
    with tempfile.TemporaryDirectory() as tempdir:
        with open(file_relative_path(__file__, 'dagster.yaml'), 'r', encoding='utf8') as template_fd:
            with open(os.path.join(tempdir, 'dagster.yaml'), 'w', encoding='utf8') as target_fd:
                template = template_fd.read().format(hostname=hostname, port=port)
                target_fd.write(template)
        with DagsterInstance.from_config(tempdir) as instance:
            assert get_columns(instance, 'asset_keys') & new_columns == set()
            instance.upgrade()
            assert new_columns <= get_columns(instance, 'asset_keys')

def test_add_dynamic_partitions_table(conn_string):
    if False:
        print('Hello World!')
    (hostname, port) = _reconstruct_from_file(conn_string, file_relative_path(__file__, 'snapshot_1_0_17_add_cached_status_data_column.sql'))
    with tempfile.TemporaryDirectory() as tempdir:
        with open(file_relative_path(__file__, 'dagster.yaml'), 'r', encoding='utf8') as template_fd:
            with open(os.path.join(tempdir, 'dagster.yaml'), 'w', encoding='utf8') as target_fd:
                template = template_fd.read().format(hostname=hostname, port=port)
                target_fd.write(template)
        with DagsterInstance.from_config(tempdir) as instance:
            assert 'dynamic_partitions' not in get_tables(instance)
            instance.wipe()
            with pytest.raises(DagsterInvalidInvocationError, match='does not exist'):
                instance.get_dynamic_partitions('foo')
            instance.upgrade()
            assert 'dynamic_partitions' in get_tables(instance)
            assert instance.get_dynamic_partitions('foo') == []

def _get_table_row_count(run_storage, table, with_non_null_id=False):
    if False:
        for i in range(10):
            print('nop')
    import sqlalchemy as db
    query = db_select([db.func.count()]).select_from(table)
    if with_non_null_id:
        query = query.where(table.c.id.isnot(None))
    with run_storage.connect() as conn:
        row_count = conn.execute(query).fetchone()[0]
    return row_count

def test_add_primary_keys(conn_string):
    if False:
        for i in range(10):
            print('nop')
    from dagster._core.storage.runs.schema import DaemonHeartbeatsTable, InstanceInfo, KeyValueStoreTable
    (hostname, port) = _reconstruct_from_file(conn_string, file_relative_path(__file__, 'snapshot_1_1_22_pre_primary_key.sql'))
    with tempfile.TemporaryDirectory() as tempdir:
        with open(file_relative_path(__file__, 'dagster.yaml'), 'r', encoding='utf8') as template_fd:
            with open(os.path.join(tempdir, 'dagster.yaml'), 'w', encoding='utf8') as target_fd:
                template = template_fd.read().format(hostname=hostname, port=port)
                target_fd.write(template)
        with DagsterInstance.from_config(tempdir) as instance:
            assert 'id' not in get_columns(instance, 'kvs')
            instance.run_storage.set_cursor_values({'a': 'A'})
            instance.run_storage.set_cursor_values({'a': 'A'})
            kvs_row_count = _get_table_row_count(instance.run_storage, KeyValueStoreTable)
            assert kvs_row_count > 0
            assert 'id' not in get_columns(instance, 'instance_info')
            instance_info_row_count = _get_table_row_count(instance.run_storage, InstanceInfo)
            assert instance_info_row_count > 0
            assert 'id' not in get_columns(instance, 'daemon_heartbeats')
            heartbeat = DaemonHeartbeat(timestamp=datetime.datetime.now().timestamp(), daemon_type='test', daemon_id='test')
            instance.run_storage.add_daemon_heartbeat(heartbeat)
            instance.run_storage.add_daemon_heartbeat(heartbeat)
            daemon_heartbeats_row_count = _get_table_row_count(instance.run_storage, DaemonHeartbeatsTable)
            assert daemon_heartbeats_row_count > 0
            instance.upgrade()
            assert 'id' in get_columns(instance, 'kvs')
            with instance.run_storage.connect():
                kvs_id_count = _get_table_row_count(instance.run_storage, KeyValueStoreTable, with_non_null_id=True)
            assert kvs_id_count == kvs_row_count
            assert 'id' in get_columns(instance, 'instance_info')
            with instance.run_storage.connect():
                instance_info_id_count = _get_table_row_count(instance.run_storage, InstanceInfo, with_non_null_id=True)
            assert instance_info_id_count == instance_info_row_count
            assert 'id' in get_columns(instance, 'daemon_heartbeats')
            with instance.run_storage.connect():
                daemon_heartbeats_id_count = _get_table_row_count(instance.run_storage, DaemonHeartbeatsTable, with_non_null_id=True)
            assert daemon_heartbeats_id_count == daemon_heartbeats_row_count

def test_bigint_migration(conn_string):
    if False:
        while True:
            i = 10
    (hostname, port) = _reconstruct_from_file(conn_string, file_relative_path(__file__, 'snapshot_1_1_22_pre_primary_key.sql'))

    def _get_integer_id_tables(conn):
        if False:
            while True:
                i = 10
        inspector = db.inspect(conn)
        integer_tables = set()
        for table in inspector.get_table_names():
            type_by_col_name = {c['name']: c['type'] for c in db.inspect(conn).get_columns(table)}
            id_type = type_by_col_name.get('id')
            if id_type and str(id_type) == 'INTEGER':
                integer_tables.add(table)
        return integer_tables
    with tempfile.TemporaryDirectory() as tempdir:
        with open(file_relative_path(__file__, 'dagster.yaml'), 'r', encoding='utf8') as template_fd:
            with open(os.path.join(tempdir, 'dagster.yaml'), 'w', encoding='utf8') as target_fd:
                template = template_fd.read().format(hostname=hostname, port=port)
                target_fd.write(template)
        with DagsterInstance.from_config(tempdir) as instance:
            with instance.run_storage.connect() as conn:
                assert len(_get_integer_id_tables(conn)) > 0
            with instance.event_log_storage.index_connection() as conn:
                assert len(_get_integer_id_tables(conn)) > 0
            with instance.schedule_storage.connect() as conn:
                assert len(_get_integer_id_tables(conn)) > 0
            run_bigint_migration(instance)
            with instance.run_storage.connect() as conn:
                assert len(_get_integer_id_tables(conn)) == 0
            with instance.event_log_storage.index_connection() as conn:
                assert len(_get_integer_id_tables(conn)) == 0
            with instance.schedule_storage.connect() as conn:
                assert len(_get_integer_id_tables(conn)) == 0