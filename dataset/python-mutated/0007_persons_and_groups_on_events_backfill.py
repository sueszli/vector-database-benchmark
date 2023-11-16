from functools import cached_property
from typing import Dict, Tuple, Union
import structlog
from django.conf import settings
from posthog.async_migrations.definition import AsyncMigrationDefinition, AsyncMigrationOperation, AsyncMigrationOperationSQL
from posthog.async_migrations.disk_util import analyze_enough_disk_space_free_for_table
from posthog.async_migrations.utils import execute_op_clickhouse, run_optimize_table, sleep_until_finished
from posthog.client import sync_execute
from posthog.models.event.sql import EVENTS_DATA_TABLE
from posthog.utils import str_to_bool
logger = structlog.get_logger(__name__)
'\nMigration summary\n=================\n\nBackfill the sharded_events table to add data for the following columns:\n\n- person_id\n- person_properties\n- person_created_at\n- groupX_properties\n- groupX_created_at\n\nThis allows us to switch entirely to only querying the events table for insights,\nwithout having to hit additional tables for persons, groups, and distinct IDs.\n\nMigration strategy\n==================\n\nWe will run the following operations on the cluster\n(or on one node per shard if shard-level configuration is provided):\n\n1. Update `person_properties` and `groupX_properties` columns to use ZSTD(3) compression\n2. Create temporary tables with the relevant columns from `person`, `person_distinct_id`, and `groups`\n3. Copy data from the main tables into them\n4. Optimize the temporary tables to remove duplicates and remove deleted data\n5. Create a dictionary to query each temporary table with caching\n6. Run an ALTER TABLE ... UPDATE to backfill all the data using the dictionaries\n\nConstraints\n===========\n\n1. The migration requires a lot of extra space for the new columns. At least 2x disk space is required to avoid issues while migrating.\n2. We use ZSTD(3) compression on the new columns to save on space and speed up large reads.\n3. New columns need to be populated for new rows before running this async migration.\n'
TEMPORARY_PERSONS_TABLE_NAME = 'tmp_person_0007'
TEMPORARY_PDI2_TABLE_NAME = 'tmp_person_distinct_id2_0007'
TEMPORARY_GROUPS_TABLE_NAME = 'tmp_groups_0007'
STORAGE_POLICY_SETTING = lambda : ", storage_policy = 'hot_to_cold'" if settings.CLICKHOUSE_ENABLE_STORAGE_POLICY else ''
DEFAULT_ACCEPTED_INCONSISTENT_DATA_RATIO = 0.01

class Migration(AsyncMigrationDefinition):
    description = 'Backfill persons and groups data on the sharded_events table'
    depends_on = '0006_persons_and_groups_on_events_backfill'
    posthog_min_version = '1.40.0'
    posthog_max_version = '1.41.99'
    parameters = {'PERSON_DICT_CACHE_SIZE': (5000000, 'ClickHouse cache size (in rows) for persons data.', int), 'PERSON_DISTINCT_ID_DICT_CACHE_SIZE': (5000000, 'ClickHouse cache size (in rows) for person distinct id data.', int), 'GROUPS_DICT_CACHE_SIZE': (1000000, 'ClickHouse cache size (in rows) for groups data.', int), 'RUN_DATA_VALIDATION_POSTCHECK': ('True', 'Whether to run a postcheck validating the backfilled data.', str), 'TIMESTAMP_LOWER_BOUND': ('2020-01-01', 'Timestamp lower bound for events to backfill', str), 'TIMESTAMP_UPPER_BOUND': ('2024-01-01', 'Timestamp upper bound for events to backfill', str), 'TEAM_ID': (None, 'The team_id of team to run backfill for. If unset the backfill will run for all teams.', int)}

    def precheck(self):
        if False:
            i = 10
            return i + 15
        return analyze_enough_disk_space_free_for_table(EVENTS_DATA_TABLE(), required_ratio=2.0)

    def is_required(self) -> bool:
        if False:
            while True:
                i = 10
        rows_to_backfill_check = sync_execute("\n            SELECT 1\n            FROM events\n            WHERE\n                empty(person_id) OR\n                person_created_at = toDateTime(0) OR\n                person_properties = '' OR\n                group0_properties = '' OR\n                group1_properties = '' OR\n                group2_properties = '' OR\n                group3_properties = '' OR\n                group4_properties = ''\n            LIMIT 1\n            ")
        return len(rows_to_backfill_check) > 0

    @cached_property
    def operations(self):
        if False:
            for i in range(10):
                print('nop')
        return [AsyncMigrationOperation(fn=lambda query_id: self._update_properties_column_compression_codec(query_id, 'ZSTD(3)'), rollback_fn=lambda query_id: self._update_properties_column_compression_codec(query_id, 'LZ4')), AsyncMigrationOperationSQL(sql=f'\n                    CREATE TABLE {settings.CLICKHOUSE_DATABASE}.{TEMPORARY_PERSONS_TABLE_NAME} {{on_cluster_clause}}\n                    AS {settings.CLICKHOUSE_DATABASE}.person\n                    ENGINE = ReplacingMergeTree(version)\n                    ORDER BY (team_id, id)\n                    SETTINGS index_granularity = 128 {STORAGE_POLICY_SETTING()}\n                ', rollback=f'DROP TABLE IF EXISTS {TEMPORARY_PERSONS_TABLE_NAME} {{on_cluster_clause}}', per_shard=True), AsyncMigrationOperationSQL(sql=f'\n                    CREATE TABLE {settings.CLICKHOUSE_DATABASE}.{TEMPORARY_PDI2_TABLE_NAME} {{on_cluster_clause}}\n                    AS {settings.CLICKHOUSE_DATABASE}.person_distinct_id2\n                    ENGINE = ReplacingMergeTree(version)\n                    ORDER BY (team_id, distinct_id)\n                    SETTINGS index_granularity = 128\n                ', rollback=f'DROP TABLE IF EXISTS {TEMPORARY_PDI2_TABLE_NAME} {{on_cluster_clause}}', per_shard=True), AsyncMigrationOperationSQL(sql=f'\n                    CREATE TABLE {settings.CLICKHOUSE_DATABASE}.{TEMPORARY_GROUPS_TABLE_NAME} {{on_cluster_clause}}\n                    AS {settings.CLICKHOUSE_DATABASE}.groups\n                    ENGINE = ReplacingMergeTree(_timestamp)\n                    ORDER BY (team_id, group_type_index, group_key)\n                    SETTINGS index_granularity = 128 {STORAGE_POLICY_SETTING()}\n                ', rollback=f'DROP TABLE IF EXISTS {TEMPORARY_GROUPS_TABLE_NAME} {{on_cluster_clause}}', per_shard=True), AsyncMigrationOperationSQL(sql=f'\n                    ALTER TABLE {settings.CLICKHOUSE_DATABASE}.{TEMPORARY_PERSONS_TABLE_NAME} {{on_cluster_clause}}\n                    REPLACE PARTITION tuple() FROM {settings.CLICKHOUSE_DATABASE}.person\n                ', rollback=None, per_shard=True), AsyncMigrationOperationSQL(sql=f'\n                    ALTER TABLE {settings.CLICKHOUSE_DATABASE}.{TEMPORARY_PDI2_TABLE_NAME} {{on_cluster_clause}}\n                    REPLACE PARTITION tuple() FROM {settings.CLICKHOUSE_DATABASE}.person_distinct_id2\n                ', rollback=None, per_shard=True), AsyncMigrationOperationSQL(sql=f'\n                    ALTER TABLE {settings.CLICKHOUSE_DATABASE}.{TEMPORARY_GROUPS_TABLE_NAME} {{on_cluster_clause}}\n                    REPLACE PARTITION tuple() FROM {settings.CLICKHOUSE_DATABASE}.groups\n                ', rollback=None, per_shard=True), AsyncMigrationOperation(fn=lambda query_id: run_optimize_table(unique_name='0007_persons_and_groups_on_events_backfill_person', query_id=query_id, table_name=f'{settings.CLICKHOUSE_DATABASE}.{TEMPORARY_PERSONS_TABLE_NAME}', final=True, deduplicate=True, per_shard=True)), AsyncMigrationOperation(fn=lambda query_id: run_optimize_table(unique_name='0007_persons_and_groups_on_events_backfill_pdi2', query_id=query_id, table_name=f'{settings.CLICKHOUSE_DATABASE}.{TEMPORARY_PDI2_TABLE_NAME}', final=True, deduplicate=True, per_shard=True)), AsyncMigrationOperation(fn=lambda query_id: run_optimize_table(unique_name='0007_persons_and_groups_on_events_backfill_groups', query_id=query_id, table_name=f'{settings.CLICKHOUSE_DATABASE}.{TEMPORARY_GROUPS_TABLE_NAME}', final=True, deduplicate=True, per_shard=True)), AsyncMigrationOperationSQL(sql=f'\n                    ALTER TABLE {settings.CLICKHOUSE_DATABASE}.{TEMPORARY_PDI2_TABLE_NAME} {{on_cluster_clause}}\n                    DELETE WHERE is_deleted = 1 OR person_id IN (\n                        SELECT id FROM {TEMPORARY_PERSONS_TABLE_NAME} WHERE is_deleted=1\n                    )\n                ', sql_settings={'mutations_sync': 2}, rollback=None, per_shard=True), AsyncMigrationOperationSQL(sql=f'\n                    ALTER TABLE {settings.CLICKHOUSE_DATABASE}.{TEMPORARY_PERSONS_TABLE_NAME} {{on_cluster_clause}}\n                    DELETE WHERE is_deleted = 1\n                ', sql_settings={'mutations_sync': 2}, rollback=None, per_shard=True), AsyncMigrationOperation(fn=self._create_dictionaries, rollback_fn=self._clear_temporary_tables), AsyncMigrationOperation(fn=self._run_backfill_mutation), AsyncMigrationOperation(fn=self._wait_for_mutation_done), AsyncMigrationOperation(fn=lambda query_id: self._postcheck(query_id)), AsyncMigrationOperation(fn=self._clear_temporary_tables)]

    def _dictionary_connection_string(self):
        if False:
            i = 10
            return i + 15
        result = f"DB '{settings.CLICKHOUSE_DATABASE}'"
        if settings.CLICKHOUSE_USER:
            result += f" USER '{settings.CLICKHOUSE_USER}'"
        if settings.CLICKHOUSE_PASSWORD:
            result += f" PASSWORD '{settings.CLICKHOUSE_PASSWORD}'"
        return result

    def _update_properties_column_compression_codec(self, query_id, codec):
        if False:
            return 10
        columns = ['person_properties', 'group0_properties', 'group1_properties', 'group2_properties', 'group3_properties', 'group4_properties']
        for column in columns:
            execute_op_clickhouse(query_id=query_id, sql=f"ALTER TABLE {settings.CLICKHOUSE_DATABASE}.{EVENTS_DATA_TABLE()} ON CLUSTER '{settings.CLICKHOUSE_CLUSTER}' MODIFY COLUMN {column} VARCHAR Codec({codec})")

    def _postcheck(self, _: str):
        if False:
            i = 10
            return i + 15
        if str_to_bool(self.get_parameter('RUN_DATA_VALIDATION_POSTCHECK')):
            self._check_person_data()
            self._check_groups_data()

    def _where_clause(self) -> Tuple[str, Dict[str, Union[str, int]]]:
        if False:
            i = 10
            return i + 15
        team_id = self.get_parameter('TEAM_ID')
        team_id_filter = f' AND team_id = %(team_id)s' if team_id else ''
        where_clause = f'WHERE timestamp > toDateTime(%(timestamp_lower_bound)s) AND timestamp < toDateTime(%(timestamp_upper_bound)s) {team_id_filter}'
        return (where_clause, {'team_id': team_id, 'timestamp_lower_bound': self.get_parameter('TIMESTAMP_LOWER_BOUND'), 'timestamp_upper_bound': self.get_parameter('TIMESTAMP_UPPER_BOUND')})

    def _check_person_data(self, threshold=DEFAULT_ACCEPTED_INCONSISTENT_DATA_RATIO):
        if False:
            i = 10
            return i + 15
        (where_clause, where_clause_params) = self._where_clause()
        incomplete_person_data_ratio = sync_execute(f"\n            SELECT countIf(\n                empty(person_id) OR\n                person_created_at = toDateTime(0) OR\n                person_properties = ''\n            ) / count() FROM events\n            SAMPLE 10000000\n            {where_clause}\n            ", where_clause_params)[0][0]
        if incomplete_person_data_ratio > threshold:
            incomplete_events_percentage = incomplete_person_data_ratio * 100
            raise Exception(f'Backfill did not work succesfully. ~{int(incomplete_events_percentage)}% of events did not get the correct data for persons.')

    def _check_groups_data(self, threshold=DEFAULT_ACCEPTED_INCONSISTENT_DATA_RATIO):
        if False:
            print('Hello World!')
        (where_clause, where_clause_params) = self._where_clause()
        incomplete_groups_data_ratio = sync_execute(f"\n            SELECT countIf(\n                group0_properties = '' OR\n                group0_created_at != dictGetDateTime('{settings.CLICKHOUSE_DATABASE}.groups_dict', 'created_at', tuple(team_id, 0, $group_0)) OR\n                group1_properties = '' OR\n                group1_created_at != dictGetDateTime('{settings.CLICKHOUSE_DATABASE}.groups_dict', 'created_at', tuple(team_id, 1, $group_1)) OR\n                group2_properties = '' OR\n                group2_created_at != dictGetDateTime('{settings.CLICKHOUSE_DATABASE}.groups_dict', 'created_at', tuple(team_id, 2, $group_2)) OR\n                group3_properties = '' OR\n                group3_created_at != dictGetDateTime('{settings.CLICKHOUSE_DATABASE}.groups_dict', 'created_at', tuple(team_id, 3, $group_3)) OR\n                group4_properties = '' OR\n                group4_created_at != dictGetDateTime('{settings.CLICKHOUSE_DATABASE}.groups_dict', 'created_at', tuple(team_id, 4, $group_4))\n            ) / count() FROM events\n            SAMPLE 10000000\n            {where_clause}\n            ", where_clause_params)[0][0]
        if incomplete_groups_data_ratio > threshold:
            incomplete_events_percentage = incomplete_groups_data_ratio * 100
            raise Exception(f'Backfill did not work succesfully. ~{int(incomplete_events_percentage)}% of events did not get the correct data for groups.')

    def _run_backfill_mutation(self, query_id):
        if False:
            print('Hello World!')
        if self._count_running_mutations() > 0:
            return
        (where_clause, where_clause_params) = self._where_clause()
        execute_op_clickhouse(f"\n                ALTER TABLE {EVENTS_DATA_TABLE()}\n                {{on_cluster_clause}}\n                UPDATE\n                    person_id = if(\n                        empty(person_id),\n                        toUUID(dictGet('{settings.CLICKHOUSE_DATABASE}.person_distinct_id2_dict', 'person_id', tuple(team_id, distinct_id))),\n                        person_id\n                    ),\n                    person_properties = if(\n                        person_properties = '',\n                        dictGetStringOrDefault(\n                            '{settings.CLICKHOUSE_DATABASE}.person_dict',\n                            'properties',\n                            tuple(\n                                team_id,\n                                toUUID(dictGet('{settings.CLICKHOUSE_DATABASE}.person_distinct_id2_dict', 'person_id', tuple(team_id, distinct_id)))\n                            ),\n                            toJSONString(map())\n                        ),\n                        person_properties\n                    ),\n                    person_created_at = if(\n                        person_created_at = toDateTime(0),\n                        dictGetDateTime(\n                            '{settings.CLICKHOUSE_DATABASE}.person_dict',\n                            'created_at',\n                            tuple(\n                                team_id,\n                                toUUID(dictGet('{settings.CLICKHOUSE_DATABASE}.person_distinct_id2_dict', 'person_id', tuple(team_id, distinct_id)))\n                            )\n                        ),\n                        person_created_at\n                    ),\n                    group0_properties = if(\n                        group0_properties = '',\n                        dictGetStringOrDefault('{settings.CLICKHOUSE_DATABASE}.groups_dict', 'group_properties', tuple(team_id, 0, $group_0), toJSONString(map())),\n                        group0_properties\n                    ),\n                    group1_properties = if(\n                        group1_properties = '',\n                        dictGetStringOrDefault('{settings.CLICKHOUSE_DATABASE}.groups_dict', 'group_properties', tuple(team_id, 1, $group_1), toJSONString(map())),\n                        group1_properties\n                    ),\n                    group2_properties = if(\n                        group2_properties = '',\n                        dictGetStringOrDefault('{settings.CLICKHOUSE_DATABASE}.groups_dict', 'group_properties', tuple(team_id, 2, $group_2), toJSONString(map())),\n                        group2_properties\n                    ),\n                    group3_properties = if(\n                        group3_properties = '',\n                        dictGetStringOrDefault('{settings.CLICKHOUSE_DATABASE}.groups_dict', 'group_properties', tuple(team_id, 3, $group_3), toJSONString(map())),\n                        group3_properties\n                    ),\n                    group4_properties = if(\n                        group4_properties = '',\n                        dictGetStringOrDefault('{settings.CLICKHOUSE_DATABASE}.groups_dict', 'group_properties', tuple(team_id, 4, $group_4), toJSONString(map())),\n                        group4_properties\n                    ),\n                    group0_created_at = if(\n                        group0_created_at = toDateTime(0),\n                        dictGetDateTime('{settings.CLICKHOUSE_DATABASE}.groups_dict', 'created_at', tuple(team_id, 0, $group_0)),\n                        group0_created_at\n                    ),\n                    group1_created_at = if(\n                        group1_created_at = toDateTime(0),\n                        dictGetDateTime('{settings.CLICKHOUSE_DATABASE}.groups_dict', 'created_at', tuple(team_id, 1, $group_1)),\n                        group1_created_at\n                    ),\n                    group2_created_at = if(\n                        group2_created_at = toDateTime(0),\n                        dictGetDateTime('{settings.CLICKHOUSE_DATABASE}.groups_dict', 'created_at', tuple(team_id, 2, $group_2)),\n                        group2_created_at\n                    ),\n                    group3_created_at = if(\n                        group3_created_at = toDateTime(0),\n                        dictGetDateTime('{settings.CLICKHOUSE_DATABASE}.groups_dict', 'created_at', tuple(team_id, 3, $group_3)),\n                        group3_created_at\n                    ),\n                    group4_created_at = if(\n                        group4_created_at = toDateTime(0),\n                        dictGetDateTime('{settings.CLICKHOUSE_DATABASE}.groups_dict', 'created_at', tuple(team_id, 4, $group_4)),\n                        group4_created_at\n                    )\n                {where_clause}\n            ", where_clause_params, settings={'max_execution_time': 0}, per_shard=True, query_id=query_id)

    def _create_dictionaries(self, query_id):
        if False:
            while True:
                i = 10
        (execute_op_clickhouse(f'\n                CREATE DICTIONARY IF NOT EXISTS {settings.CLICKHOUSE_DATABASE}.person_dict {{on_cluster_clause}}\n                (\n                    team_id Int64,\n                    id UUID,\n                    properties String,\n                    created_at DateTime\n                )\n                PRIMARY KEY team_id, id\n                SOURCE(CLICKHOUSE(TABLE {TEMPORARY_PERSONS_TABLE_NAME} {self._dictionary_connection_string()}))\n                LAYOUT(complex_key_cache(size_in_cells %(cache_size)s max_threads_for_updates 6 allow_read_expired_keys 1))\n                Lifetime(60000)\n            ', {'cache_size': self.get_parameter('PERSON_DICT_CACHE_SIZE')}, per_shard=True, query_id=query_id),)
        (execute_op_clickhouse(f'\n                CREATE DICTIONARY IF NOT EXISTS {settings.CLICKHOUSE_DATABASE}.person_distinct_id2_dict {{on_cluster_clause}}\n                (\n                    team_id Int64,\n                    distinct_id String,\n                    person_id UUID\n                )\n                PRIMARY KEY team_id, distinct_id\n                SOURCE(CLICKHOUSE(TABLE {TEMPORARY_PDI2_TABLE_NAME} {self._dictionary_connection_string()}))\n                LAYOUT(complex_key_cache(size_in_cells %(cache_size)s max_threads_for_updates 6 allow_read_expired_keys 1))\n                Lifetime(60000)\n            ', {'cache_size': self.get_parameter('PERSON_DISTINCT_ID_DICT_CACHE_SIZE')}, per_shard=True, query_id=query_id),)
        execute_op_clickhouse(f'\n                CREATE DICTIONARY IF NOT EXISTS {settings.CLICKHOUSE_DATABASE}.groups_dict {{on_cluster_clause}}\n                (\n                    team_id Int64,\n                    group_type_index UInt8,\n                    group_key String,\n                    group_properties String,\n                    created_at DateTime\n                )\n                PRIMARY KEY team_id, group_type_index, group_key\n                SOURCE(CLICKHOUSE(TABLE {TEMPORARY_GROUPS_TABLE_NAME} {self._dictionary_connection_string()}))\n                LAYOUT(complex_key_cache(size_in_cells %(cache_size)s max_threads_for_updates 6 allow_read_expired_keys 1))\n                Lifetime(60000)\n            ', {'cache_size': self.get_parameter('GROUPS_DICT_CACHE_SIZE')}, per_shard=True, query_id=query_id)

    def _wait_for_mutation_done(self, query_id):
        if False:
            while True:
                i = 10
        sleep_until_finished('events table backill', lambda : self._count_running_mutations() > 0)

    def _count_running_mutations(self):
        if False:
            print('Hello World!')
        return sync_execute("\n            SELECT count()\n            FROM clusterAllReplicas(%(cluster)s, system, 'mutations')\n            WHERE not is_done AND command LIKE %(pattern)s\n            ", {'cluster': settings.CLICKHOUSE_CLUSTER, 'pattern': '%person_created_at = toDateTime(0)%'})[0][0]

    def _clear_temporary_tables(self, query_id):
        if False:
            while True:
                i = 10
        queries = [f'DROP DICTIONARY IF EXISTS {settings.CLICKHOUSE_DATABASE}.person_dict {{on_cluster_clause}}', f'DROP DICTIONARY IF EXISTS {settings.CLICKHOUSE_DATABASE}.person_distinct_id2_dict {{on_cluster_clause}}', f'DROP DICTIONARY IF EXISTS {settings.CLICKHOUSE_DATABASE}.groups_dict {{on_cluster_clause}}', f'DROP TABLE IF EXISTS {settings.CLICKHOUSE_DATABASE}.{TEMPORARY_PERSONS_TABLE_NAME} {{on_cluster_clause}}', f'DROP TABLE IF EXISTS {settings.CLICKHOUSE_DATABASE}.{TEMPORARY_PDI2_TABLE_NAME} {{on_cluster_clause}}', f'DROP TABLE IF EXISTS {settings.CLICKHOUSE_DATABASE}.{TEMPORARY_GROUPS_TABLE_NAME} {{on_cluster_clause}}']
        for query in queries:
            execute_op_clickhouse(query_id=query_id, sql=query, per_shard=True)

    def healthcheck(self):
        if False:
            print('Hello World!')
        result = sync_execute('SELECT free_space FROM system.disks')
        if int(result[0][0]) < 100000000:
            return (False, 'ClickHouse available storage below 100MB')
        return (True, None)