from functools import cached_property
import structlog
from posthog.async_migrations.definition import AsyncMigrationDefinition, AsyncMigrationOperationSQL
from posthog.client import sync_execute
from posthog.constants import AnalyticsDBMS
from posthog.version_requirement import ServiceVersionRequirement
from datetime import datetime
from posthog.cloud_utils import is_cloud
logger = structlog.get_logger(__name__)

class Migration(AsyncMigrationDefinition):
    description = 'Move partitions with data in the future or far in the past to a backup table so we can exclude this data from queries and delete it later.'
    depends_on = '0009_minmax_indexes_for_materialized_columns'
    posthog_min_version = '1.43.0'
    posthog_max_version = '1.49.99'
    parameters = {'OLDEST_PARTITION_TO_KEEP': ('200001', 'ID of the oldest partition to keep', str), 'NEWEST_PARTITION_TO_KEEP': ('202308', 'ID of the newest partition to keep', str), 'OPTIMIZE_TABLE': (False, 'Optimize sharded_events table after moving partitions?', bool)}
    service_version_requirements = [ServiceVersionRequirement(service='clickhouse', supported_version='>=22.3.0')]

    def is_required(self) -> bool:
        if False:
            print('Hello World!')
        return is_cloud()

    def _get_partitions_to_move(self):
        if False:
            return 10
        result = sync_execute(f"\n            SELECT DISTINCT partition_id FROM system.parts\n            WHERE\n                table = 'sharded_events'\n                AND (\n                    partition < '{self.get_parameter('OLDEST_PARTITION_TO_KEEP')}'\n                    OR partition > '{self.get_parameter('NEWEST_PARTITION_TO_KEEP')}'\n                )\n            ORDER BY partition_id\n        ")
        return [row[0] for row in result]

    @cached_property
    def operations(self):
        if False:
            while True:
                i = 10
        now = datetime.now()
        suffix = f'{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'
        backup_table_name = f'events_backup_0010_move_old_partitions_{suffix}'
        shard = '{shard}'
        replica = '{replica}'
        operations = [AsyncMigrationOperationSQL(database=AnalyticsDBMS.CLICKHOUSE, sql=f"\n                CREATE TABLE {backup_table_name}\n                ON CLUSTER 'posthog'\n                AS sharded_events\n                ENGINE = ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/posthog.{backup_table_name}', '{replica}', _timestamp)\n                PARTITION BY toYYYYMM(timestamp)\n                ORDER BY (team_id, toDate(timestamp), event, cityHash64(distinct_id), cityHash64(uuid))\n                SAMPLE BY cityHash64(distinct_id)\n                SETTINGS index_granularity = 8192\n                ", rollback=None, per_shard=False)]
        partitions_to_move = self._get_partitions_to_move()
        for partition in partitions_to_move:
            operations.append(AsyncMigrationOperationSQL(database=AnalyticsDBMS.CLICKHOUSE, sql=f"\n                    ALTER TABLE sharded_events MOVE PARTITION '{partition}' TO TABLE {backup_table_name}\n                    ", per_shard=True, rollback=None))
        if self.get_parameter('OPTIMIZE_TABLE'):
            operations.append(AsyncMigrationOperationSQL(database=AnalyticsDBMS.CLICKHOUSE, sql=f'\n                    OPTIMIZE TABLE sharded_events FINAL\n                    ', per_shard=True, rollback=None))
        return operations