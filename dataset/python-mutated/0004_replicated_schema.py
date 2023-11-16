from typing import Optional, cast
import structlog
from django.conf import settings
from posthog.async_migrations.definition import AsyncMigrationDefinition
from posthog.client import sync_execute
logger = structlog.get_logger(__name__)
"\nMigration summary:\n\nSchema change to migrate tables to support replication and more than one shard.\n\nThis allows for higher scalability as more hosts can be added under ClickHouse.\n\nThe migration strategy:\n\n    1. We have a list of tables that might need replacing below.\n    2. For each one, we replace the current engine with the appropriate Replicated by:\n        a. creating a new table with the right engine and identical schema\n        b. temporarily stopping ingestion to the table by dropping the kafka table\n        c. using `ALTER TABLE ATTACH/DROP PARTITIONS` to move data to the new table.\n        d. rename tables\n    3. Once all tables are updated, we create the required distributed tables and re-enable ingestion\n\nWe use ATTACH/DROP tables to do the table migration instead of a normal INSERT. This method allows\nmoving data without increasing disk usage between identical schemas.\n\n`events` and `session_recording_events` require extra steps as they're also sharded:\n\n    1. The new table should be named `sharded_TABLENAME`\n    2. When re-enabling ingestion, we create `TABLENAME` and `writable_TABLENAME` tables\n       which are responsible for distributed reads and writes\n    3. We re-create materialized views to write to `writable_TABLENAME`\n\nConstraints:\n\n    1. This migration relies on there being exactly one ClickHouse node when it's run.\n    2. For person and events tables, the schema tries to preserve any materialized columns.\n    3. This migration requires there to be no ongoing part merges while it's executing.\n    4. This migration depends on 0002_events_sample_by. If it didn't, this could be a normal migration.\n    5. This migration depends on the person_distinct_id2 async migration to have completed.\n    6. We can't stop ingestion by dropping/detaching materialized view as we can't restore to the right (non-replicated) schema afterwards.\n    7. Async migrations might fail _before_ a step executes and rollbacks need to account for that, which complicates renaming logic.\n    8. For person_distinct_id2 table moving parts might fail due to upstream issues with zookeeper parts being created automatically. We retry up to 3 times.\n"

class Migration(AsyncMigrationDefinition):
    description = 'Replace tables with replicated counterparts'
    depends_on = '0003_fill_person_distinct_id2'
    posthog_min_version = '1.36.1'
    posthog_max_version = '1.36.99'

    def is_required(self):
        if False:
            print('Hello World!')
        return 'Distributed' not in cast(str, self.get_current_engine('events'))

    def get_current_engine(self, table_name: str) -> Optional[str]:
        if False:
            while True:
                i = 10
        result = sync_execute('SELECT engine_full FROM system.tables WHERE database = %(database)s AND name = %(name)s', {'database': settings.CLICKHOUSE_DATABASE, 'name': table_name})
        return result[0][0] if len(result) > 0 else None