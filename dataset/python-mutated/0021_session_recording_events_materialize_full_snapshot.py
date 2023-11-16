from infi.clickhouse_orm import migrations
from posthog.client import sync_execute
from posthog.settings import CLICKHOUSE_CLUSTER
SESSION_RECORDING_EVENTS_MATERIALIZED_COLUMN_COMMENTS_SQL = lambda : "\n    ALTER TABLE session_recording_events\n    ON CLUSTER '{cluster}'\n    COMMENT COLUMN has_full_snapshot 'column_materializer::has_full_snapshot'\n".format(cluster=CLICKHOUSE_CLUSTER)

def create_has_full_snapshot_materialized_column(database):
    if False:
        while True:
            i = 10
    sync_execute(f"\n        ALTER TABLE sharded_session_recording_events\n        ON CLUSTER '{CLICKHOUSE_CLUSTER}'\n        ADD COLUMN IF NOT EXISTS\n        has_full_snapshot Int8 MATERIALIZED JSONExtractBool(snapshot_data, 'has_full_snapshot')\n    ")
    sync_execute(f"\n        ALTER TABLE session_recording_events\n        ON CLUSTER '{CLICKHOUSE_CLUSTER}'\n        ADD COLUMN IF NOT EXISTS\n        has_full_snapshot Int8\n    ")
    sync_execute(SESSION_RECORDING_EVENTS_MATERIALIZED_COLUMN_COMMENTS_SQL())
operations = [migrations.RunPython(create_has_full_snapshot_materialized_column)]