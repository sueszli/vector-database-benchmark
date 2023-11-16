from infi.clickhouse_orm import migrations
from posthog.client import sync_execute
from posthog.session_recordings.sql.session_recording_event_sql import MATERIALIZED_COLUMNS
from posthog.settings import CLICKHOUSE_CLUSTER

def create_events_summary_mat_columns(database):
    if False:
        i = 10
        return i + 15
    columns_to_add = ['events_summary', 'click_count', 'keypress_count', 'timestamps_summary', 'first_event_timestamp', 'last_event_timestamp', 'urls']
    for column in columns_to_add:
        data = MATERIALIZED_COLUMNS[column]
        sync_execute(f"\n            ALTER TABLE sharded_session_recording_events\n            ON CLUSTER '{CLICKHOUSE_CLUSTER}'\n            ADD COLUMN IF NOT EXISTS\n            {column} {data['schema']} {data['materializer']}\n        ")
        sync_execute(f"\n            ALTER TABLE session_recording_events\n            ON CLUSTER '{CLICKHOUSE_CLUSTER}'\n            ADD COLUMN IF NOT EXISTS\n            {column} {data['schema']}\n        ")
        sync_execute(f"\n                ALTER TABLE session_recording_events\n                ON CLUSTER '{CLICKHOUSE_CLUSTER}'\n                COMMENT COLUMN {column} 'column_materializer::{column}'\n            ")
operations = [migrations.RunPython(create_events_summary_mat_columns)]