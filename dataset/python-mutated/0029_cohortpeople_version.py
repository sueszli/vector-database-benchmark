from infi.clickhouse_orm import migrations
from posthog.client import sync_execute
from posthog.settings import CLICKHOUSE_CLUSTER
ADD_COLUMNS_BASE_SQL = "\nALTER TABLE {table}\nON CLUSTER '{cluster}'\nADD COLUMN IF NOT EXISTS version UInt64,\nMODIFY ORDER BY (team_id, cohort_id, person_id, version)\n"

def add_columns_to_required_tables(_):
    if False:
        return 10
    sync_execute(ADD_COLUMNS_BASE_SQL.format(table='cohortpeople', cluster=CLICKHOUSE_CLUSTER))
operations = [migrations.RunPython(add_columns_to_required_tables)]