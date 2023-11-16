from infi.clickhouse_orm import migrations
from posthog.clickhouse.materialized_columns import get_materialized_columns, materialize
from posthog.client import sync_execute
from posthog.settings import CLICKHOUSE_CLUSTER

def does_column_exist(database, table_name, column_name):
    if False:
        while True:
            i = 10
    cols = sync_execute(f"\n            SELECT 1\n            FROM system.columns\n            WHERE table = '{table_name}' AND name = '{column_name}' AND database = '{database}'\n        ")
    return len(cols) == 1

def ensure_only_new_column_exists(database, table_name, old_column_name, new_column_name):
    if False:
        while True:
            i = 10
    if does_column_exist(database, table_name, new_column_name):
        sync_execute(f"\n                ALTER TABLE {table_name}\n                ON CLUSTER '{CLICKHOUSE_CLUSTER}'\n                DROP COLUMN IF EXISTS {old_column_name}\n            ")
    else:
        sync_execute(f"\n                ALTER TABLE {table_name}\n                ON CLUSTER '{CLICKHOUSE_CLUSTER}'\n                RENAME COLUMN IF EXISTS {old_column_name} TO {new_column_name}\n            ")

def materialize_session_and_window_id(database):
    if False:
        return 10
    properties = ['$session_id', '$window_id']
    for property_name in properties:
        materialized_columns = get_materialized_columns('events', use_cache=False)
        if (property_name, 'properties') not in materialized_columns:
            materialize('events', property_name, property_name)
        possible_old_column_names = {'mat_' + property_name}
        current_materialized_column_name = materialized_columns.get(property_name, None)
        if current_materialized_column_name != property_name:
            possible_old_column_names.add(current_materialized_column_name)
        for possible_old_column_name in possible_old_column_names:
            ensure_only_new_column_exists(database, 'sharded_events', possible_old_column_name, property_name)
            ensure_only_new_column_exists(database, 'events', possible_old_column_name, property_name)
operations = [migrations.RunPython(materialize_session_and_window_id)]