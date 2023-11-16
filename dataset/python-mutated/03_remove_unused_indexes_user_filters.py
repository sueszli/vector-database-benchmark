from synapse.config.homeserver import HomeServerConfig
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine, Sqlite3Engine

def run_update(cur: LoggingTransaction, database_engine: BaseDatabaseEngine, config: HomeServerConfig) -> None:
    if False:
        print('Hello World!')
    '\n    Fix to drop unused indexes caused by incorrectly adding UNIQUE constraint to\n    columns `user_id` and `full_user_id` of table `user_filters` in previous migration.\n    '
    if isinstance(database_engine, Sqlite3Engine):
        cur.execute('DROP TABLE IF EXISTS temp_user_filters')
        create_sql = '\n        CREATE TABLE temp_user_filters (\n            full_user_id text NOT NULL,\n            user_id text NOT NULL,\n            filter_id bigint NOT NULL,\n            filter_json bytea NOT NULL\n        )\n        '
        cur.execute(create_sql)
        copy_sql = '\n        INSERT INTO temp_user_filters (\n            user_id,\n            filter_id,\n            filter_json,\n            full_user_id)\n            SELECT user_id, filter_id, filter_json, full_user_id FROM user_filters\n        '
        cur.execute(copy_sql)
        drop_sql = '\n        DROP TABLE user_filters\n        '
        cur.execute(drop_sql)
        rename_sql = '\n        ALTER TABLE temp_user_filters RENAME to user_filters\n        '
        cur.execute(rename_sql)
        index_sql = '\n        CREATE UNIQUE INDEX IF NOT EXISTS user_filters_unique ON\n        user_filters (user_id, filter_id)\n        '
        cur.execute(index_sql)