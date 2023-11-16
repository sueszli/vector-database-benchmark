from synapse.config.homeserver import HomeServerConfig
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine, PostgresEngine

def run_upgrade(cur: LoggingTransaction, database_engine: BaseDatabaseEngine, config: HomeServerConfig) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Part 3 of a multi-step migration to drop the column `user_id` and replace it with\n    `full_user_id`. See the database schema docs for more information on the full\n    migration steps.\n    '
    hostname = config.server.server_name
    if isinstance(database_engine, PostgresEngine):
        check_sql = '\n        SELECT user_id from user_filters WHERE full_user_id IS NULL\n        '
        cur.execute(check_sql)
        res = cur.fetchall()
        if res:
            process_rows_sql = "\n            UPDATE user_filters\n            SET full_user_id = '@' || user_id || ?\n            WHERE user_id IN (\n                SELECT user_id FROM user_filters WHERE full_user_id IS NULL\n            )\n            "
            cur.execute(process_rows_sql, (f':{hostname}',))
        validate_sql = '\n        ALTER TABLE user_filters VALIDATE CONSTRAINT full_user_id_not_null\n        '
        cur.execute(validate_sql)
    else:
        cur.execute('DROP TABLE IF EXISTS temp_user_filters')
        create_sql = '\n        CREATE TABLE temp_user_filters (\n            full_user_id text NOT NULL,\n            user_id text NOT NULL,\n            filter_id bigint NOT NULL,\n            filter_json bytea NOT NULL\n        )\n        '
        cur.execute(create_sql)
        index_sql = '\n        CREATE UNIQUE INDEX IF NOT EXISTS user_filters_unique ON\n            temp_user_filters (user_id, filter_id)\n        '
        cur.execute(index_sql)
        copy_sql = "\n        INSERT INTO temp_user_filters (\n            user_id,\n            filter_id,\n            filter_json,\n            full_user_id)\n            SELECT user_id, filter_id, filter_json, '@' || user_id || ':' || ? FROM user_filters\n        "
        cur.execute(copy_sql, (f'{hostname}',))
        drop_sql = '\n        DROP TABLE user_filters\n        '
        cur.execute(drop_sql)
        rename_sql = '\n        ALTER TABLE temp_user_filters RENAME to user_filters\n        '
        cur.execute(rename_sql)