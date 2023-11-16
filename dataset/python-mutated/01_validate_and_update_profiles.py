from synapse.config.homeserver import HomeServerConfig
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine, PostgresEngine

def run_upgrade(cur: LoggingTransaction, database_engine: BaseDatabaseEngine, config: HomeServerConfig) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Part 3 of a multi-step migration to drop the column `user_id` and replace it with\n    `full_user_id`. See the database schema docs for more information on the full\n    migration steps.\n    '
    hostname = config.server.server_name
    if isinstance(database_engine, PostgresEngine):
        check_sql = '\n        SELECT user_id from profiles WHERE full_user_id IS NULL\n        '
        cur.execute(check_sql)
        res = cur.fetchall()
        if res:
            process_rows_sql = "\n            UPDATE profiles\n            SET full_user_id = '@' || user_id || ?\n            WHERE user_id IN (\n                SELECT user_id FROM profiles WHERE full_user_id IS NULL\n            )\n            "
            cur.execute(process_rows_sql, (f':{hostname}',))
        validate_sql = '\n        ALTER TABLE profiles VALIDATE CONSTRAINT full_user_id_not_null\n        '
        cur.execute(validate_sql)
    else:
        cur.execute('DROP TABLE IF EXISTS temp_profiles')
        create_sql = '\n        CREATE TABLE temp_profiles (\n            full_user_id text NOT NULL,\n            user_id text,\n            displayname text,\n            avatar_url text,\n            UNIQUE (full_user_id),\n            UNIQUE (user_id)\n        )\n        '
        cur.execute(create_sql)
        copy_sql = "\n        INSERT INTO temp_profiles (\n            user_id,\n            displayname,\n            avatar_url,\n            full_user_id)\n            SELECT user_id, displayname, avatar_url, '@' || user_id || ':' || ? FROM profiles\n        "
        cur.execute(copy_sql, (f'{hostname}',))
        drop_sql = '\n        DROP TABLE profiles\n        '
        cur.execute(drop_sql)
        rename_sql = '\n        ALTER TABLE temp_profiles RENAME to profiles\n        '
        cur.execute(rename_sql)