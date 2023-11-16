import logging
from io import StringIO
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine, PostgresEngine
from synapse.storage.prepare_database import execute_statements_from_stream
logger = logging.getLogger(__name__)
'\nThis migration updates the user_filters table as follows:\n\n - drops any (user_id, filter_id) duplicates\n - makes the columns NON-NULLable\n - turns the index into a UNIQUE index\n'

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        print('Hello World!')
    if isinstance(database_engine, PostgresEngine):
        select_clause = '\n            SELECT DISTINCT ON (user_id, filter_id) user_id, filter_id, filter_json\n            FROM user_filters\n        '
    else:
        select_clause = '\n            SELECT * FROM user_filters GROUP BY user_id, filter_id\n        '
    sql = '\n            DROP TABLE IF EXISTS user_filters_migration;\n            DROP INDEX IF EXISTS user_filters_unique;\n            CREATE TABLE user_filters_migration (\n                user_id TEXT NOT NULL,\n                filter_id BIGINT NOT NULL,\n                filter_json BYTEA NOT NULL\n            );\n            INSERT INTO user_filters_migration (user_id, filter_id, filter_json)\n                %s;\n            CREATE UNIQUE INDEX user_filters_unique ON user_filters_migration\n                (user_id, filter_id);\n            DROP TABLE user_filters;\n            ALTER TABLE user_filters_migration RENAME TO user_filters;\n        ' % (select_clause,)
    execute_statements_from_stream(cur, StringIO(sql))