"""
Main purpose of this upgrade is to change the unique key on the
pushers table again (it was missed when the v16 full schema was
made) but this also changes the pushkey and data columns to text.
When selecting a bytea column into a text column, postgres inserts
the hex encoded data, and there's no portable way of getting the
UTF-8 bytes, so we have to do it in Python.
"""
import logging
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine
logger = logging.getLogger(__name__)

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        return 10
    logger.info('Porting pushers table...')
    cur.execute('\n        CREATE TABLE IF NOT EXISTS pushers2 (\n          id BIGINT PRIMARY KEY,\n          user_name TEXT NOT NULL,\n          access_token BIGINT DEFAULT NULL,\n          profile_tag VARCHAR(32) NOT NULL,\n          kind VARCHAR(8) NOT NULL,\n          app_id VARCHAR(64) NOT NULL,\n          app_display_name VARCHAR(64) NOT NULL,\n          device_display_name VARCHAR(128) NOT NULL,\n          pushkey TEXT NOT NULL,\n          ts BIGINT NOT NULL,\n          lang VARCHAR(8),\n          data TEXT,\n          last_token TEXT,\n          last_success BIGINT,\n          failing_since BIGINT,\n          UNIQUE (app_id, pushkey, user_name)\n        )\n    ')
    cur.execute('SELECT\n        id, user_name, access_token, profile_tag, kind,\n        app_id, app_display_name, device_display_name,\n        pushkey, ts, lang, data, last_token, last_success,\n        failing_since\n        FROM pushers\n    ')
    count = 0
    for tuple_row in cur.fetchall():
        row = list(tuple_row)
        row[8] = bytes(row[8]).decode('utf-8')
        row[11] = bytes(row[11]).decode('utf-8')
        cur.execute('\n                INSERT into pushers2 (\n                id, user_name, access_token, profile_tag, kind,\n                app_id, app_display_name, device_display_name,\n                pushkey, ts, lang, data, last_token, last_success,\n                failing_since\n                ) values (%s)\n            ' % ','.join(['?' for _ in range(len(row))]), row)
        count += 1
    cur.execute('DROP TABLE pushers')
    cur.execute('ALTER TABLE pushers2 RENAME TO pushers')
    logger.info('Moved %d pushers to new table', count)