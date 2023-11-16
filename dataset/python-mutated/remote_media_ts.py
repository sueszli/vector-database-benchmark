import time
from synapse.config.homeserver import HomeServerConfig
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine
ALTER_TABLE = 'ALTER TABLE remote_media_cache ADD COLUMN last_access_ts BIGINT'

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        while True:
            i = 10
    cur.execute(ALTER_TABLE)

def run_upgrade(cur: LoggingTransaction, database_engine: BaseDatabaseEngine, config: HomeServerConfig) -> None:
    if False:
        i = 10
        return i + 15
    cur.execute('UPDATE remote_media_cache SET last_access_ts = ?', (int(time.time() * 1000),))