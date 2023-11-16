"""
This migration denormalises the account_data table into an ignored users table.
"""
import logging
from io import StringIO
from synapse.storage._base import db_to_json
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine
from synapse.storage.prepare_database import execute_statements_from_stream
logger = logging.getLogger(__name__)

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        i = 10
        return i + 15
    logger.info('Creating ignored_users table')
    execute_statements_from_stream(cur, StringIO(_create_commands))
    insert_sql = '\n    INSERT INTO ignored_users (ignorer_user_id, ignored_user_id) VALUES (?, ?)\n    '
    logger.info('Converting existing ignore lists')
    cur.execute("SELECT user_id, content FROM account_data WHERE account_data_type = 'm.ignored_user_list'")
    for (user_id, content_json) in cur.fetchall():
        content = db_to_json(content_json)
        ignored_users = content.get('ignored_users', {})
        if isinstance(ignored_users, dict) and ignored_users:
            cur.execute_batch(insert_sql, [(user_id, u) for u in ignored_users])
    logger.info('Adding constraints to ignored_users table')
    execute_statements_from_stream(cur, StringIO(_constraints_commands))
_create_commands = '\n-- Users which are ignored when calculating push notifications. This data is\n-- denormalized from account data.\nCREATE TABLE IF NOT EXISTS ignored_users(\n    ignorer_user_id TEXT NOT NULL,  -- The user ID of the user who is ignoring another user. (This is a local user.)\n    ignored_user_id TEXT NOT NULL  -- The user ID of the user who is being ignored. (This is a local or remote user.)\n);\n'
_constraints_commands = '\nCREATE UNIQUE INDEX ignored_users_uniqueness ON ignored_users (ignorer_user_id, ignored_user_id);\n\n-- Add an index on ignored_users since look-ups are done to get all ignorers of an ignored user.\nCREATE INDEX ignored_users_ignored_user_id ON ignored_users (ignored_user_id);\n'