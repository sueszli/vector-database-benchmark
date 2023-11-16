from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine
from synapse.storage.prepare_database import get_statements
FIX_INDEXES = '\n-- rebuild indexes as uniques\nDROP INDEX groups_invites_g_idx;\nCREATE UNIQUE INDEX group_invites_g_idx ON group_invites(group_id, user_id);\nDROP INDEX groups_users_g_idx;\nCREATE UNIQUE INDEX group_users_g_idx ON group_users(group_id, user_id);\n\n-- rename other indexes to actually match their table names..\nDROP INDEX groups_users_u_idx;\nCREATE INDEX group_users_u_idx ON group_users(user_id);\nDROP INDEX groups_invites_u_idx;\nCREATE INDEX group_invites_u_idx ON group_invites(user_id);\nDROP INDEX groups_rooms_g_idx;\nCREATE UNIQUE INDEX group_rooms_g_idx ON group_rooms(group_id, room_id);\nDROP INDEX groups_rooms_r_idx;\nCREATE INDEX group_rooms_r_idx ON group_rooms(room_id);\n'

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        i = 10
        return i + 15
    rowid = database_engine.row_id_name
    cur.execute('\n        DELETE FROM group_users WHERE %s NOT IN (\n           SELECT min(%s) FROM group_users GROUP BY group_id, user_id\n        );\n    ' % (rowid, rowid))
    cur.execute('\n        DELETE FROM group_invites WHERE %s NOT IN (\n           SELECT min(%s) FROM group_invites GROUP BY group_id, user_id\n        );\n    ' % (rowid, rowid))
    for statement in get_statements(FIX_INDEXES.splitlines()):
        cur.execute(statement)