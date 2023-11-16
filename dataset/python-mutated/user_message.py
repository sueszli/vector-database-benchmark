from typing import List
from django.db import connection
from psycopg2.extras import execute_values
from psycopg2.sql import SQL
from zerver.models import UserMessage

class UserMessageLite:
    """
    The Django ORM is too slow for bulk operations.  This class
    is optimized for the simple use case of inserting a bunch of
    rows into zerver_usermessage.
    """

    def __init__(self, user_profile_id: int, message_id: int, flags: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.user_profile_id = user_profile_id
        self.message_id = message_id
        self.flags = flags

    def flags_list(self) -> List[str]:
        if False:
            print('Hello World!')
        return UserMessage.flags_list_for_flags(self.flags)

def bulk_insert_ums(ums: List[UserMessageLite]) -> None:
    if False:
        i = 10
        return i + 15
    "\n    Doing bulk inserts this way is much faster than using Django,\n    since we don't have any ORM overhead.  Profiling with 1000\n    users shows a speedup of 0.436 -> 0.027 seconds, so we're\n    talking about a 15x speedup.\n    "
    if not ums:
        return
    vals = [(um.user_profile_id, um.message_id, um.flags) for um in ums]
    query = SQL('\n        INSERT into\n            zerver_usermessage (user_profile_id, message_id, flags)\n        VALUES %s\n    ')
    with connection.cursor() as cursor:
        execute_values(cursor.cursor, query, vals)