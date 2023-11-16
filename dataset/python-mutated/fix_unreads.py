import logging
import time
from typing import Callable, List, TypeVar
from django.db import connection
from django.db.backends.utils import CursorWrapper
from psycopg2.sql import SQL
from zerver.models import UserProfile
T = TypeVar('T')
'\nNOTE!  Be careful modifying this library, as it is used\nin a migration, and it needs to be valid for the state\nof the database that is in place when the 0104_fix_unreads\nmigration runs.\n'
logger = logging.getLogger('zulip.fix_unreads')
logger.setLevel(logging.WARNING)

def update_unread_flags(cursor: CursorWrapper, user_message_ids: List[int]) -> None:
    if False:
        for i in range(10):
            print('nop')
    query = SQL('\n        UPDATE zerver_usermessage\n        SET flags = flags | 1\n        WHERE id IN %(user_message_ids)s\n    ')
    cursor.execute(query, {'user_message_ids': tuple(user_message_ids)})

def get_timing(message: str, f: Callable[[], T]) -> T:
    if False:
        return 10
    start = time.time()
    logger.info(message)
    ret = f()
    elapsed = time.time() - start
    logger.info('elapsed time: %.03f\n', elapsed)
    return ret

def fix_unsubscribed(cursor: CursorWrapper, user_profile: UserProfile) -> None:
    if False:
        for i in range(10):
            print('nop')

    def find_recipients() -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        query = SQL('\n            SELECT\n                zerver_subscription.recipient_id\n            FROM\n                zerver_subscription\n            INNER JOIN zerver_recipient ON (\n                zerver_recipient.id = zerver_subscription.recipient_id\n            )\n            WHERE (\n                zerver_subscription.user_profile_id = %(user_profile_id)s AND\n                zerver_recipient.type = 2 AND\n                (NOT zerver_subscription.active)\n            )\n        ')
        cursor.execute(query, {'user_profile_id': user_profile.id})
        rows = cursor.fetchall()
        recipient_ids = [row[0] for row in rows]
        logger.info('%s', recipient_ids)
        return recipient_ids
    recipient_ids = get_timing('get recipients', find_recipients)
    if not recipient_ids:
        return

    def find() -> List[int]:
        if False:
            i = 10
            return i + 15
        query = SQL('\n            SELECT\n                zerver_usermessage.id\n            FROM\n                zerver_usermessage\n            INNER JOIN zerver_message ON (\n                zerver_message.id = zerver_usermessage.message_id\n            )\n            WHERE (\n                zerver_usermessage.user_profile_id = %(user_profile_id)s AND\n                (zerver_usermessage.flags & 1) = 0 AND\n                zerver_message.recipient_id in %(recipient_ids)s\n            )\n        ')
        cursor.execute(query, {'user_profile_id': user_profile.id, 'recipient_ids': tuple(recipient_ids)})
        rows = cursor.fetchall()
        user_message_ids = [row[0] for row in rows]
        logger.info('rows found: %d', len(user_message_ids))
        return user_message_ids
    user_message_ids = get_timing('finding unread messages for non-active streams', find)
    if not user_message_ids:
        return

    def fix() -> None:
        if False:
            print('Hello World!')
        update_unread_flags(cursor, user_message_ids)
    get_timing('fixing unread messages for non-active streams', fix)

def fix(user_profile: UserProfile) -> None:
    if False:
        while True:
            i = 10
    logger.info('\n---\nFixing %s:', user_profile.id)
    with connection.cursor() as cursor:
        fix_unsubscribed(cursor, user_profile)