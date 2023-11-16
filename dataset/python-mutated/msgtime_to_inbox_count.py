"""Converts msgtime for users to inbox_count, for inbox count tracking."""
import sys
from r2.lib.db import queries
from r2.lib.db.operators import desc
from r2.lib.utils import fetch_things2, progress
from r2.models import Account, Message
from pylons import app_globals as g

def _keep(msg, account):
    if False:
        for i in range(10):
            print('nop')
    "Adapted from listingcontroller.MessageController's keep_fn."
    if msg._deleted:
        return False
    if msg._spam and msg.author_id != account._id:
        return False
    if msg.author_id in account.enemies:
        return False
    if isinstance(msg, Message) and msg.to_id == account._id and msg.del_on_recipient:
        return False
    if msg.author_id == account._id:
        return False
    return True
resume_id = long(sys.argv[1]) if len(sys.argv) > 1 else None
msg_accounts = Account._query(sort=desc('_date'), data=True)
if resume_id:
    msg_accounts._filter(Account.c._id < resume_id)
for account in progress(fetch_things2(msg_accounts), estimate=resume_id):
    current_inbox_count = account.inbox_count
    unread_messages = list(queries.get_unread_inbox(account))
    if account._id % 100000 == 0:
        g.reset_caches()
    if not len(unread_messages):
        if current_inbox_count:
            account._incr('inbox_count', -current_inbox_count)
    else:
        msgs = Message._by_fullname(unread_messages, data=True, return_dict=False, ignore_missing=True)
        kept_msgs = sum((1 for msg in msgs if _keep(msg, account)))
        if kept_msgs or current_inbox_count:
            account._incr('inbox_count', kept_msgs - current_inbox_count)