"""
Script for backunfilling data from deleted users.

You might want to change `run_changed()` to `run_changed(use_safe_get=True)`
in `reddit-consumer-cloudsearch_q.conf` unless you're sure *everything* in
`LinksByAccount` is a valid `Link`. Otherwise, you're gonna back up the
cloudsearch queue.
"""
import time
import sys
from r2.lib.db.operators import desc
from r2.lib.utils import fetch_things2, progress
from r2.lib import amqp
from r2.models import Account

def get_queue_length(name):
    if False:
        i = 10
        return i + 15
    chan = amqp.connection_manager.get_channel()
    queue_response = chan.queue_declare(name, passive=True)
    return queue_response[1]

def backfill_deleted_accounts(resume_id=None):
    if False:
        while True:
            i = 10
    del_accts = Account._query(Account.c._deleted == True, sort=desc('_date'))
    if resume_id:
        del_accts._filter(Account.c._id < resume_id)
    for (i, account) in enumerate(progress(fetch_things2(del_accts))):
        if i % 1000 == 0:
            del_len = get_queue_length('del_account_q')
            cs_len = get_queue_length('cloudsearch_changes')
            while del_len > 1000 or cs_len > 10000:
                sys.stderr.write('CS: %d, DEL: %d' % (cs_len, del_len) + '\n')
                sys.stderr.flush()
                time.sleep(1)
                del_len = get_queue_length('del_account_q')
                cs_len = get_queue_length('cloudsearch_changes')
        amqp.add_item('account_deleted', account._fullname)