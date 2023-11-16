"""Fill in the num_gildings for users

This is used to determine which gilding trophy level they should have.
"""
from pylons import app_globals as g
from r2.models import Account
from r2.models.gold import gold_table, ENGINE
from r2admin.lib.trophies import add_to_trophy_queue
from sqlalchemy.sql.expression import select
from sqlalchemy.sql.functions import count as sa_count

def update_num_gildings(update_trophy=True, user_id=None):
    if False:
        while True:
            i = 10
    'Returns total number of link, comment, and user gildings'
    query = select([gold_table.c.paying_id, sa_count(gold_table.c.trans_id)]).where(gold_table.c.trans_id.like('X%')).group_by(gold_table.c.paying_id).order_by(sa_count(gold_table.c.trans_id).desc())
    if user_id:
        query = query.where(gold_table.c.paying_id == str(user_id))
    rows = ENGINE.execute(query)
    total_updated = 0
    for (paying_id, count) in rows:
        try:
            a = Account._byID(int(paying_id), data=True)
            a.num_gildings = count
            a._commit()
            total_updated += 1
            if update_trophy and a.pref_public_server_seconds:
                add_to_trophy_queue(a, 'gilding')
        except:
            g.log.debug('update_num_gildings: paying_id %s is invalid' % paying_id)
    g.log.debug('update_num_gildings: updated %s accounts' % total_updated)