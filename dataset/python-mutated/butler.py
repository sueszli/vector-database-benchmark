from pylons import tmpl_context as c
from pylons import app_globals as g
from r2.lib.db import queries
from r2.lib.db.tdb_sql import CreationError
from r2.lib import amqp
from r2.lib.utils import extract_user_mentions
from r2.models import query_cache, Thing, Comment, Account, Inbox, NotFound

def notify_mention(user, thing):
    if False:
        i = 10
        return i + 15
    try:
        inbox_rel = Inbox._add(user, thing, 'mention')
    except CreationError:
        g.log.error('duplicate mention for (%s, %s)', user, thing)
        return
    with query_cache.CachedQueryMutator() as m:
        m.insert(queries.get_inbox_comment_mentions(user), [inbox_rel])
        queries.set_unread(thing, user, unread=True, mutator=m)

def remove_mention_notification(mention):
    if False:
        for i in range(10):
            print('nop')
    inbox_owner = mention._thing1
    thing = mention._thing2
    with query_cache.CachedQueryMutator() as m:
        m.delete(queries.get_inbox_comment_mentions(inbox_owner), [mention])
        queries.set_unread(thing, inbox_owner, unread=False, mutator=m)

def readd_mention_notification(mention):
    if False:
        return 10
    'Reinsert into inbox after a comment has been unspammed'
    inbox_owner = mention._thing1
    thing = mention._thing2
    with query_cache.CachedQueryMutator() as m:
        m.insert(queries.get_inbox_comment_mentions(inbox_owner), [mention])
        unread = getattr(mention, 'unread_preremoval', True)
        queries.set_unread(thing, inbox_owner, unread=unread, mutator=m)

def monitor_mentions(comment):
    if False:
        i = 10
        return i + 15
    if comment._spam or comment._deleted:
        return
    sender = comment.author_slow
    if getattr(sender, 'butler_ignore', False):
        return
    if sender.in_timeout:
        return
    subreddit = comment.subreddit_slow
    usernames = extract_user_mentions(comment.body)
    inbox_class = Inbox.rel(Account, Comment)
    if len(usernames) > g.butler_max_mentions:
        return
    c.user_is_loggedin = True
    for username in usernames:
        try:
            account = Account._by_name(username)
        except NotFound:
            continue
        if account == sender:
            continue
        if not account.pref_monitor_mentions:
            continue
        if not subreddit.can_view(account):
            continue
        if account.is_enemy(sender):
            continue
        rels = inbox_class._fast_query(account, comment, ('inbox', 'selfreply', 'mention'))
        if filter(None, rels.values()):
            continue
        notify_mention(account, comment)

def run():
    if False:
        return 10

    @g.stats.amqp_processor('butler_q')
    def process_message(msg):
        if False:
            print('Hello World!')
        fname = msg.body
        item = Thing._by_fullname(fname, data=True)
        monitor_mentions(item)
    amqp.consume_items('butler_q', process_message, verbose=True)