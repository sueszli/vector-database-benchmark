from r2.models import Subreddit
from r2.lib.memoize import memoize
from r2.lib.db.operators import desc
from r2.lib import utils
from r2.lib.db import tdb_cassandra
from r2.lib.cache import CL_ONE

class SubredditsByPartialName(tdb_cassandra.View):
    _use_db = True
    _value_type = 'pickle'
    _connection_pool = 'main'
    _read_consistency_level = CL_ONE

def load_all_reddits():
    if False:
        return 10
    query_cache = {}
    q = Subreddit._query(Subreddit.c.type == 'public', Subreddit.c._spam == False, Subreddit.c._downs > 1, sort=(desc('_downs'), desc('_ups')), data=True)
    for sr in utils.fetch_things2(q):
        if sr.quarantine:
            continue
        name = sr.name.lower()
        for i in xrange(len(name)):
            prefix = name[:i + 1]
            names = query_cache.setdefault(prefix, [])
            if len(names) < 10:
                names.append((sr.name, sr.over_18))
    for (name_prefix, subreddits) in query_cache.iteritems():
        SubredditsByPartialName._set_values(name_prefix, {'tups': subreddits})

def search_reddits(query, include_over_18=True):
    if False:
        i = 10
        return i + 15
    query = str(query.lower())
    try:
        result = SubredditsByPartialName._byID(query)
        return [name for (name, over_18) in getattr(result, 'tups', []) if not over_18 or include_over_18]
    except tdb_cassandra.NotFound:
        return []

@memoize('popular_searches', stale=True, time=3600)
def popular_searches(include_over_18=True):
    if False:
        while True:
            i = 10
    top_reddits = Subreddit._query(Subreddit.c.type == 'public', sort=desc('_downs'), limit=100, data=True)
    top_searches = {}
    for sr in top_reddits:
        if sr.quarantine:
            continue
        if sr.over_18 and (not include_over_18):
            continue
        name = sr.name.lower()
        for i in xrange(min(len(name), 3)):
            query = name[:i + 1]
            r = search_reddits(query, include_over_18)
            top_searches[query] = r
    return top_searches