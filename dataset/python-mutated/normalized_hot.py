import heapq
import itertools
from datetime import datetime, timedelta
from pylons import app_globals as g
from r2.config import feature
from r2.lib.db.queries import _get_links, CachedResults
from r2.lib.db.sorts import epoch_seconds
MAX_PER_SUBREDDIT = 150
MAX_LINKS = 1000

def get_hot_tuples(sr_ids, ageweight=None):
    if False:
        while True:
            i = 10
    queries_by_sr_id = {sr_id: _get_links(sr_id, sort='hot', time='all') for sr_id in sr_ids}
    CachedResults.fetch_multi(queries_by_sr_id.values(), stale=True)
    tuples_by_srid = {sr_id: [] for sr_id in sr_ids}
    now_seconds = epoch_seconds(datetime.now(g.tz))
    for (sr_id, q) in queries_by_sr_id.iteritems():
        if not q.data:
            continue
        hot_factor = get_hot_factor(q.data[0], now_seconds, ageweight)
        for (link_name, hot, timestamp) in q.data[:MAX_PER_SUBREDDIT]:
            effective_hot = hot / hot_factor
            tuples_by_srid[sr_id].append((-effective_hot, -hot, link_name, timestamp))
    return tuples_by_srid

def get_hot_factor(qdata, now, ageweight):
    if False:
        i = 10
        return i + 15
    'Return a "hot factor" score for a link\'s hot tuple.\n\n    Recalculate the item\'s hot score as if it had been submitted\n    more recently than it was. This will cause the `effective_hot` value in\n    get_hot_tuples to move older first items back\n\n    ageweight should be a float from 0.0 - 1.0, which "scales" how far\n    between the original submission time and "now" to use as the base\n    for the new hot score. Smaller values will favor older #1 posts in\n    multireddits; larger values will drop older posts further in the ranking\n    (or possibly off the ranking entirely).\n\n    '
    ageweight = float(ageweight or 0.0)
    (link_name, hot, timestamp) = qdata
    return max(hot + (now - timestamp) * ageweight / 45000.0, 1.0)

def normalized_hot(sr_ids, obey_age_limit=True, ageweight=None):
    if False:
        while True:
            i = 10
    timer = g.stats.get_timer('normalized_hot')
    timer.start()
    if not sr_ids:
        return []
    if not feature.is_enabled('scaled_normalized_hot'):
        ageweight = None
    tuples_by_srid = get_hot_tuples(sr_ids, ageweight=ageweight)
    if obey_age_limit:
        cutoff = datetime.now(g.tz) - timedelta(days=g.HOT_PAGE_AGE)
        oldest = epoch_seconds(cutoff)
    else:
        oldest = 0.0
    merged = heapq.merge(*tuples_by_srid.values())
    generator = (link_name for (ehot, hot, link_name, timestamp) in merged if timestamp > oldest)
    ret = list(itertools.islice(generator, MAX_LINKS))
    timer.stop()
    return ret