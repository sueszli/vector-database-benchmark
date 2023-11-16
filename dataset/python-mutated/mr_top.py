import sys
from r2.models import Link, Comment
from r2.lib.db.sorts import epoch_seconds, score, controversy
from r2.lib.db import queries
from r2.lib import mr_tools
from r2.lib.utils import timeago, UrlParser
from r2.lib.jsontemplates import make_fullname
thingcls_by_name = {'link': Link, 'comment': Comment}
data_fields_by_name = {'link': {'url': str, 'sr_id': int, 'author_id': int}, 'comment': {'sr_id': int, 'author_id': int}}

def join_things(thing_type):
    if False:
        while True:
            i = 10
    mr_tools.join_things(data_fields_by_name[thing_type].keys())

def _get_cutoffs(intervals):
    if False:
        i = 10
        return i + 15
    cutoffs = {}
    for interval in intervals:
        if interval == 'all':
            cutoffs['all'] = 0.0
        else:
            cutoffs[interval] = epoch_seconds(timeago('1 %s' % interval))
    return cutoffs

def time_listings(intervals, thing_type):
    if False:
        for i in range(10):
            print('nop')
    cutoff_by_interval = _get_cutoffs(intervals)

    @mr_tools.dataspec_m_thing(*data_fields_by_name[thing_type].items())
    def process(thing):
        if False:
            return 10
        if thing.deleted:
            return
        thing_cls = thingcls_by_name[thing.thing_type]
        fname = make_fullname(thing_cls, thing.thing_id)
        thing_score = score(thing.ups, thing.downs)
        thing_controversy = controversy(thing.ups, thing.downs)
        for (interval, cutoff) in cutoff_by_interval.iteritems():
            if thing.timestamp < cutoff:
                continue
            yield ('user/%s/top/%s/%d' % (thing.thing_type, interval, thing.author_id), thing_score, thing.timestamp, fname)
            yield ('user/%s/controversial/%s/%d' % (thing.thing_type, interval, thing.author_id), thing_controversy, thing.timestamp, fname)
            if thing.spam:
                continue
            if thing.thing_type == 'link':
                yield ('sr/link/top/%s/%d' % (interval, thing.sr_id), thing_score, thing.timestamp, fname)
                yield ('sr/link/controversial/%s/%d' % (interval, thing.sr_id), thing_controversy, thing.timestamp, fname)
                if thing.url:
                    try:
                        parsed = UrlParser(thing.url)
                    except ValueError:
                        continue
                    for domain in parsed.domain_permutations():
                        yield ('domain/link/top/%s/%s' % (interval, domain), thing_score, thing.timestamp, fname)
                        yield ('domain/link/controversial/%s/%s' % (interval, domain), thing_controversy, thing.timestamp, fname)
    mr_tools.mr_map(process)

def store_keys(key, maxes):
    if False:
        for i in range(10):
            print('nop')
    (category, thing_cls, sort, time, id) = key.split('/')
    query = None
    if category == 'user':
        if thing_cls == 'link':
            query = queries._get_submitted(int(id), sort, time)
        elif thing_cls == 'comment':
            query = queries._get_comments(int(id), sort, time)
    elif category == 'sr':
        if thing_cls == 'link':
            query = queries._get_links(int(id), sort, time)
    elif category == 'domain':
        if thing_cls == 'link':
            query = queries.get_domain_links(id, sort, time)
    assert query, 'unknown query type for %s' % (key,)
    item_tuples = [tuple([item[-1]] + [float(x) for x in item[:-1]]) for item in maxes]
    lock = time == 'all'
    query._replace(item_tuples, lock=lock)

def write_permacache(fd=sys.stdin):
    if False:
        while True:
            i = 10
    mr_tools.mr_reduce_max_per_key(lambda x: map(float, x[:-1]), num=1000, post=store_keys, fd=fd)

def reduce_listings(fd=sys.stdin):
    if False:
        for i in range(10):
            print('nop')
    mr_tools.mr_reduce_max_per_key(lambda x: map(float, x[:-1]), num=1000, fd=fd)