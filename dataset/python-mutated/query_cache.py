"""
This module provides a Cassandra-backed lockless query cache.  Rather than
doing complicated queries on the fly to populate listings, a list of items that
would be in that listing are maintained in Cassandra for fast lookup.  The
result can then be fed to IDBuilder to generate a final result.

Whenever an operation occurs that would modify the contents of the listing, the
listing should be updated somehow.  In some cases, this can be done by directly
mutating the listing and in others it must be done offline in batch processing
jobs.

"""
import json
import random
import datetime
import collections
from pylons import app_globals as g
from pycassa.system_manager import ASCII_TYPE, UTF8_TYPE
from pycassa.batch import Mutator
from r2.models import Thing
from r2.lib.db import tdb_cassandra
from r2.lib.db.operators import asc, desc, BooleanOp
from r2.lib.db.sorts import epoch_seconds
from r2.lib.utils import flatten, to36
CONNECTION_POOL = g.cassandra_pools['main']
PRUNE_CHANCE = g.querycache_prune_chance
MAX_CACHED_ITEMS = 1000
LOG = g.log

class ThingTupleComparator(object):
    """A callable usable for comparing sort-data in a cached query.

    The query cache stores minimal sort data on each thing to be able to order
    the items in a cached query.  This class provides the ordering for those
    thing tuples.

    """

    def __init__(self, sorts):
        if False:
            print('Hello World!')
        self.sorts = sorts

    def __call__(self, t1, t2):
        if False:
            i = 10
            return i + 15
        for (i, s) in enumerate(self.sorts):
            (v1, v2) = (t1[i + 1], t2[i + 1])
            if v1 != v2:
                return cmp(v1, v2) if isinstance(s, asc) else cmp(v2, v1)
        return 0

class _CachedQueryBase(object):

    def __init__(self, sort):
        if False:
            while True:
                i = 10
        self.sort = sort
        self.sort_cols = [s.col for s in self.sort]
        self.data = []
        self._fetched = False

    def fetch(self, force=False):
        if False:
            while True:
                i = 10
        "Fill the cached query's sorted item list from Cassandra.\n\n        If the query has already been fetched, this method is a no-op unless\n        force=True.\n\n        "
        if not force and self._fetched:
            return
        self._fetch()
        self._sort_data()
        self._fetched = True

    def _fetch(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def _sort_data(self):
        if False:
            while True:
                i = 10
        comparator = ThingTupleComparator(self.sort_cols)
        self.data.sort(cmp=comparator)

    def __iter__(self):
        if False:
            return 10
        self.fetch()
        for x in self.data[:MAX_CACHED_ITEMS]:
            yield x[0]

class CachedQuery(_CachedQueryBase):
    """A materialized view of a complex query.

    Complicated queries can take way too long to sort in the databases.  This
    class provides a fast-access view of a given listing's items.  The cache
    stores each item's ID and a minimal subset of its data as required for
    sorting.

    Each time the listing is fetched, it is sorted. Because of this, we need to
    ensure the listing does not grow too large.  On each insert, a "pruning"
    can occur (with a configurable probability) which will remove excess items
    from the end of the listing.

    Use CachedQueryMutator to make changes to the cached query's item list.

    """

    def __init__(self, model, key, sort, filter_fn, is_precomputed):
        if False:
            i = 10
            return i + 15
        self.model = model
        self.key = key
        self.filter = filter_fn
        self.timestamps = None
        self.is_precomputed = is_precomputed
        super(CachedQuery, self).__init__(sort)

    def _make_item_tuple(self, item):
        if False:
            i = 10
            return i + 15
        'Return an item tuple from the result of a query.\n\n        The item tuple is used to sort the items in a query without having to\n        look them up.\n\n        '
        filtered_item = self.filter(item)
        lst = [filtered_item._fullname]
        for col in self.sort_cols:
            attr = getattr(item, col)
            if isinstance(attr, datetime.datetime):
                attr = epoch_seconds(attr)
            lst.append(attr)
        return tuple(lst)

    def _fetch(self):
        if False:
            print('Hello World!')
        self._fetch_multi([self])

    @classmethod
    def _fetch_multi(self, queries):
        if False:
            for i in range(10):
                print('nop')
        'Fetch the unsorted query results for multiple queries at once.\n\n        In the case of precomputed queries, do an extra lookup first to\n        determine which row key to find the latest precomputed values for the\n        query in.\n\n        '
        by_model = collections.defaultdict(list)
        for q in queries:
            by_model[q.model].append(q)
        cached_queries = {}
        for (model, queries) in by_model.iteritems():
            (pure, need_mangling) = ([], [])
            for q in queries:
                if not q.is_precomputed:
                    pure.append(q.key)
                else:
                    need_mangling.append(q.key)
            mangled = model.index_mangle_keys(need_mangling)
            fetched = model.get(pure + mangled.keys())
            for (key, values) in fetched.iteritems():
                key = mangled.get(key, key)
                cached_queries[key] = values
        for q in queries:
            cached_query = cached_queries.get(q.key)
            if cached_query:
                (q.data, q.timestamps) = cached_query

    def _cols_from_things(self, things):
        if False:
            i = 10
            return i + 15
        cols = {}
        for thing in things:
            t = self._make_item_tuple(thing)
            cols[t[0]] = tuple(t[1:])
        return cols

    def _insert(self, mutator, things):
        if False:
            print('Hello World!')
        if not things:
            return
        cols = self._cols_from_things(things)
        self.model.insert(mutator, self.key, cols)

    def _replace(self, mutator, things, ttl):
        if False:
            while True:
                i = 10
        cols = self._cols_from_things(things)
        self.model.replace(mutator, self.key, cols, ttl)

    def _delete(self, mutator, things):
        if False:
            while True:
                i = 10
        if not things:
            return
        fullnames = [self.filter(x)._fullname for x in things]
        self.model.remove(mutator, self.key, fullnames)

    def _prune(self, mutator):
        if False:
            print('Hello World!')
        to_keep = [t[0] for t in self.data[:MAX_CACHED_ITEMS]]
        to_prune = [t[0] for t in self.data[MAX_CACHED_ITEMS:]]
        if to_prune:
            oldest_keep = min((self.timestamps[_id] for _id in to_keep))
            fast_prunable = [_id for _id in to_prune if self.timestamps[_id] < oldest_keep]
            num_to_prune = len(to_prune)
            num_fast_prunable = len(fast_prunable)
            num_unpruned_if_fast = num_to_prune - num_fast_prunable
            if num_fast_prunable > num_to_prune * 0.5 and num_unpruned_if_fast < MAX_CACHED_ITEMS * 0.5:
                newest_prune = max((self.timestamps[_id] for _id in fast_prunable))
                self.model.remove_older_than(mutator, self.key, newest_prune)
                event_name = 'fast_pruned'
                num_pruned = num_fast_prunable
            else:
                prune_size = int(1.5 * 1 / PRUNE_CHANCE)
                to_prune = to_prune[-prune_size:]
                self.model.remove_if_unchanged(mutator, self.key, to_prune, self.timestamps)
                event_name = 'pruned'
                num_pruned = len(to_prune)
            cf_name = self.model.__name__
            query_name = self.key.split('.')[0]
            counter_key = 'cache.%s.%s' % (cf_name, query_name)
            counter = g.stats.get_counter(counter_key)
            if counter:
                counter.increment(event_name, delta=num_pruned)

    @classmethod
    def _prune_multi(cls, queries):
        if False:
            return 10
        cls._fetch_multi(queries)
        with Mutator(CONNECTION_POOL) as m:
            for q in queries:
                q._sort_data()
                q._prune(m)

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(self.key)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.key == other.key

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        return not self.__eq__(other)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '%s(%s, %r)' % (self.__class__.__name__, self.model.__name__, self.key)

class MergedCachedQuery(_CachedQueryBase):
    """A cached query built by merging multiple sub-queries.

    Merged queries can be read, but cannot be modified as it is not easy to
    determine from a given item which sub-query should get modified.

    """

    def __init__(self, queries):
        if False:
            for i in range(10):
                print('nop')
        self.queries = queries
        if queries:
            sort = queries[0].sort
            assert all((sort == q.sort for q in queries))
        else:
            sort = []
        super(MergedCachedQuery, self).__init__(sort)

    def _fetch(self):
        if False:
            return 10
        CachedQuery._fetch_multi(self.queries)
        self.data = flatten([q.data for q in self.queries])

class CachedQueryMutator(object):
    """Utility to manipulate cached queries with batching.

    This implements the context manager protocol so it can be used with the
    with statement for clean batches.

    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.mutator = Mutator(CONNECTION_POOL)
        self.to_prune = set()

    def __enter__(self):
        if False:
            print('Hello World!')
        return self

    def __exit__(self, type, value, traceback):
        if False:
            print('Hello World!')
        self.send()

    def insert(self, query, things):
        if False:
            print('Hello World!')
        'Insert items into the given cached query.\n\n        If the items are already in the query, they will have their sorts\n        updated.\n\n        This will sometimes trigger pruning with a configurable probability\n        (see g.querycache_prune_chance).\n\n        '
        if not things:
            return
        LOG.debug('Inserting %r into query %r', things, query)
        assert not query.is_precomputed
        query._insert(self.mutator, things)
        if random.random() / len(things) < PRUNE_CHANCE:
            self.to_prune.add(query)

    def replace(self, query, things, ttl=None):
        if False:
            for i in range(10):
                print('nop')
        "Replace a precomputed query with a new set of things.\n\n        The query index will be updated. If a TTL is specified, it will be\n        applied to all columns generated by this action allowing old\n        precomputed queries to fall away after they're no longer useful.\n\n        "
        assert query.is_precomputed
        if isinstance(ttl, datetime.timedelta):
            ttl = ttl.total_seconds()
        query._replace(self.mutator, things, ttl)

    def delete(self, query, things):
        if False:
            while True:
                i = 10
        'Remove things from the query.'
        if not things:
            return
        LOG.debug('Deleting %r from query %r', things, query)
        query._delete(self.mutator, things)

    def send(self):
        if False:
            while True:
                i = 10
        'Commit the mutations batched up so far and potentially do pruning.\n\n        This is automatically called by __exit__ when used as a context\n        manager.\n\n        '
        self.mutator.send()
        if self.to_prune:
            LOG.debug('Pruning queries %r', self.to_prune)
            CachedQuery._prune_multi(self.to_prune)

def filter_identity(x):
    if False:
        for i in range(10):
            print('nop')
    'Return the same thing given.\n\n    Use this as the filter_fn of simple Thing-based cached queries so that\n    the enumerated things will be returned for rendering.\n\n    '
    return x

def filter_thing2(x):
    if False:
        return 10
    'Return the thing2 of a given relationship.\n\n    Use this as the filter_fn of a cached Relation query so that the related\n    things will be returned for rendering.\n\n    '
    return x._thing2

def filter_thing(x):
    if False:
        i = 10
        return i + 15
    'Return "thing" from a proxy object.\n\n    Use this as the filter_fn when some object that\'s not a Thing or Relation\n    is used as the basis of a cached query.\n\n    '
    return x.thing

def _is_query_precomputed(query):
    if False:
        for i in range(10):
            print('nop')
    'Return if this query must be updated offline in a batch job.\n\n    Simple queries can be modified in place in the query cache, but ones\n    with more complicated eligibility criteria, such as a time limit ("top\n    this month") cannot be modified this way and must instead be\n    recalculated periodically.  Rather than replacing a single row\n    repeatedly, the precomputer stores in a new row every time it runs and\n    updates an index of the latest run.\n\n    '
    rules = list(query._rules)
    while rules:
        rule = rules.pop()
        if isinstance(rule, BooleanOp):
            rules.extend(rule.ops)
            continue
        if rule.lval.name == '_date':
            return True
    return False

class FakeQuery(object):
    """A somewhat query-like object for conveying sort information."""

    def __init__(self, sort, precomputed=False):
        if False:
            print('Hello World!')
        self._sort = sort
        self.precomputed = precomputed

def cached_query(model, filter_fn=filter_identity):
    if False:
        while True:
            i = 10
    'Decorate a function describing a cached query.\n\n    The decorated function is expected to follow the naming convention common\n    in queries.py -- "get_something".  The cached query\'s key will be generated\n    from the combination of the function name and its arguments separated by\n    periods.\n\n    The decorated function should return a raw thingdb query object\n    representing the query that is being cached. If there is no valid\n    underlying query to build off of, a FakeQuery specifying the correct\n    sorting criteria for the enumerated objects can be returned.\n\n    '

    def cached_query_decorator(fn):
        if False:
            print('Hello World!')

        def cached_query_wrapper(*args):
            if False:
                while True:
                    i = 10
            assert fn.__name__.startswith('get_')
            row_key_components = [fn.__name__[len('get_'):]]
            if len(args) > 0:
                if isinstance(args[0], Thing):
                    args = list(args)
                    args[0] = args[0]._id
                if isinstance(args[0], (int, long)):
                    serialized = to36(args[0])
                else:
                    serialized = str(args[0])
                row_key_components.append(serialized)
            row_key_components.extend((str(x) for x in args[1:]))
            row_key = '.'.join(row_key_components)
            query = fn(*args)
            query_sort = query._sort
            try:
                is_precomputed = query.precomputed
            except AttributeError:
                is_precomputed = _is_query_precomputed(query)
            return CachedQuery(model, row_key, query_sort, filter_fn, is_precomputed)
        return cached_query_wrapper
    return cached_query_decorator

def merged_cached_query(fn):
    if False:
        print('Hello World!')
    'Decorate a function describing a cached query made up of others.\n\n    The decorated function should return a sequence of cached queries whose\n    results will be merged together into a final listing.\n\n    '

    def merge_wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        queries = fn(*args, **kwargs)
        return MergedCachedQuery(queries)
    return merge_wrapper

class _BaseQueryCache(object):
    """The model through which cached queries to interact with Cassandra.

    Each cached query is stored as a distinct row in Cassandra.  The row key is
    given by higher level code (see the cached_query decorator above).  Each
    item in the materialized result of the query is stored as a separate
    column.  Each column name is the fullname of the item, while each value is
    the stuff CachedQuery needs to be able to sort the items (see
    CachedQuery._make_item_tuple).

    """
    __metaclass__ = tdb_cassandra.ThingMeta
    _connection_pool = 'main'
    _extra_schema_creation_args = dict(key_validation_class=ASCII_TYPE, default_validation_class=UTF8_TYPE)
    _compare_with = ASCII_TYPE
    _use_db = False
    _type_prefix = None
    _cf_name = None

    @classmethod
    def get(cls, keys):
        if False:
            for i in range(10):
                print('nop')
        'Retrieve the items in a set of cached queries.\n\n        For each cached query, this returns the thing tuples and the column\n        timestamps for them.  The latter is useful for conditional removal\n        during pruning.\n\n        '
        rows = cls._cf.multiget(keys, include_timestamp=True, column_count=tdb_cassandra.max_column_count)
        res = {}
        for (row, columns) in rows.iteritems():
            data = []
            timestamps = []
            for (key, (value, timestamp)) in columns.iteritems():
                value = json.loads(value)
                data.append((key,) + tuple(value))
                timestamps.append((key, timestamp))
            res[row] = (data, dict(timestamps))
        return res

    @classmethod
    def index_mangle_keys(cls, keys):
        if False:
            while True:
                i = 10
        if not keys:
            return {}
        index_keys = ['/'.join((key, 'index')) for key in keys]
        rows = cls._cf.multiget(index_keys, column_reversed=True, column_count=1)
        res = {}
        for (key, columns) in rows.iteritems():
            root_key = key.rsplit('/')[0]
            index_component = columns.keys()[0]
            mangled = '/'.join((root_key, index_component))
            res[mangled] = root_key
        return res

    @classmethod
    @tdb_cassandra.will_write
    def insert(cls, mutator, key, columns, ttl=None):
        if False:
            while True:
                i = 10
        'Insert things into the cached query.\n\n        This works as an upsert; if the thing already exists, it is updated. If\n        not, it is actually inserted.\n\n        '
        updates = dict(((key, json.dumps(value)) for (key, value) in columns.iteritems()))
        mutator.insert(cls._cf, key, updates, ttl=ttl)

    @classmethod
    @tdb_cassandra.will_write
    def replace(cls, mutator, key, columns, ttl):
        if False:
            i = 10
            return i + 15
        job_key = datetime.datetime.now(g.tz).isoformat()
        cls.insert(mutator, key + '/' + job_key, columns, ttl=ttl)
        mutator.insert(cls._cf, key + '/index', {job_key: ''}, ttl=ttl)

    @classmethod
    @tdb_cassandra.will_write
    def remove(cls, mutator, key, columns):
        if False:
            print('Hello World!')
        'Unconditionally remove things from the cached query.'
        mutator.remove(cls._cf, key, columns=columns)

    @classmethod
    @tdb_cassandra.will_write
    def remove_if_unchanged(cls, mutator, key, columns, timestamps):
        if False:
            i = 10
            return i + 15
        'Remove things from the cached query if unchanged.\n\n        If the things have been changed since the specified timestamps, they\n        will not be removed.  This is useful for avoiding race conditions while\n        pruning.\n\n        '
        for col in columns:
            mutator.remove(cls._cf, key, columns=[col], timestamp=timestamps.get(col))

    @classmethod
    @tdb_cassandra.will_write
    def remove_older_than(cls, mutator, key, removal_timestamp):
        if False:
            return 10
        "Remove things older than the specified timestamp.\n\n        Removing specific columns can cause tombstones to build up. When a row\n        has tons of tombstones fetching that row gets slow because Cassandra\n        must retrieve all the tombstones as well. Issuing a row remove with\n        the timestamp specified clears out all the columns modified before\n        that timestamp and somehow doesn't result in tombstones being left\n        behind. This behavior was verified via request tracing.\n\n        "
        mutator.remove(cls._cf, key, timestamp=removal_timestamp)

class UserQueryCache(_BaseQueryCache):
    """A query cache column family for user-keyed queries."""
    _use_db = True

class SubredditQueryCache(_BaseQueryCache):
    """A query cache column family for subreddit-keyed queries."""
    _use_db = True