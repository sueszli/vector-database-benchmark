import inspect
import time
from sentry.tsdb.base import BaseTSDB, TSDBModel
from sentry.tsdb.dummy import DummyTSDB
from sentry.tsdb.redis import RedisTSDB
from sentry.tsdb.snuba import SnubaTSDB
READ = 0
WRITE = 1

def single_model_argument(callargs):
    if False:
        for i in range(10):
            print('nop')
    return {callargs['model']}

def multiple_model_argument(callargs):
    if False:
        print('Hello World!')
    return set(callargs['models'])

def dont_do_this(callargs):
    if False:
        for i in range(10):
            print('nop')
    raise NotImplementedError('do not run this please')
method_specifications = {'get_range': (READ, single_model_argument), 'get_sums': (READ, single_model_argument), 'get_distinct_counts_series': (READ, single_model_argument), 'get_distinct_counts_totals': (READ, single_model_argument), 'get_distinct_counts_union': (READ, single_model_argument), 'get_most_frequent': (READ, single_model_argument), 'get_most_frequent_series': (READ, single_model_argument), 'get_frequency_series': (READ, single_model_argument), 'get_frequency_totals': (READ, single_model_argument), 'incr': (WRITE, single_model_argument), 'incr_multi': (WRITE, lambda callargs: {item[0] for item in callargs['items']}), 'merge': (WRITE, single_model_argument), 'delete': (WRITE, multiple_model_argument), 'record': (WRITE, single_model_argument), 'record_multi': (WRITE, lambda callargs: {model for (model, key, values) in callargs['items']}), 'merge_distinct_counts': (WRITE, single_model_argument), 'delete_distinct_counts': (WRITE, multiple_model_argument), 'record_frequency_multi': (WRITE, lambda callargs: {model for (model, data) in callargs['requests']}), 'merge_frequencies': (WRITE, single_model_argument), 'delete_frequencies': (WRITE, multiple_model_argument), 'flush': (WRITE, dont_do_this)}
assert set(method_specifications) == BaseTSDB.__read_methods__ | BaseTSDB.__write_methods__, 'all read and write methods must have a specification defined'
model_backends = {model: ('redis', 'redis') if model not in SnubaTSDB.model_query_settings else ('snuba', 'dummy') for model in TSDBModel}

def selector_func(method, callargs, switchover_timestamp=None):
    if False:
        for i in range(10):
            print('nop')
    spec = method_specifications.get(method)
    if spec is None:
        return 'redis'
    if switchover_timestamp is not None and time.time() < switchover_timestamp:
        return 'redis'
    (operation_type, model_extractor) = spec
    backends = {model_backends[model][operation_type] for model in model_extractor(callargs)}
    assert len(backends) == 1, 'request was not directed to a single backend'
    return backends.pop()

def make_method(key):
    if False:
        print('Hello World!')

    def method(self, *a, **kw):
        if False:
            while True:
                i = 10
        callargs = inspect.getcallargs(getattr(BaseTSDB, key), self, *a, **kw)
        backend = selector_func(key, callargs, self.switchover_timestamp)
        return getattr(self.backends[backend], key)(*a, **kw)
    return method

class RedisSnubaTSDBMeta(type):

    def __new__(cls, name, bases, attrs):
        if False:
            for i in range(10):
                print('nop')
        for key in method_specifications.keys():
            attrs[key] = make_method(key)
        return type.__new__(cls, name, bases, attrs)

class RedisSnubaTSDB(BaseTSDB, metaclass=RedisSnubaTSDBMeta):

    def __init__(self, switchover_timestamp=None, **options):
        if False:
            while True:
                i = 10
        '\n        A TSDB backend that uses the Snuba outcomes and events datasets as far\n        as possible instead of reading/writing to redis. Reading will trigger a\n        Snuba query, while writing is a noop as Snuba reads from outcomes.\n\n        Note: Using this backend requires you to start Snuba outcomes consumers\n        (not to be confused with the outcomes consumers in Sentry itself).\n\n        :param switchover_timestamp: When set, only start reading from snuba\n            after this timestamp (as returned by `time.time()`). When this\n            timestamp has not been reached yet, this backend just degrades to\n            Redis for *all* keys.\n\n            The default `None` will start reading from Snuba immediately and is\n            equivalent to setting a past timestamp.\n        '
        self.switchover_timestamp = switchover_timestamp
        self.backends = {'dummy': DummyTSDB(), 'redis': RedisTSDB(**options.pop('redis', {})), 'snuba': SnubaTSDB(**options.pop('snuba', {}))}
        super().__init__(**options)