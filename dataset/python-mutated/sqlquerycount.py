import logging
import threading
from collections import defaultdict
from sentry.debug.utils.patch_context import PatchContext
DEFAULT_MAX_QUERIES = 25
DEFAULT_MAX_DUPES = 3

class State(threading.local):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.count = 0
        self.query_hashes = defaultdict(int)

    def record_query(self, sql):
        if False:
            return 10
        self.count += 1
        self.query_hashes[hash(sql)] += 1

    def count_dupes(self):
        if False:
            i = 10
            return i + 15
        return sum((1 for n in self.query_hashes.values() if n > 1))

class CursorWrapper:

    def __init__(self, cursor, connection, state):
        if False:
            while True:
                i = 10
        self.cursor = cursor
        self.connection = connection
        self._state = state

    def execute(self, sql, params=()):
        if False:
            return 10
        try:
            return self.cursor.execute(sql, params)
        finally:
            self._state.record_query(sql)

    def executemany(self, sql, paramlist):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self.cursor.executemany(sql, paramlist)
        finally:
            self._state.record_query(sql)

    def __getattr__(self, attr):
        if False:
            for i in range(10):
                print('nop')
        if attr in self.__dict__:
            return self.__dict__[attr]
        else:
            return getattr(self.cursor, attr)

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self.cursor)

def get_cursor_wrapper(state):
    if False:
        return 10

    def cursor(func, self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        result = func(self, *args, **kwargs)
        return CursorWrapper(result, self, state)
    return cursor

class SqlQueryCountMonitor:

    def __init__(self, context, max_queries=DEFAULT_MAX_QUERIES, max_dupes=DEFAULT_MAX_DUPES, logger=None, **kwargs):
        if False:
            while True:
                i = 10
        self.context = context
        self.max_queries = max_queries
        self.max_dupes = max_dupes
        self.logger = logger or logging.getLogger(__name__)
        self.state = State()
        self._cursor = get_cursor_wrapper(self.state)
        self._patcher = PatchContext('django.db.backends.BaseDatabaseWrapper.cursor', self._cursor)

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.stop()

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        self._patcher.patch()

    def stop(self):
        if False:
            while True:
                i = 10
        self._patcher.unpatch()
        num_dupes = self.state.count_dupes()
        if self.state.count > self.max_queries:
            self.log_max_queries(num_dupes)
        if num_dupes > self.max_dupes:
            self.log_max_dupes(num_dupes)

    def log_max_dupes(self, num_dupes):
        if False:
            print('Hello World!')
        state = self.state
        context = {'stack': True, 'data': {'query_count': state.count, 'num_dupes': num_dupes}}
        self.logger.warning('%d duplicate queries executed in %s', num_dupes, self.context, extra=context)

    def log_max_queries(self, num_dupes):
        if False:
            while True:
                i = 10
        state = self.state
        context = {'stack': True, 'data': {'query_count': state.count, 'num_dupes': num_dupes}}
        self.logger.warning('%d queries executed in %s', state.count, self.context, extra=context)