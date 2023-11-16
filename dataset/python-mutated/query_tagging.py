import threading
from typing import Any, Optional
thread_local_storage = threading.local()

def get_query_tags():
    if False:
        while True:
            i = 10
    try:
        return thread_local_storage.query_tags
    except AttributeError:
        return {}

def get_query_tag_value(key: str) -> Optional[Any]:
    if False:
        while True:
            i = 10
    try:
        return thread_local_storage.query_tags[key]
    except (AttributeError, KeyError):
        return None

def tag_queries(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    tags = {key: value for (key, value) in kwargs.items() if value is not None}
    try:
        thread_local_storage.query_tags.update(tags)
    except AttributeError:
        thread_local_storage.query_tags = tags

def reset_query_tags():
    if False:
        for i in range(10):
            print('nop')
    thread_local_storage.query_tags = {}

class QueryCounter:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.total_query_time = 0.0

    @property
    def query_time_ms(self):
        if False:
            for i in range(10):
                print('nop')
        return self.total_query_time * 1000

    def __call__(self, execute, *args, **kwargs):
        if False:
            print('Hello World!')
        import time
        start_time = time.perf_counter()
        try:
            return execute(*args, **kwargs)
        finally:
            self.total_query_time += time.perf_counter() - start_time