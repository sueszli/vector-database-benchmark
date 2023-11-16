from __future__ import annotations
import logging
import re
from time import time
from sentry.attachments import attachment_cache
from sentry.eventstore.processing import event_processing_store
from sentry.ingest.consumer.processors import CACHE_TIMEOUT
from sentry.tasks.store import preprocess_event, preprocess_event_from_reprocessing
from sentry.utils.canonical import CANONICAL_TYPES
_dist_re = re.compile('^[a-zA-Z0-9_.-]+$')
logger = logging.getLogger('sentry.api')

class APIError(Exception):
    http_status = 400
    msg = 'Invalid request'

    def __init__(self, msg: str | None=None) -> None:
        if False:
            print('Hello World!')
        if msg:
            self.msg = msg

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.msg or ''

class APIUnauthorized(APIError):
    http_status = 401
    msg = 'Unauthorized'

class APIForbidden(APIError):
    http_status = 403

def insert_data_to_database_legacy(data, start_time=None, from_reprocessing=False, attachments=None):
    if False:
        print('Hello World!')
    '\n    Yet another "fast path" to ingest an event without making it go\n    through Relay. Please consider using functions from the ingest consumer\n    instead, or, if you\'re within tests, to use `TestCase.store_event`.\n    '
    if start_time is None:
        start_time = time()
    if isinstance(data, CANONICAL_TYPES):
        data = dict(data.items())
    cache_key = event_processing_store.store(data)
    if attachments is not None:
        attachment_cache.set(cache_key, attachments, cache_timeout=CACHE_TIMEOUT)
    task = from_reprocessing and preprocess_event_from_reprocessing or preprocess_event
    task.delay(cache_key=cache_key, start_time=start_time, event_id=data['event_id'])