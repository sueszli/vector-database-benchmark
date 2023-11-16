from __future__ import annotations
import dataclasses
import logging
import random
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Callable
from urllib.parse import urljoin
import sentry_sdk
from django.conf import settings
from requests.exceptions import RequestException
from sentry import options
from sentry.lang.native.sources import get_bundle_index_urls, get_internal_artifact_lookup_source, get_scraping_config, sources_for_symbolication
from sentry.models.project import Project
from sentry.net.http import Session
from sentry.utils import json, metrics
MAX_ATTEMPTS = 3
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SymbolicatorTaskKind:
    is_js: bool = False
    is_low_priority: bool = False
    is_reprocessing: bool = False

    def with_low_priority(self, is_low_priority: bool) -> SymbolicatorTaskKind:
        if False:
            print('Hello World!')
        return dataclasses.replace(self, is_low_priority=is_low_priority)

    def with_js(self, is_js: bool) -> SymbolicatorTaskKind:
        if False:
            return 10
        return dataclasses.replace(self, is_js=is_js)

class SymbolicatorPools(Enum):
    default = 'default'
    js = 'js'
    lpq = 'lpq'
    lpq_js = 'lpq_js'

class Symbolicator:

    def __init__(self, task_kind: SymbolicatorTaskKind, on_request: Callable[[], None], project: Project, event_id: str):
        if False:
            for i in range(10):
                print('nop')
        URLS = settings.SYMBOLICATOR_POOL_URLS
        pool = SymbolicatorPools.default.value
        if task_kind.is_low_priority:
            if task_kind.is_js:
                pool = SymbolicatorPools.lpq_js.value
            else:
                pool = SymbolicatorPools.lpq.value
        elif task_kind.is_js:
            pool = SymbolicatorPools.js.value
        base_url = URLS.get(pool) or URLS.get(SymbolicatorPools.default.value) or options.get('symbolicator.options')['url']
        base_url = base_url.rstrip('/')
        assert base_url
        self.base_url = base_url
        self.on_request = on_request
        self.project = project
        self.event_id = event_id

    def _process(self, task_name: str, path: str, **kwargs):
        if False:
            while True:
                i = 10
        '\n        This function will submit a symbolication task to a Symbolicator and handle\n        polling it using the `SymbolicatorSession`.\n        It will also correctly handle `TaskIdNotFound` and `ServiceUnavailable` errors.\n        '
        session = SymbolicatorSession(url=self.base_url, project_id=str(self.project.id), event_id=str(self.event_id), timeout=settings.SYMBOLICATOR_POLL_TIMEOUT)
        task_id: str | None = None
        json_response = None
        with session:
            while True:
                try:
                    if not task_id:
                        json_response = session.create_task(path, **kwargs)
                    else:
                        json_response = session.query_task(task_id)
                except TaskIdNotFound:
                    task_id = None
                    continue
                except ServiceUnavailable:
                    session.reset_worker_id()
                    task_id = None
                    continue
                finally:
                    self.on_request()
                metrics.incr('events.symbolicator.response', tags={'response': json_response.get('status') or 'null', 'task_name': task_name})
                if json_response['status'] == 'pending':
                    task_id = json_response['request_id']
                    continue
                return json_response

    def process_minidump(self, minidump):
        if False:
            print('Hello World!')
        (sources, process_response) = sources_for_symbolication(self.project)
        scraping_config = get_scraping_config(self.project)
        data = {'sources': json.dumps(sources), 'scraping': json.dumps(scraping_config), 'options': '{"dif_candidates": true}'}
        res = self._process('process_minidump', 'minidump', data=data, files={'upload_file_minidump': minidump})
        return process_response(res)

    def process_applecrashreport(self, report):
        if False:
            for i in range(10):
                print('nop')
        (sources, process_response) = sources_for_symbolication(self.project)
        scraping_config = get_scraping_config(self.project)
        data = {'sources': json.dumps(sources), 'scraping': json.dumps(scraping_config), 'options': '{"dif_candidates": true}'}
        res = self._process('process_applecrashreport', 'applecrashreport', data=data, files={'apple_crash_report': report})
        return process_response(res)

    def process_payload(self, stacktraces, modules, signal=None, apply_source_context=True):
        if False:
            for i in range(10):
                print('nop')
        (sources, process_response) = sources_for_symbolication(self.project)
        scraping_config = get_scraping_config(self.project)
        json = {'sources': sources, 'options': {'dif_candidates': True, 'apply_source_context': apply_source_context}, 'stacktraces': stacktraces, 'modules': modules, 'scraping': scraping_config}
        if signal:
            json['signal'] = signal
        res = self._process('symbolicate_stacktraces', 'symbolicate', json=json)
        return process_response(res)

    def process_js(self, stacktraces, modules, release, dist, apply_source_context=True):
        if False:
            for i in range(10):
                print('nop')
        source = get_internal_artifact_lookup_source(self.project)
        scraping_config = get_scraping_config(self.project)
        json = {'source': source, 'stacktraces': stacktraces, 'modules': modules, 'options': {'apply_source_context': apply_source_context}, 'scraping': scraping_config}
        try:
            (debug_id_index, url_index) = get_bundle_index_urls(self.project, release, dist)
            if debug_id_index:
                json['debug_id_index'] = debug_id_index
            if url_index:
                json['url_index'] = url_index
        except Exception as e:
            sentry_sdk.capture_exception(e)
        if release is not None:
            json['release'] = release
        if dist is not None:
            json['dist'] = dist
        return self._process('symbolicate_js_stacktraces', 'symbolicate-js', json=json)

class TaskIdNotFound(Exception):
    pass

class ServiceUnavailable(Exception):
    pass

class SymbolicatorSession:
    """
    The `SymbolicatorSession` is a glorified HTTP request wrapper that does the following things:

    - Maintains a `worker_id` which is used downstream in the load balancer for routing.
    - Maintains `timeout` parameters which are passed to Symbolicator.
    - Converts 404 and 503 errors into proper classes so they can be handled upstream.
    - Otherwise, it retries failed requests.
    """
    _worker_id = None

    def __init__(self, url=None, project_id=None, event_id=None, timeout=None):
        if False:
            while True:
                i = 10
        self.url = url
        self.project_id = project_id
        self.event_id = event_id
        self.timeout = timeout
        self.session = None
        self.worker_id = self._get_worker_id()

    def __enter__(self):
        if False:
            return 10
        self.open()
        return self

    def __exit__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        self.close()

    def open(self):
        if False:
            while True:
                i = 10
        if self.session is None:
            self.session = Session()

    def close(self):
        if False:
            print('Hello World!')
        if self.session is not None:
            self.session.close()
            self.session = None

    def _request(self, method, path, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if not self.session:
            raise RuntimeError('Session not opened')
        url = urljoin(self.url, path)
        kwargs.setdefault('headers', {})['x-sentry-project-id'] = self.project_id
        kwargs.setdefault('headers', {})['x-sentry-event-id'] = self.event_id
        kwargs.setdefault('headers', {})['x-sentry-worker-id'] = self.worker_id
        attempts = 0
        wait = 0.5
        while True:
            try:
                with metrics.timer('events.symbolicator.session.request', tags={'attempt': attempts}):
                    response = self.session.request(method, url, timeout=self.timeout + 1, **kwargs)
                metrics.incr('events.symbolicator.status_code', tags={'status_code': response.status_code})
                if method.lower() == 'get' and path.startswith('requests/') and (response.status_code == 404):
                    raise TaskIdNotFound()
                if response.status_code in (502, 503):
                    raise ServiceUnavailable()
                if response.ok:
                    json = response.json()
                    if json['status'] != 'pending':
                        metrics.timing('events.symbolicator.response.completed.size', len(response.content))
                else:
                    with sentry_sdk.push_scope():
                        sentry_sdk.set_extra('symbolicator_response', response.text)
                        sentry_sdk.capture_message('Symbolicator request failed')
                    json = {'status': 'failed', 'message': 'internal server error'}
                return json
            except (OSError, RequestException) as e:
                metrics.incr('events.symbolicator.request_error', tags={'exc': '.'.join([e.__class__.__module__, e.__class__.__name__]), 'attempt': attempts})
                attempts += 1
                if attempts > MAX_ATTEMPTS:
                    logger.error('Failed to contact symbolicator', exc_info=True)
                    raise
                time.sleep(wait)
                wait *= 2.0

    def create_task(self, path, **kwargs):
        if False:
            i = 10
            return i + 15
        params = {'timeout': self.timeout, 'scope': self.project_id}
        with metrics.timer('events.symbolicator.create_task', tags={'path': path}):
            return self._request(method='post', path=path, params=params, **kwargs)

    def query_task(self, task_id):
        if False:
            for i in range(10):
                print('nop')
        params = {'timeout': self.timeout, 'scope': self.project_id}
        task_url = f'requests/{task_id}'
        with metrics.timer('events.symbolicator.query_task'):
            return self._request('get', task_url, params=params)

    @classmethod
    def _get_worker_id(cls) -> str:
        if False:
            return 10
        if random.random() <= options.get('symbolicator.worker-id-randomization-sample-rate'):
            return uuid.uuid4().hex
        if cls._worker_id is None:
            cls._worker_id = str(uuid.uuid4().int % 5000)
        return cls._worker_id

    @classmethod
    def _reset_worker_id(cls):
        if False:
            print('Hello World!')
        cls._worker_id = None

    def reset_worker_id(self):
        if False:
            i = 10
            return i + 15
        self._reset_worker_id()
        self.worker_id = self._get_worker_id()