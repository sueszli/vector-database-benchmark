import time
import urllib.parse
import forcediphttpsadapter.adapters
import requests
from zope.interface import implementer
from warehouse import tasks
from warehouse.cache.origin.interfaces import IOriginCache
from warehouse.metrics.interfaces import IMetricsService

class UnsuccessfulPurgeError(Exception):
    pass

@tasks.task(bind=True, ignore_result=True, acks_late=True)
def purge_key(task, request, key):
    if False:
        for i in range(10):
            print('nop')
    cacher = request.find_service(IOriginCache)
    metrics = request.find_service(IMetricsService, context=None)
    request.log.info('Purging %s', key)
    try:
        cacher.purge_key(key, metrics=metrics)
    except (requests.ConnectionError, requests.HTTPError, requests.Timeout, UnsuccessfulPurgeError) as exc:
        request.log.error('Error purging %s: %s', key, str(exc))
        raise task.retry(exc=exc)

@implementer(IOriginCache)
class FastlyCache:

    def __init__(self, *, api_endpoint, api_connect_via, api_key, service_id, purger):
        if False:
            while True:
                i = 10
        self.api_endpoint = api_endpoint
        self.api_connect_via = api_connect_via
        self.api_key = api_key
        self.service_id = service_id
        self._purger = purger

    @classmethod
    def create_service(cls, context, request):
        if False:
            return 10
        return cls(api_endpoint=request.registry.settings.get('origin_cache.api_endpoint', 'https://api.fastly.com'), api_connect_via=request.registry.settings.get('origin_cache.api_connect_via', None), api_key=request.registry.settings['origin_cache.api_key'], service_id=request.registry.settings['origin_cache.service_id'], purger=request.task(purge_key).delay)

    def cache(self, keys, request, response, *, seconds=None, stale_while_revalidate=None, stale_if_error=None):
        if False:
            print('Hello World!')
        existing_keys = set(response.headers.get('Surrogate-Key', '').split())
        response.headers['Surrogate-Key'] = ' '.join(sorted(set(keys) | existing_keys))
        values = []
        if seconds is not None:
            values.append(f'max-age={seconds}')
        if stale_while_revalidate is not None:
            values.append(f'stale-while-revalidate={stale_while_revalidate}')
        if stale_if_error is not None:
            values.append(f'stale-if-error={stale_if_error}')
        if values:
            response.headers['Surrogate-Control'] = ', '.join(values)

    def purge(self, keys):
        if False:
            for i in range(10):
                print('nop')
        for key in keys:
            self._purger(key)

    def _purge_key(self, key, connect_via=None):
        if False:
            while True:
                i = 10
        path = '/service/{service_id}/purge/{key}'.format(service_id=self.service_id, key=key)
        url = urllib.parse.urljoin(self.api_endpoint, path)
        headers = {'Accept': 'application/json', 'Fastly-Key': self.api_key, 'Fastly-Soft-Purge': '1'}
        session = requests.Session()
        if connect_via is not None:
            session.mount(self.api_endpoint, forcediphttpsadapter.adapters.ForcedIPHTTPSAdapter(dest_ip=self.api_connect_via))
        resp = session.post(url, headers=headers)
        resp.raise_for_status()
        if resp.json().get('status') != 'ok':
            raise UnsuccessfulPurgeError(f'Could not purge {key!r}')

    def _double_purge_key(self, key, connect_via=None):
        if False:
            return 10
        self._purge_key(key, connect_via=connect_via)
        time.sleep(2)
        self._purge_key(key, connect_via=connect_via)

    def purge_key(self, key, metrics=None):
        if False:
            i = 10
            return i + 15
        try:
            self._purge_key(key, connect_via=self.api_connect_via)
        except requests.ConnectionError:
            if self.api_connect_via is None:
                raise
            else:
                metrics.increment('warehouse.cache.origin.fastly.connect_via.failed', tags=[f'ip_address:{self.api_connect_via}'])
                self._double_purge_key(key)

@implementer(IOriginCache)
class NullFastlyCache(FastlyCache):
    """Same as FastlyCache, but it doesn't issue any requests"""

    def _purge_key(self, key, connect_via=None):
        if False:
            i = 10
            return i + 15
        path = '/service/{service_id}/purge/{key}'.format(service_id=self.service_id, key=key)
        url = urllib.parse.urljoin(self.api_endpoint, path)
        headers = {'Accept': 'application/json', 'Fastly-Key': self.api_key, 'Fastly-Soft-Purge': '1'}
        print('Origin cache purge issued:')
        print(f'* URL: {url!r}')
        print(f'* Headers: {headers!r}')