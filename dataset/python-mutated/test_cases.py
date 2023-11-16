from collections import defaultdict
from elastic_transport import ApiResponseMeta, HttpHeaders
from elasticsearch import Elasticsearch

class DummyTransport:

    def __init__(self, hosts, responses=None, **_):
        if False:
            return 10
        self.hosts = hosts
        self.responses = responses
        self.call_count = 0
        self.calls = defaultdict(list)

    def perform_request(self, method, target, **kwargs):
        if False:
            print('Hello World!')
        (status, resp) = (200, {})
        if self.responses:
            (status, resp) = self.responses[self.call_count]
        self.call_count += 1
        self.calls[method, target].append(kwargs)
        return (ApiResponseMeta(status=status, http_version='1.1', headers=HttpHeaders({'X-elastic-product': 'Elasticsearch'}), duration=0.0, node=None), resp)

class DummyAsyncTransport:

    def __init__(self, hosts, responses=None, **_):
        if False:
            return 10
        self.hosts = hosts
        self.responses = responses
        self.call_count = 0
        self.calls = defaultdict(list)

    async def perform_request(self, method, target, **kwargs):
        (status, resp) = (200, {})
        if self.responses:
            (status, resp) = self.responses[self.call_count]
        self.call_count += 1
        self.calls[method, target].append(kwargs)
        return (ApiResponseMeta(status=status, http_version='1.1', headers=HttpHeaders({'X-elastic-product': 'Elasticsearch'}), duration=0.0, node=None), resp)

class DummyTransportTestCase:

    def setup_method(self, _):
        if False:
            for i in range(10):
                print('nop')
        self.client = Elasticsearch('http://localhost:9200', transport_class=DummyTransport)

    def assert_call_count_equals(self, count):
        if False:
            i = 10
            return i + 15
        assert count == self.client.transport.call_count

    def assert_url_called(self, method, url, count=1):
        if False:
            i = 10
            return i + 15
        assert (method, url) in self.client.transport.calls
        calls = self.client.transport.calls[method, url]
        assert count == len(calls)
        return calls