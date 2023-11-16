from queue import Queue
import pytest
from werkzeug.exceptions import NotFound
from localstack import config
from localstack.http import Request, Response, Router
from localstack.http.client import HttpClient
from localstack.http.dispatcher import handler_dispatcher
from localstack.http.proxy import Proxy
from localstack.services.s3.virtual_host import S3VirtualHostProxyHandler, add_s3_vhost_rules

class _RequestCollectingClient(HttpClient):
    requests: Queue

    def __init__(self):
        if False:
            return 10
        self.requests = Queue()

    def request(self, request: Request, server: str | None=None) -> Response:
        if False:
            for i in range(10):
                print('nop')
        self.requests.put((request, server))
        return Response()

    def create_proxy(self) -> Proxy:
        if False:
            for i in range(10):
                print('nop')
        '\n        Factory used to plug into S3VirtualHostProxyHandler._create_proxy\n        :return: a proxy using this client\n        '
        return Proxy(config.internal_service_url(host='localhost'), preserve_host=False, client=self)

class TestS3VirtualHostProxyHandler:

    def test_vhost_without_region(self):
        if False:
            return 10
        router = Router(dispatcher=handler_dispatcher())
        collector = _RequestCollectingClient()
        handler = S3VirtualHostProxyHandler()
        handler._create_proxy = collector.create_proxy
        add_s3_vhost_rules(router, handler)
        router.dispatch(Request(path='/my/key', headers={'Host': 'abucket.s3.localhost.localstack.cloud:4566'}))
        (request, server) = collector.requests.get()
        assert request.url == 'http://s3.localhost.localstack.cloud:4566/abucket/my/key'
        assert server == 'http://localhost:4566'
        router.dispatch(Request(path='/', headers={'Host': 'abucket.s3.localhost.localstack.cloud:4566'}))
        (request, server) = collector.requests.get()
        assert request.url == 'http://s3.localhost.localstack.cloud:4566/abucket/'
        router.dispatch(Request(path='/key', headers={'Host': 'abucket.s3.amazonaws.com'}))
        (request, server) = collector.requests.get()
        assert request.url == 'http://s3.localhost.localstack.cloud:4566/abucket/key'

    def test_vhost_with_region(self):
        if False:
            while True:
                i = 10
        router = Router(dispatcher=handler_dispatcher())
        collector = _RequestCollectingClient()
        handler = S3VirtualHostProxyHandler()
        handler._create_proxy = collector.create_proxy
        add_s3_vhost_rules(router, handler)
        router.dispatch(Request(path='/my/key', headers={'Host': 'abucket.s3.eu-central-1.localhost.localstack.cloud:4566'}))
        (request, server) = collector.requests.get()
        assert request.url == 'http://s3.localhost.localstack.cloud:4566/abucket/my/key'
        assert server == 'http://localhost:4566'
        router.dispatch(Request(path='/my/key', headers={'Host': 'abucket.s3.us-gov-east-1a.localhost.localstack.cloud:4566'}))
        (request, server) = collector.requests.get()
        assert request.url == 'http://s3.localhost.localstack.cloud:4566/abucket/my/key'
        assert server == 'http://localhost:4566'
        router.dispatch(Request(path='/', headers={'Host': 'abucket.s3.eu-central-1.localhost.localstack.cloud:4566'}))
        (request, server) = collector.requests.get()
        assert request.url == 'http://s3.localhost.localstack.cloud:4566/abucket/'
        router.dispatch(Request(path='/key', headers={'Host': 'abucket.s3.eu-central-1.amazonaws.com'}))
        (request, server) = collector.requests.get()
        assert request.url == 'http://s3.localhost.localstack.cloud:4566/abucket/key'

    def test_path_without_region(self):
        if False:
            print('Hello World!')
        router = Router(dispatcher=handler_dispatcher())
        collector = _RequestCollectingClient()
        handler = S3VirtualHostProxyHandler()
        handler._create_proxy = collector.create_proxy
        add_s3_vhost_rules(router, handler)
        with pytest.raises(NotFound):
            router.dispatch(Request(path='/abucket/my/key', headers={'Host': 's3.localhost.localstack.cloud:4566'}))

    def test_path_with_region(self):
        if False:
            print('Hello World!')
        router = Router(dispatcher=handler_dispatcher())
        collector = _RequestCollectingClient()
        handler = S3VirtualHostProxyHandler()
        handler._create_proxy = collector.create_proxy
        add_s3_vhost_rules(router, handler)
        router.dispatch(Request(path='/abucket/my/key', headers={'Host': 's3.eu-central-1.localhost.localstack.cloud:4566'}))
        (request, server) = collector.requests.get()
        assert request.url == 'http://s3.localhost.localstack.cloud:4566/abucket/my/key'
        assert server == 'http://localhost:4566'
        router.dispatch(Request(path='/abucket', headers={'Host': 's3.eu-central-1.localhost.localstack.cloud:4566'}))
        (request, server) = collector.requests.get()
        assert request.url == 'http://s3.localhost.localstack.cloud:4566/abucket'
        router.dispatch(Request(path='/abucket/key', headers={'Host': 's3.eu-central-1.amazonaws.com'}))
        (request, server) = collector.requests.get()
        assert request.url == 'http://s3.localhost.localstack.cloud:4566/abucket/key'

def test_vhost_rule_matcher():
    if False:
        i = 10
        return i + 15

    def echo_params(request, params):
        if False:
            return 10
        r = Response()
        r.set_json(params)
        return r
    router = Router()
    add_s3_vhost_rules(router, echo_params)
    assert router.dispatch(Request(path='/abucket/key', headers={'Host': 's3.eu-central-1.amazonaws.com'})).json == {'bucket': 'abucket', 'region': 'eu-central-1.', 'domain': 'amazonaws.com', 'path': 'key'}
    assert router.dispatch(Request(path='/my/key', headers={'Host': 'abucket.s3.eu-central-1.localhost.localstack.cloud:4566'})).json == {'bucket': 'abucket', 'region': 'eu-central-1.', 'domain': 'localhost.localstack.cloud:4566', 'path': 'my/key'}
    assert router.dispatch(Request(path='/my/key', headers={'Host': 'abucket.s3.localhost.localstack.cloud:4566'})).json == {'bucket': 'abucket', 'region': '', 'domain': 'localhost.localstack.cloud:4566', 'path': 'my/key'}