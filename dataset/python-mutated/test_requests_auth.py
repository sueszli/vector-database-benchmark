import warnings
import pytest
import requests
from elastic_transport import RequestsHttpNode, Urllib3HttpNode
from elastic_transport.client_utils import DEFAULT
from requests.auth import HTTPBasicAuth
from elasticsearch import AsyncElasticsearch, Elasticsearch

class CustomRequestHttpNode(RequestsHttpNode):
    pass

class CustomUrllib3HttpNode(Urllib3HttpNode):
    pass

@pytest.mark.parametrize('node_class', ['requests', RequestsHttpNode, CustomRequestHttpNode])
def test_requests_auth(node_class):
    if False:
        print('Hello World!')
    http_auth = HTTPBasicAuth('username', 'password')
    with warnings.catch_warnings(record=True) as w:
        client = Elasticsearch('http://localhost:9200', http_auth=http_auth, node_class=node_class)
    assert len(w) == 0
    node = client.transport.node_pool.get()
    assert isinstance(node, RequestsHttpNode)
    assert isinstance(node.session, requests.Session)
    assert node.session.auth is http_auth

@pytest.mark.parametrize('client_class', [Elasticsearch, AsyncElasticsearch])
@pytest.mark.parametrize('node_class', ['urllib3', 'aiohttp', None, DEFAULT, CustomUrllib3HttpNode])
def test_error_for_requests_auth_node_class(client_class, node_class):
    if False:
        print('Hello World!')
    http_auth = HTTPBasicAuth('username', 'password')
    with pytest.raises(ValueError) as e:
        client_class('http://localhost:9200', http_auth=http_auth, node_class=node_class)
    assert str(e.value) == "Using a custom 'requests.auth.AuthBase' class for 'http_auth' must be used with node_class='requests'"

def test_error_for_requests_auth_async():
    if False:
        for i in range(10):
            print('nop')
    http_auth = HTTPBasicAuth('username', 'password')
    with pytest.raises(ValueError) as e:
        AsyncElasticsearch('http://localhost:9200', http_auth=http_auth, node_class='requests')
    assert str(e.value) == "Specified 'node_class' is not async, should be async instead"