"""Tests for the custom hook policy."""
try:
    from unittest import mock
except ImportError:
    import mock
from azure.core import PipelineClient
from azure.core.pipeline.policies import CustomHookPolicy, UserAgentPolicy
from azure.core.pipeline.transport import HttpTransport
import pytest
from utils import HTTP_REQUESTS

@pytest.mark.parametrize('http_request', HTTP_REQUESTS)
def test_response_hook_policy_in_init(http_request):
    if False:
        i = 10
        return i + 15

    def test_callback(response):
        if False:
            for i in range(10):
                print('nop')
        raise ValueError()
    transport = mock.MagicMock(spec=HttpTransport)
    url = 'http://localhost'
    custom_hook_policy = CustomHookPolicy(raw_response_hook=test_callback)
    policies = [UserAgentPolicy('myuseragent'), custom_hook_policy]
    client = PipelineClient(base_url=url, policies=policies, transport=transport)
    request = http_request('GET', url)
    with pytest.raises(ValueError):
        client._pipeline.run(request)

@pytest.mark.parametrize('http_request', HTTP_REQUESTS)
def test_response_hook_policy_in_request(http_request):
    if False:
        return 10

    def test_callback(response):
        if False:
            for i in range(10):
                print('nop')
        raise ValueError()
    transport = mock.MagicMock(spec=HttpTransport)
    url = 'http://localhost'
    custom_hook_policy = CustomHookPolicy()
    policies = [UserAgentPolicy('myuseragent'), custom_hook_policy]
    client = PipelineClient(base_url=url, policies=policies, transport=transport)
    request = http_request('GET', url)
    with pytest.raises(ValueError):
        client._pipeline.run(request, raw_response_hook=test_callback)

@pytest.mark.parametrize('http_request', HTTP_REQUESTS)
def test_response_hook_policy_in_both(http_request):
    if False:
        while True:
            i = 10

    def test_callback(response):
        if False:
            i = 10
            return i + 15
        raise ValueError()

    def test_callback_request(response):
        if False:
            return 10
        raise TypeError()
    transport = mock.MagicMock(spec=HttpTransport)
    url = 'http://localhost'
    custom_hook_policy = CustomHookPolicy(raw_response_hook=test_callback)
    policies = [UserAgentPolicy('myuseragent'), custom_hook_policy]
    client = PipelineClient(base_url=url, policies=policies, transport=transport)
    request = http_request('GET', url)
    with pytest.raises(TypeError):
        client._pipeline.run(request, raw_response_hook=test_callback_request)

@pytest.mark.parametrize('http_request', HTTP_REQUESTS)
def test_request_hook_policy_in_init(http_request):
    if False:
        while True:
            i = 10

    def test_callback(response):
        if False:
            return 10
        raise ValueError()
    transport = mock.MagicMock(spec=HttpTransport)
    url = 'http://localhost'
    custom_hook_policy = CustomHookPolicy(raw_request_hook=test_callback)
    policies = [UserAgentPolicy('myuseragent'), custom_hook_policy]
    client = PipelineClient(base_url=url, policies=policies, transport=transport)
    request = http_request('GET', url)
    with pytest.raises(ValueError):
        client._pipeline.run(request)

@pytest.mark.parametrize('http_request', HTTP_REQUESTS)
def test_request_hook_policy_in_request(http_request):
    if False:
        return 10

    def test_callback(response):
        if False:
            for i in range(10):
                print('nop')
        raise ValueError()
    transport = mock.MagicMock(spec=HttpTransport)
    url = 'http://localhost'
    custom_hook_policy = CustomHookPolicy()
    policies = [UserAgentPolicy('myuseragent'), custom_hook_policy]
    client = PipelineClient(base_url=url, policies=policies, transport=transport)
    request = http_request('GET', url)
    with pytest.raises(ValueError):
        client._pipeline.run(request, raw_request_hook=test_callback)

@pytest.mark.parametrize('http_request', HTTP_REQUESTS)
def test_request_hook_policy_in_both(http_request):
    if False:
        for i in range(10):
            print('nop')

    def test_callback(response):
        if False:
            print('Hello World!')
        raise ValueError()

    def test_callback_request(response):
        if False:
            print('Hello World!')
        raise TypeError()
    transport = mock.MagicMock(spec=HttpTransport)
    url = 'http://localhost'
    custom_hook_policy = CustomHookPolicy(raw_request_hook=test_callback)
    policies = [UserAgentPolicy('myuseragent'), custom_hook_policy]
    client = PipelineClient(base_url=url, policies=policies, transport=transport)
    request = http_request('GET', url)
    with pytest.raises(TypeError):
        client._pipeline.run(request, raw_request_hook=test_callback_request)