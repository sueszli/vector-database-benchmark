from typing import Iterable, MutableSequence, Union
from azure.core.pipeline import Pipeline
from azure.core import PipelineClient
from azure.core.rest import HttpRequest, HttpResponse
from azure.core.pipeline.policies import HTTPPolicy, SansIOHTTPPolicy, UserAgentPolicy, RedirectPolicy, RetryPolicy

def test_example_requests():
    if False:
        i = 10
        return i + 15
    request = HttpRequest('GET', 'https://bing.com')
    policies: Iterable[Union[HTTPPolicy, SansIOHTTPPolicy]] = [UserAgentPolicy('myuseragent'), RedirectPolicy(), RetryPolicy()]
    from azure.core.pipeline.transport import RequestsTransport
    with Pipeline(transport=RequestsTransport(), policies=policies) as pipeline:
        response = pipeline.run(request)
    assert isinstance(response.http_response.status_code, int)

def test_example_pipeline():
    if False:
        i = 10
        return i + 15
    from azure.core.pipeline import Pipeline
    from azure.core.pipeline.policies import RedirectPolicy, UserAgentPolicy
    from azure.core.rest import HttpRequest
    from azure.core.pipeline.transport import RequestsTransport
    request = HttpRequest('GET', 'https://bing.com')
    policies: Iterable[Union[HTTPPolicy, SansIOHTTPPolicy]] = [UserAgentPolicy('myuseragent'), RedirectPolicy()]
    with Pipeline(transport=RequestsTransport(), policies=policies) as pipeline:
        response = pipeline.run(request)
    assert isinstance(response.http_response.status_code, int)

def test_example_pipeline_client():
    if False:
        while True:
            i = 10
    url = 'https://bing.com'
    from azure.core import PipelineClient
    from azure.core.rest import HttpRequest
    from azure.core.pipeline.policies import RedirectPolicy, UserAgentPolicy
    policies: Iterable[Union[HTTPPolicy, SansIOHTTPPolicy]] = [UserAgentPolicy('myuseragent'), RedirectPolicy()]
    client: PipelineClient[HttpRequest, HttpResponse] = PipelineClient(base_url=url, policies=policies)
    request = HttpRequest('GET', 'https://bing.com')
    pipeline_response = client._pipeline.run(request)
    response = pipeline_response.http_response
    assert isinstance(response.status_code, int)

def test_example_redirect_policy():
    if False:
        print('Hello World!')
    url = 'https://bing.com'
    from azure.core.rest import HttpRequest
    from azure.core.pipeline.policies import RedirectPolicy
    redirect_policy = RedirectPolicy()
    redirect_policy.allow = True
    redirect_policy.max_redirects = 10
    redirect_policy = RedirectPolicy.no_redirects()
    client: PipelineClient[HttpRequest, HttpResponse] = PipelineClient(base_url=url, policies=[redirect_policy])
    request = HttpRequest('GET', url)
    pipeline_response = client._pipeline.run(request, permit_redirects=True, redirect_max=5)
    response = pipeline_response.http_response
    assert isinstance(response.status_code, int)

def test_example_no_redirects():
    if False:
        print('Hello World!')
    url = 'https://bing.com'
    redirect_policy = RedirectPolicy.no_redirects()
    client: PipelineClient[HttpRequest, HttpResponse] = PipelineClient(base_url=url, policies=[redirect_policy])
    request = HttpRequest('GET', url)
    pipeline_response = client._pipeline.run(request)
    response = pipeline_response.http_response
    assert response.status_code == 301

def test_example_retry_policy():
    if False:
        print('Hello World!')
    url = 'https://bing.com'
    policies: MutableSequence[Union[HTTPPolicy, SansIOHTTPPolicy]] = [UserAgentPolicy('myuseragent'), RedirectPolicy()]
    from azure.core.pipeline.policies import RetryPolicy
    retry_policy = RetryPolicy()
    retry_policy.total_retries = 5
    retry_policy.connect_retries = 2
    retry_policy.read_retries = 4
    retry_policy.status_retries = 3
    retry_policy.backoff_factor = 0.5
    retry_policy.backoff_max = 120
    retry_policy = RetryPolicy.no_retries()
    policies.append(retry_policy)
    client: PipelineClient[HttpRequest, HttpResponse] = PipelineClient(base_url=url, policies=policies)
    request = HttpRequest('GET', url)
    pipeline_response = client._pipeline.run(request, retry_total=10, retry_connect=1, retry_read=1, retry_status=5, retry_backoff_factor=0.5, retry_backoff_max=120, retry_on_methods=['GET'])
    response = pipeline_response.http_response
    assert isinstance(response.status_code, int)

def test_example_no_retries():
    if False:
        return 10
    url = 'https://bing.com'
    retry_policy = RetryPolicy.no_retries()
    client: PipelineClient[HttpRequest, HttpResponse] = PipelineClient(base_url=url, policies=[retry_policy])
    request = HttpRequest('GET', url)
    pipeline_response = client._pipeline.run(request)
    response = pipeline_response.http_response
    assert response.status_code == 301