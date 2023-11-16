import sys
from azure.core.pipeline import PipelineRequest
from azure.core.rest import HttpRequest, HttpResponse
from azure.core import PipelineClient
from azure.core.pipeline.policies import RedirectPolicy
from azure.core.pipeline.policies import UserAgentPolicy
from azure.core.pipeline.policies import SansIOHTTPPolicy
from azure.core.pipeline.policies import RequestIdPolicy

def test_example_headers_policy():
    if False:
        return 10
    url = 'https://bing.com'
    policies = [UserAgentPolicy('myuseragent'), RedirectPolicy()]
    from azure.core.pipeline.policies import HeadersPolicy
    headers_policy = HeadersPolicy()
    headers_policy.add_header('CustomValue', 'Foo')
    policies.append(headers_policy)
    client: PipelineClient[HttpRequest, HttpResponse] = PipelineClient(base_url=url, policies=policies)
    request = HttpRequest('GET', url)
    pipeline_response = client._pipeline.run(request, headers={'CustomValue': 'Bar'})
    response = pipeline_response.http_response
    assert isinstance(response.status_code, int)

def test_example_request_id_policy():
    if False:
        print('Hello World!')
    url = 'https://bing.com'
    policies = [UserAgentPolicy('myuseragent'), RedirectPolicy()]
    from azure.core.pipeline.policies import HeadersPolicy
    request_id_policy = RequestIdPolicy()
    request_id_policy.set_request_id('azconfig-test')
    policies.append(request_id_policy)
    client: PipelineClient[HttpRequest, HttpResponse] = PipelineClient(base_url=url, policies=policies)
    request = HttpRequest('GET', url)
    pipeline_response = client._pipeline.run(request, request_id='azconfig-test')
    response = pipeline_response.http_response
    assert isinstance(response.status_code, int)

def test_example_user_agent_policy():
    if False:
        while True:
            i = 10
    url = 'https://bing.com'
    redirect_policy = RedirectPolicy()
    from azure.core.pipeline.policies import UserAgentPolicy
    user_agent_policy = UserAgentPolicy()
    user_agent_policy.add_user_agent('CustomValue')
    policies = [redirect_policy, user_agent_policy]
    client: PipelineClient[HttpRequest, HttpResponse] = PipelineClient(base_url=url, policies=policies)
    request = HttpRequest('GET', url)
    pipeline_response = client._pipeline.run(request, user_agent='AnotherValue')
    response = pipeline_response.http_response
    assert isinstance(response.status_code, int)

def example_network_trace_logging():
    if False:
        i = 10
        return i + 15
    filename = 'log.txt'
    url = 'https://bing.com'
    policies = [UserAgentPolicy('myuseragent'), RedirectPolicy()]
    from azure.core.pipeline.policies import NetworkTraceLoggingPolicy
    import sys
    import logging
    logger = logging.getLogger('azure')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(handler)
    file_handler = logging.FileHandler(filename)
    logger.addHandler(file_handler)
    logging_policy = NetworkTraceLoggingPolicy()
    logging_policy.enable_http_logger = True
    policies.append(logging_policy)
    client: PipelineClient[HttpRequest, HttpResponse] = PipelineClient(base_url=url, policies=policies)
    request = HttpRequest('GET', url)
    pipeline_response = client._pipeline.run(request, logging_enable=True)
    response = pipeline_response.http_response
    assert isinstance(response.status_code, int)

def example_proxy_policy():
    if False:
        return 10
    from azure.core.pipeline.policies import ProxyPolicy
    proxy_policy = ProxyPolicy()
    proxy_policy.proxies = {'http': 'http://10.10.1.10:3148'}
    proxy_policy.proxies = {'https': 'http://user:password@10.10.1.10:1180/'}

def test_example_per_call_policy():
    if False:
        for i in range(10):
            print('nop')
    'Per call policy example.\n\n    This example shows how to define your own policy and inject it with the "per_call_policies" parameter.\n    '
    from azure.core.pipeline.policies import SansIOHTTPPolicy

    class MyPolicy(SansIOHTTPPolicy[HttpRequest, HttpResponse]):

        def on_request(self, request: PipelineRequest[HttpRequest]) -> None:
            if False:
                i = 10
                return i + 15
            current_url = request.http_request.url
            request.http_request.url = current_url.replace('google', 'bing')
    client: PipelineClient[HttpRequest, HttpResponse] = PipelineClient(base_url='https://google.com', per_call_policies=MyPolicy())
    request = HttpRequest('GET', 'https://google.com/')
    response: HttpResponse = client.send_request(request)
    assert 'bing' in response.url