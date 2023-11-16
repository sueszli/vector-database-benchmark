import re
from contextlib import contextmanager
from typing import Optional
import requests
from werkzeug.datastructures import Headers
from werkzeug.wrappers import Request as WerkzeugRequest
from localstack.aws.api import RequestContext
from localstack.aws.chain import HandlerChain
from localstack.aws.gateway import Gateway
from localstack.config import HostAndPort
from localstack.http import Response
from localstack.http.hypercorn import GatewayServer, ProxyServer
from localstack.utils.net import IP_REGEX, get_free_tcp_port
from localstack.utils.serving import Server

@contextmanager
def server_context(server: Server, timeout: Optional[float]=10):
    if False:
        print('Hello World!')
    server.start()
    server.wait_is_up(timeout)
    try:
        yield server
    finally:
        server.shutdown()

def test_gateway_server():
    if False:
        i = 10
        return i + 15

    def echo_request_handler(_: HandlerChain, context: RequestContext, response: Response):
        if False:
            print('Hello World!')
        response.set_response(context.request.data)
        response.status_code = 200
        response.headers = context.request.headers
    gateway = Gateway()
    gateway.request_handlers.append(echo_request_handler)
    gateway_listen = HostAndPort(host='127.0.0.1', port=get_free_tcp_port())
    server = GatewayServer(gateway, gateway_listen, use_ssl=True)
    with server_context(server):
        get_response = requests.get(f'https://localhost.localstack.cloud:{gateway_listen.port}', data="Let's see if this works...")
        assert get_response.text == "Let's see if this works..."

def test_proxy_server(httpserver):
    if False:
        return 10
    httpserver.expect_request('/base-path/relative-path').respond_with_data('Reached Mock Server.')
    gateway_listen = HostAndPort(host='127.0.0.1', port=get_free_tcp_port())
    proxy_server = ProxyServer(httpserver.url_for('/base-path'), gateway_listen, use_ssl=True)
    with server_context(proxy_server):
        response = requests.get(f'https://localhost.localstack.cloud:{gateway_listen.port}/relative-path', data='data')
        assert response.text == 'Reached Mock Server.'

def test_proxy_server_properly_handles_headers(httpserver):
    if False:
        for i in range(10):
            print('nop')
    gateway_listen = HostAndPort(host='127.0.0.1', port=get_free_tcp_port())

    def header_echo_handler(request: WerkzeugRequest) -> Response:
        if False:
            return 10
        headers = Headers(request.headers)
        assert 'Multi-Value-Header' in headers
        assert headers['Multi-Value-Header'] == 'Value-1,Value-2'
        assert headers['Host'] == f'localhost.localstack.cloud:{gateway_listen.port}'
        assert len(request.access_route) == 2
        assert request.access_route[0] == '127.0.0.3'
        assert re.match(IP_REGEX, request.access_route[1])
        return Response(headers=headers)
    httpserver.expect_request('').respond_with_handler(header_echo_handler)
    proxy_server = ProxyServer(httpserver.url_for('/'), gateway_listen, use_ssl=True)
    with server_context(proxy_server):
        response = requests.request('GET', f'https://localhost.localstack.cloud:{gateway_listen.port}/', headers={'Multi-Value-Header': 'Value-1,Value-2', 'X-Forwarded-For': '127.0.0.3'})
        assert 'Multi-Value-Header' in response.headers
        assert response.headers['Multi-Value-Header'] == 'Value-1,Value-2'

def test_proxy_server_with_chunked_request(httpserver, httpserver_echo_request_metadata):
    if False:
        while True:
            i = 10
    chunks = [bytes(f'{n:2}', 'utf-8') for n in range(0, 100)]

    def handler(request: WerkzeugRequest) -> Response:
        if False:
            return 10
        assert b''.join(chunks) == request.get_data(parse_form_data=False)
        return Response()
    httpserver.expect_request('/').respond_with_handler(handler)
    gateway_listen = HostAndPort(host='127.0.0.1', port=get_free_tcp_port())
    proxy_server = ProxyServer(httpserver.url_for('/'), gateway_listen, use_ssl=True)

    def chunk_generator():
        if False:
            i = 10
            return i + 15
        for chunk in chunks:
            yield chunk
    with server_context(proxy_server):
        response = requests.get(f'https://localhost.localstack.cloud:{gateway_listen.port}/', data=chunk_generator())
        assert response

def test_proxy_server_with_streamed_response(httpserver):
    if False:
        while True:
            i = 10
    chunks = [bytes(f'{n:2}', 'utf-8') for n in range(0, 100)]

    def chunk_generator():
        if False:
            while True:
                i = 10
        for chunk in chunks:
            yield chunk

    def stream_response_handler(_: WerkzeugRequest) -> Response:
        if False:
            i = 10
            return i + 15
        return Response(response=chunk_generator())
    httpserver.expect_request('').respond_with_handler(stream_response_handler)
    gateway_listen = HostAndPort(host='127.0.0.1', port=get_free_tcp_port())
    proxy_server = ProxyServer(httpserver.url_for('/'), gateway_listen, use_ssl=True)
    with server_context(proxy_server):
        with requests.get(f'https://localhost.localstack.cloud:{gateway_listen.port}/', stream=True) as r:
            r.raise_for_status()
            chunk_iterator = r.iter_content(chunk_size=None)
            received_chunks = list(chunk_iterator)
            assert b''.join(chunks) == b''.join(received_chunks)