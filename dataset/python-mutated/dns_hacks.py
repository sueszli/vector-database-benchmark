import sys
import socket
import concurrent
from concurrent import futures
import ipaddress
from typing import Optional
import dns
import dns.resolver
from .logging import get_logger
_logger = get_logger(__name__)
_dns_threads_executor = None

def configure_dns_depending_on_proxy(is_proxy: bool) -> None:
    if False:
        return 10
    if not hasattr(socket, '_getaddrinfo'):
        socket._getaddrinfo = socket.getaddrinfo
    if is_proxy:

        def getaddrinfo(host, port, *args, **kwargs):
            if False:
                print('Hello World!')
            if _is_force_system_dns_for_host(host):
                return socket._getaddrinfo(host, port, *args, **kwargs)
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (host, port))]
        socket.getaddrinfo = getaddrinfo
    elif sys.platform == 'win32':
        try:
            _prepare_windows_dns_hack()
        except Exception as e:
            _logger.exception('failed to apply windows dns hack.')
        else:
            socket.getaddrinfo = _fast_getaddrinfo
    else:
        socket.getaddrinfo = socket._getaddrinfo

def _prepare_windows_dns_hack():
    if False:
        for i in range(10):
            print('nop')
    resolver = dns.resolver.get_default_resolver()
    if resolver.cache is None:
        resolver.cache = dns.resolver.Cache()
    resolver.lifetime = max(resolver.lifetime or 1, 30.0)
    global _dns_threads_executor
    if _dns_threads_executor is None:
        _dns_threads_executor = concurrent.futures.ThreadPoolExecutor(max_workers=20, thread_name_prefix='dns_resolver')

def _is_force_system_dns_for_host(host: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return str(host) in ('localhost', 'localhost.')

def _fast_getaddrinfo(host, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')

    def needs_dns_resolving(host):
        if False:
            while True:
                i = 10
        try:
            ipaddress.ip_address(host)
            return False
        except ValueError:
            pass
        if _is_force_system_dns_for_host(host):
            return False
        return True

    def resolve_with_dnspython(host):
        if False:
            for i in range(10):
                print('nop')
        addrs = []
        expected_errors = (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, concurrent.futures.CancelledError, concurrent.futures.TimeoutError)
        ipv6_fut = _dns_threads_executor.submit(dns.resolver.resolve, host, dns.rdatatype.AAAA)
        ipv4_fut = _dns_threads_executor.submit(dns.resolver.resolve, host, dns.rdatatype.A)
        try:
            answers = ipv6_fut.result()
            addrs += [str(answer) for answer in answers]
        except expected_errors as e:
            pass
        except BaseException as e:
            _logger.info(f'dnspython failed to resolve dns (AAAA) for {repr(host)} with error: {repr(e)}')
        try:
            answers = ipv4_fut.result()
            addrs += [str(answer) for answer in answers]
        except expected_errors as e:
            if not addrs:
                raise socket.gaierror(11001, 'getaddrinfo failed') from e
        except BaseException as e:
            _logger.info(f'dnspython failed to resolve dns (A) for {repr(host)} with error: {repr(e)}')
        if addrs:
            return addrs
        return [host]
    addrs = [host]
    if needs_dns_resolving(host):
        addrs = resolve_with_dnspython(host)
    list_of_list_of_socketinfos = [socket._getaddrinfo(addr, *args, **kwargs) for addr in addrs]
    list_of_socketinfos = [item for lst in list_of_list_of_socketinfos for item in lst]
    return list_of_socketinfos