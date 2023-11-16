from __future__ import print_function, absolute_import, division
import socket
import unittest
import gevent.testing as greentest
from gevent.tests.test__socket_dns import TestCase, add
from gevent.testing.sysinfo import OSX
from gevent.testing.sysinfo import RESOLVER_DNSPYTHON
from gevent.testing.sysinfo import RESOLVER_ARES
from gevent.testing.sysinfo import PYPY
from gevent.testing.sysinfo import PY2

class Test6(TestCase):
    NORMALIZE_GHBA_IGNORE_ALIAS = True
    host = 'aaaa.test-ipv6.com'

    def _normalize_result_gethostbyaddr(self, result):
        if False:
            i = 10
            return i + 15
        return ()
    if RESOLVER_ARES and PY2:

        def _normalize_result_getnameinfo(self, result):
            if False:
                print('Hello World!')
            (ipaddr, service) = result
            if ipaddr.endswith('%0'):
                ipaddr = ipaddr[:-2]
            return (ipaddr, service)
    if not OSX and RESOLVER_DNSPYTHON:

        def _run_test_getnameinfo(self, *_args, **_kwargs):
            if False:
                i = 10
                return i + 15
            return ((), 0, (), 0)

    def _run_test_gethostbyname(self, *_args, **_kwargs):
        if False:
            i = 10
            return i + 15
        raise unittest.SkipTest('gethostbyname[_ex] does not support IPV6')
    _run_test_gethostbyname_ex = _run_test_gethostbyname

    def test_empty(self):
        if False:
            while True:
                i = 10
        self._test('getaddrinfo', self.host, 'http')

    def test_inet(self):
        if False:
            print('Hello World!')
        self._test('getaddrinfo', self.host, None, socket.AF_INET)

    def test_inet6(self):
        if False:
            return 10
        self._test('getaddrinfo', self.host, None, socket.AF_INET6)

    def test_unspec(self):
        if False:
            for i in range(10):
                print('nop')
        self._test('getaddrinfo', self.host, None, socket.AF_UNSPEC)

class Test6_google(Test6):
    host = 'ipv6.google.com'
    if greentest.RUNNING_ON_CI:

        def _normalize_result_getnameinfo(self, result):
            if False:
                for i in range(10):
                    print('nop')
            return ()
        if PYPY:
            _normalize_result_getaddrinfo = _normalize_result_getnameinfo
add(Test6, Test6.host)
add(Test6_google, Test6_google.host)

class Test6_ds(Test6):
    host = 'ds.test-ipv6.com'
    _normalize_result_gethostbyname = Test6._normalize_result_gethostbyaddr
add(Test6_ds, Test6_ds.host)
if __name__ == '__main__':
    greentest.main()