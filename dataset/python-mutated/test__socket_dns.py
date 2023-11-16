from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import gevent
from gevent import monkey
import os
import re
import unittest
import socket
from time import time
import traceback
import gevent.socket as gevent_socket
import gevent.testing as greentest
from gevent.testing import util
from gevent.testing.six import xrange
from gevent.testing import flaky
from gevent.testing.skipping import skipWithoutExternalNetwork
resolver = gevent.get_hub().resolver
util.debug('Resolver: %s', resolver)
if getattr(resolver, 'pool', None) is not None:
    resolver.pool.size = 1
from gevent.testing.sysinfo import RESOLVER_NOT_SYSTEM
from gevent.testing.sysinfo import RESOLVER_DNSPYTHON
from gevent.testing.sysinfo import RESOLVER_ARES
from gevent.testing.sysinfo import PY2
from gevent.testing.sysinfo import PYPY
import gevent.testing.timing
assert gevent_socket.gaierror is socket.gaierror
assert gevent_socket.error is socket.error
RUN_ALL_HOST_TESTS = os.getenv('GEVENTTEST_RUN_ALL_ETC_HOST_TESTS', '')

def add(klass, hostname, name=None, skip=None, skip_reason=None, require_equal_errors=True):
    if False:
        while True:
            i = 10
    call = callable(hostname)

    def _setattr(k, n, func):
        if False:
            return 10
        if skip:
            func = greentest.skipIf(skip, skip_reason)(func)
        if not hasattr(k, n):
            setattr(k, n, func)
    if name is None:
        if call:
            name = hostname.__name__
        else:
            name = re.sub('[^\\w]+', '_', repr(hostname))
        assert name, repr(hostname)

    def test_getaddrinfo_http(self):
        if False:
            for i in range(10):
                print('nop')
        x = hostname() if call else hostname
        self._test('getaddrinfo', x, 'http', require_equal_errors=require_equal_errors)
    test_getaddrinfo_http.__name__ = 'test_%s_getaddrinfo_http' % name
    _setattr(klass, test_getaddrinfo_http.__name__, test_getaddrinfo_http)

    def test_gethostbyname(self):
        if False:
            i = 10
            return i + 15
        x = hostname() if call else hostname
        ipaddr = self._test('gethostbyname', x, require_equal_errors=require_equal_errors)
        if not isinstance(ipaddr, Exception):
            self._test('gethostbyaddr', ipaddr, require_equal_errors=require_equal_errors)
    test_gethostbyname.__name__ = 'test_%s_gethostbyname' % name
    _setattr(klass, test_gethostbyname.__name__, test_gethostbyname)

    def test_gethostbyname_ex(self):
        if False:
            print('Hello World!')
        x = hostname() if call else hostname
        self._test('gethostbyname_ex', x, require_equal_errors=require_equal_errors)
    test_gethostbyname_ex.__name__ = 'test_%s_gethostbyname_ex' % name
    _setattr(klass, test_gethostbyname_ex.__name__, test_gethostbyname_ex)

    def test4(self):
        if False:
            for i in range(10):
                print('nop')
        x = hostname() if call else hostname
        self._test('gethostbyaddr', x, require_equal_errors=require_equal_errors)
    test4.__name__ = 'test_%s_gethostbyaddr' % name
    _setattr(klass, test4.__name__, test4)

    def test5(self):
        if False:
            i = 10
            return i + 15
        x = hostname() if call else hostname
        self._test('getnameinfo', (x, 80), 0, require_equal_errors=require_equal_errors)
    test5.__name__ = 'test_%s_getnameinfo' % name
    _setattr(klass, test5.__name__, test5)

@skipWithoutExternalNetwork('Tries to resolve and compare hostnames/addrinfo')
class TestCase(greentest.TestCase):
    maxDiff = None
    __timeout__ = 30
    switch_expected = None
    TRACE = not util.QUIET and os.getenv('GEVENT_DEBUG', '') == 'trace'
    verbose_dns = TRACE

    def trace(self, message, *args, **kwargs):
        if False:
            while True:
                i = 10
        if self.TRACE:
            util.debug(message, *args, **kwargs)
    REAL_ERRORS = (AttributeError, ValueError, NameError)

    def __run_resolver(self, function, args):
        if False:
            return 10
        try:
            result = function(*args)
            assert not isinstance(result, BaseException), repr(result)
            return result
        except self.REAL_ERRORS:
            raise
        except Exception as ex:
            if self.TRACE:
                traceback.print_exc()
            return ex

    def __trace_call(self, result, runtime, function, *args):
        if False:
            while True:
                i = 10
        util.debug(self.__format_call(function, args))
        self.__trace_fresult(result, runtime)

    def __format_call(self, function, args):
        if False:
            print('Hello World!')
        args = repr(args)
        if args.endswith(',)'):
            args = args[:-2] + ')'
        try:
            module = function.__module__.replace('gevent._socketcommon', 'gevent')
            name = function.__name__
            return '%s:%s%s' % (module, name, args)
        except AttributeError:
            return function + args

    def __trace_fresult(self, result, seconds):
        if False:
            while True:
                i = 10
        if isinstance(result, Exception):
            msg = '  -=>  raised %r' % (result,)
        else:
            msg = '  -=>  returned %r' % (result,)
        time_ms = ' %.2fms' % (seconds * 1000.0,)
        space = 80 - len(msg) - len(time_ms)
        if space > 0:
            space = ' ' * space
        else:
            space = ''
        util.debug(msg + space + time_ms)
    if not TRACE:

        def run_resolver(self, function, func_args):
            if False:
                while True:
                    i = 10
            now = time()
            return (self.__run_resolver(function, func_args), time() - now)
    else:

        def run_resolver(self, function, func_args):
            if False:
                i = 10
                return i + 15
            self.trace(self.__format_call(function, func_args))
            delta = time()
            result = self.__run_resolver(function, func_args)
            delta = time() - delta
            self.__trace_fresult(result, delta)
            return (result, delta)

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestCase, self).setUp()
        if not self.verbose_dns:
            gevent.get_hub().exception_stream = None

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.verbose_dns:
            try:
                del gevent.get_hub().exception_stream
            except AttributeError:
                pass
        super(TestCase, self).tearDown()

    def should_log_results(self, result1, result2):
        if False:
            i = 10
            return i + 15
        if not self.verbose_dns:
            return False
        if isinstance(result1, BaseException) and isinstance(result2, BaseException):
            return type(result1) is not type(result2)
        return repr(result1) != repr(result2)

    def _test(self, func_name, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Runs the function *func_name* with *args* and compares gevent and the system.\n\n        Keyword arguments are passed to the function itself; variable args are\n        used for the socket function.\n\n        Returns the gevent result.\n        '
        gevent_func = getattr(gevent_socket, func_name)
        real_func = monkey.get_original('socket', func_name)
        tester = getattr(self, '_run_test_' + func_name, self._run_test_generic)
        result = tester(func_name, real_func, gevent_func, args, **kwargs)
        (_real_result, time_real, gevent_result, time_gevent) = result
        if self.verbose_dns and time_gevent > time_real + 0.02 and (time_gevent > 0.03):
            msg = 'gevent:%s%s took %dms versus %dms stdlib' % (func_name, args, time_gevent * 1000.0, time_real * 1000.0)
            if time_gevent > time_real + 1:
                word = 'VERY'
            else:
                word = 'quite'
            util.log('\nWARNING: %s slow: %s', word, msg, color='warning')
        return gevent_result

    def _run_test_generic(self, func_name, real_func, gevent_func, func_args, require_equal_errors=True):
        if False:
            while True:
                i = 10
        (real_result, time_real) = self.run_resolver(real_func, func_args)
        (gevent_result, time_gevent) = self.run_resolver(gevent_func, func_args)
        if util.QUIET and self.should_log_results(real_result, gevent_result):
            util.log('')
            self.__trace_call(real_result, time_real, real_func, func_args)
            self.__trace_call(gevent_result, time_gevent, gevent_func, func_args)
        self.assertEqualResults(real_result, gevent_result, func_name, require_equal_errors=require_equal_errors)
        return (real_result, time_real, gevent_result, time_gevent)

    def _normalize_result(self, result, func_name):
        if False:
            return 10
        norm_name = '_normalize_result_' + func_name
        if hasattr(self, norm_name):
            return getattr(self, norm_name)(result)
        return result
    NORMALIZE_GAI_IGNORE_CANONICAL_NAME = RESOLVER_ARES
    if not RESOLVER_NOT_SYSTEM:

        def _normalize_result_getaddrinfo(self, result):
            if False:
                while True:
                    i = 10
            return result

        def _normalize_result_gethostbyname_ex(self, result):
            if False:
                return 10
            return result
    else:

        def _normalize_result_gethostbyname_ex(self, result):
            if False:
                return 10
            if isinstance(result, BaseException):
                return result
            try:
                result[2].sort()
            except AttributeError:
                pass
            except IndexError:
                return result
            ips = result[2]
            if ips == ['127.0.0.1', '127.0.0.1']:
                ips = ['127.0.0.1']
            return (result[0].lower(), [], ips)

        def _normalize_result_getaddrinfo(self, result):
            if False:
                i = 10
                return i + 15
            if isinstance(result, BaseException):
                return result
            if isinstance(result, list):
                result = [x for x in result if x[1] in (socket.SOCK_STREAM, socket.SOCK_DGRAM) and x[2] in (socket.IPPROTO_TCP, socket.IPPROTO_UDP)]
            if self.NORMALIZE_GAI_IGNORE_CANONICAL_NAME:
                result = [(family, kind, proto, '', addr) for (family, kind, proto, _, addr) in result]
            if isinstance(result, list):
                result.sort()
            return result

    def _normalize_result_getnameinfo(self, result):
        if False:
            return 10
        return result
    NORMALIZE_GHBA_IGNORE_ALIAS = False

    def _normalize_result_gethostbyaddr(self, result):
        if False:
            for i in range(10):
                print('nop')
        if not RESOLVER_NOT_SYSTEM:
            return result
        if self.NORMALIZE_GHBA_IGNORE_ALIAS and isinstance(result, tuple):
            return (result[0], [], result[2])
        return result

    def _compare_exceptions_strict(self, real_result, gevent_result, func_name):
        if False:
            for i in range(10):
                print('nop')
        if repr(real_result) == repr(gevent_result):
            return
        msg = (func_name, 'system:', repr(real_result), 'gevent:', repr(gevent_result))
        self.assertIs(type(gevent_result), type(real_result), msg)
        if isinstance(real_result, TypeError):
            return
        if PYPY and isinstance(real_result, socket.herror):
            return
        self.assertEqual(real_result.args, gevent_result.args, msg)
        if hasattr(real_result, 'errno'):
            self.assertEqual(real_result.errno, gevent_result.errno)

    def _compare_exceptions_lenient(self, real_result, gevent_result, func_name):
        if False:
            return 10
        try:
            self._compare_exceptions_strict(real_result, gevent_result, func_name)
        except AssertionError:
            if func_name not in ('getaddrinfo', 'gethostbyaddr', 'gethostbyname', 'gethostbyname_ex', 'getnameinfo') or type(real_result) not in (socket.herror, socket.gaierror) or type(gevent_result) not in (socket.herror, socket.gaierror, socket.error):
                raise
            util.log('WARNING: error type mismatch for %s: %r (gevent) != %r (stdlib)', func_name, gevent_result, real_result, color='warning')
    _compare_exceptions = _compare_exceptions_lenient if RESOLVER_NOT_SYSTEM else _compare_exceptions_strict

    def _compare_results(self, real_result, gevent_result, func_name):
        if False:
            print('Hello World!')
        if real_result == gevent_result:
            return True
        compare_func = getattr(self, '_compare_results_' + func_name, self._generic_compare_results)
        return compare_func(real_result, gevent_result, func_name)

    def _generic_compare_results(self, real_result, gevent_result, func_name):
        if False:
            i = 10
            return i + 15
        try:
            if len(real_result) != len(gevent_result):
                return False
        except TypeError:
            return False
        return all((self._compare_results(x, y, func_name) for (x, y) in zip(real_result, gevent_result)))

    def _compare_results_getaddrinfo(self, real_result, gevent_result, func_name):
        if False:
            for i in range(10):
                print('nop')
        errors = isinstance(real_result, Exception) + isinstance(gevent_result, Exception)
        if errors == 2:
            return True
        if errors == 1:
            return False
        if not set(real_result).isdisjoint(set(gevent_result)):
            return True
        return self._generic_compare_results(real_result, gevent_result, func_name)

    def _compare_address_strings(self, a, b):
        if False:
            while True:
                i = 10
        a_segments = a.count(':')
        b_segments = b.count(':')
        if a_segments and b_segments:
            if a_segments == b_segments and a_segments in (4, 5, 6, 7):
                return True
            if a.rstrip(':').startswith(b.rstrip(':')) or b.rstrip(':').startswith(a.rstrip(':')):
                return True
            if a_segments >= 2 and b_segments >= 2 and (a.split(':')[:2] == b.split(':')[:2]):
                return True
        return a.split('.', 1)[-1] == b.split('.', 1)[-1]

    def _compare_results_gethostbyname(self, real_result, gevent_result, _func_name):
        if False:
            for i in range(10):
                print('nop')
        return self._compare_address_strings(real_result, gevent_result)

    def _compare_results_gethostbyname_ex(self, real_result, gevent_result, _func_name):
        if False:
            for i in range(10):
                print('nop')
        return not set(real_result[2]).isdisjoint(set(gevent_result[2]))

    def assertEqualResults(self, real_result, gevent_result, func_name, require_equal_errors=True):
        if False:
            return 10
        errors = (OverflowError, TypeError, UnicodeError, socket.error, socket.gaierror, socket.herror)
        if isinstance(real_result, errors) and isinstance(gevent_result, errors):
            if require_equal_errors:
                self._compare_exceptions(real_result, gevent_result, func_name)
            return
        real_result = self._normalize_result(real_result, func_name)
        gevent_result = self._normalize_result(gevent_result, func_name)
        if self._compare_results(real_result, gevent_result, func_name):
            return
        if RESOLVER_NOT_SYSTEM and isinstance(real_result, errors) and (not isinstance(gevent_result, errors)):
            return
        if RESOLVER_NOT_SYSTEM and PYPY and (func_name == 'getnameinfo') and isinstance(gevent_result, socket.error) and (not isinstance(real_result, socket.error)):
            return
        self.assertEqual(real_result, gevent_result)

class TestTypeError(TestCase):
    pass
add(TestTypeError, None)
add(TestTypeError, 25)

class TestHostname(TestCase):
    NORMALIZE_GHBA_IGNORE_ALIAS = True

    def __normalize_name(self, result):
        if False:
            return 10
        if (RESOLVER_ARES or RESOLVER_DNSPYTHON) and isinstance(result, tuple):
            name = result[0]
            name = name.split('.', 1)[0]
            result = (name,) + result[1:]
        return result

    def _normalize_result_gethostbyaddr(self, result):
        if False:
            return 10
        result = TestCase._normalize_result_gethostbyaddr(self, result)
        return self.__normalize_name(result)

    def _normalize_result_getnameinfo(self, result):
        if False:
            for i in range(10):
                print('nop')
        result = TestCase._normalize_result_getnameinfo(self, result)
        if PY2:
            result = self.__normalize_name(result)
        return result
add(TestHostname, socket.gethostname, skip=greentest.RUNNING_ON_TRAVIS and greentest.RESOLVER_NOT_SYSTEM, skip_reason='Sometimes get a different result for getaddrinfo with dnspython; c-ares produces different results for localhost on Travis beginning Sept 2019')

class TestLocalhost(TestCase):

    def _normalize_result_getaddrinfo(self, result):
        if False:
            for i in range(10):
                print('nop')
        if RESOLVER_NOT_SYSTEM:
            return ()
        return super(TestLocalhost, self)._normalize_result_getaddrinfo(result)
    NORMALIZE_GHBA_IGNORE_ALIAS = True
    if greentest.RUNNING_ON_TRAVIS and greentest.PY2 and RESOLVER_NOT_SYSTEM:

        def _normalize_result_gethostbyaddr(self, result):
            if False:
                while True:
                    i = 10
            result = super(TestLocalhost, self)._normalize_result_gethostbyaddr(result)
            if isinstance(result, tuple):
                result = (result[0], result[1], ['127.0.0.1'])
            return result
add(TestLocalhost, 'ip6-localhost', skip=RESOLVER_DNSPYTHON, skip_reason='Can return gaierror(-2)')
add(TestLocalhost, 'localhost', skip=greentest.RUNNING_ON_TRAVIS, skip_reason='Can return gaierror(-2)')

class TestNonexistent(TestCase):
    pass
add(TestNonexistent, 'nonexistentxxxyyy')

class Test1234(TestCase):
    pass
add(Test1234, '1.2.3.4')

class Test127001(TestCase):
    NORMALIZE_GHBA_IGNORE_ALIAS = True
add(Test127001, '127.0.0.1')

class TestBroadcast(TestCase):
    switch_expected = False
    if RESOLVER_DNSPYTHON:

        @unittest.skip('ares raises errors for broadcasthost/255.255.255.255')
        def test__broadcast__gethostbyaddr(self):
            if False:
                print('Hello World!')
            return
        test__broadcast__gethostbyname = test__broadcast__gethostbyaddr
add(TestBroadcast, '<broadcast>')
from gevent.resolver._hostsfile import HostsFile

class SanitizedHostsFile(HostsFile):

    def iter_all_host_addr_pairs(self):
        if False:
            print('Hello World!')
        for (name, addr) in super(SanitizedHostsFile, self).iter_all_host_addr_pairs():
            if RESOLVER_NOT_SYSTEM and (name.endswith('local') or addr == '255.255.255.255' or name == 'broadcasthost' or (name == 'localhost')):
                continue
            if name.endswith('local'):
                continue
            yield (name, addr)

@greentest.skipIf(greentest.RUNNING_ON_CI, 'This sometimes randomly fails on Travis with ares and on appveyor, beginning Feb 13, 2018')
class TestEtcHosts(TestCase):
    MAX_HOSTS = int(os.getenv('GEVENTTEST_MAX_ETC_HOSTS', '10'))

    @classmethod
    def populate_tests(cls):
        if False:
            i = 10
            return i + 15
        hf = SanitizedHostsFile(os.path.join(os.path.dirname(__file__), 'hosts_file.txt'))
        all_etc_hosts = sorted(hf.iter_all_host_addr_pairs())
        if len(all_etc_hosts) > cls.MAX_HOSTS and (not RUN_ALL_HOST_TESTS):
            all_etc_hosts = all_etc_hosts[:cls.MAX_HOSTS]
        for (host, ip) in all_etc_hosts:
            add(cls, host)
            add(cls, ip)
TestEtcHosts.populate_tests()

class TestGeventOrg(TestCase):
    HOSTNAME = 'www.gevent.org'

    def _normalize_result_gethostbyname(self, result):
        if False:
            for i in range(10):
                print('nop')
        if result == '104.17.33.82':
            result = '104.17.32.82'
        return result

    def _normalize_result_gethostbyname_ex(self, result):
        if False:
            print('Hello World!')
        result = super(TestGeventOrg, self)._normalize_result_gethostbyname_ex(result)
        if result[0] == 'python-gevent.readthedocs.org':
            result = ('readthedocs.io',) + result[1:]
        return result

    def test_AI_CANONNAME(self):
        if False:
            print('Hello World!')
        args = (TestGeventOrg.HOSTNAME, None, socket.AF_INET, 0, 0, socket.AI_CANONNAME)
        gevent_result = gevent_socket.getaddrinfo(*args)
        self.assertEqual(gevent_result[0][3], 'readthedocs.io')
        real_result = socket.getaddrinfo(*args)
        self.NORMALIZE_GAI_IGNORE_CANONICAL_NAME = not all((r[3] for r in real_result))
        try:
            self.assertEqualResults(real_result, gevent_result, 'getaddrinfo')
        finally:
            del self.NORMALIZE_GAI_IGNORE_CANONICAL_NAME
add(TestGeventOrg, TestGeventOrg.HOSTNAME)

class TestFamily(TestCase):

    def test_inet(self):
        if False:
            while True:
                i = 10
        self._test('getaddrinfo', TestGeventOrg.HOSTNAME, None, socket.AF_INET)

    def test_unspec(self):
        if False:
            return 10
        self._test('getaddrinfo', TestGeventOrg.HOSTNAME, None, socket.AF_UNSPEC)

    def test_badvalue(self):
        if False:
            i = 10
            return i + 15
        self._test('getaddrinfo', TestGeventOrg.HOSTNAME, None, 255)
        self._test('getaddrinfo', TestGeventOrg.HOSTNAME, None, 255000)
        self._test('getaddrinfo', TestGeventOrg.HOSTNAME, None, -1)

    @unittest.skipIf(RESOLVER_DNSPYTHON, 'Raises the wrong errno')
    def test_badtype(self):
        if False:
            while True:
                i = 10
        self._test('getaddrinfo', TestGeventOrg.HOSTNAME, 'x')

class Test_getaddrinfo(TestCase):

    def _test_getaddrinfo(self, *args):
        if False:
            print('Hello World!')
        self._test('getaddrinfo', *args)

    def test_80(self):
        if False:
            print('Hello World!')
        self._test_getaddrinfo(TestGeventOrg.HOSTNAME, 80)

    def test_int_string(self):
        if False:
            i = 10
            return i + 15
        self._test_getaddrinfo(TestGeventOrg.HOSTNAME, '80')

    def test_0(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_getaddrinfo(TestGeventOrg.HOSTNAME, 0)

    def test_http(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_getaddrinfo(TestGeventOrg.HOSTNAME, 'http')

    def test_notexistent_tld(self):
        if False:
            while True:
                i = 10
        self._test_getaddrinfo('myhost.mytld', 53)

    def test_notexistent_dot_com(self):
        if False:
            return 10
        self._test_getaddrinfo('sdfsdfgu5e66098032453245wfdggd.com', 80)

    def test1(self):
        if False:
            return 10
        return self._test_getaddrinfo(TestGeventOrg.HOSTNAME, 52, socket.AF_UNSPEC, socket.SOCK_STREAM, 0, 0)

    def test2(self):
        if False:
            i = 10
            return i + 15
        return self._test_getaddrinfo(TestGeventOrg.HOSTNAME, 53, socket.AF_INET, socket.SOCK_DGRAM, 17)

    @unittest.skipIf(RESOLVER_DNSPYTHON, 'dnspython only returns some of the possibilities')
    def test3(self):
        if False:
            return 10
        return self._test_getaddrinfo('google.com', 'http', socket.AF_INET6)

    @greentest.skipIf(PY2, 'Enums only on Python 3.4+')
    def test_enums(self):
        if False:
            print('Hello World!')
        gai = gevent_socket.getaddrinfo('example.com', 80, socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        (af, socktype, _proto, _canonname, _sa) = gai[0]
        self.assertIs(socktype, socket.SOCK_STREAM)
        self.assertIs(af, socket.AF_INET)

class TestInternational(TestCase):
    if PY2:
        REAL_ERRORS = set(TestCase.REAL_ERRORS) - {ValueError}
        if RESOLVER_ARES:

            def test_russian_getaddrinfo_http(self):
                if False:
                    while True:
                        i = 10
                self.skipTest('ares fails to encode.')
add(TestInternational, u'президент.рф', 'russian', skip=PY2 and RESOLVER_DNSPYTHON, skip_reason='dnspython can actually resolve these', require_equal_errors=False)
add(TestInternational, u'президент.рф'.encode('idna'), 'idna', require_equal_errors=False)

@skipWithoutExternalNetwork('Tries to resolve and compare hostnames/addrinfo')
class TestInterrupted_gethostbyname(gevent.testing.timing.AbstractGenericWaitTestCase):

    @greentest.ignores_leakcheck
    def test_returns_none_after_timeout(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestInterrupted_gethostbyname, self).test_returns_none_after_timeout()

    def wait(self, timeout):
        if False:
            i = 10
            return i + 15
        with gevent.Timeout(timeout, False):
            for index in xrange(1000000):
                try:
                    gevent_socket.gethostbyname('www.x%s.com' % index)
                except socket.error:
                    pass
            raise AssertionError('Timeout was not raised')

    def cleanup(self):
        if False:
            return 10
        try:
            gevent.get_hub().threadpool.join()
        except Exception:
            traceback.print_exc()

class TestBadName(TestCase):
    pass
add(TestBadName, 'xxxxxxxxxxxx')

class TestBadIP(TestCase):
    pass
add(TestBadIP, '1.2.3.400')

@greentest.skipIf(greentest.RUNNING_ON_TRAVIS, 'Travis began returning ip6-localhost')
class Test_getnameinfo_127001(TestCase):

    def test(self):
        if False:
            return 10
        self._test('getnameinfo', ('127.0.0.1', 80), 0)

    def test_DGRAM(self):
        if False:
            print('Hello World!')
        self._test('getnameinfo', ('127.0.0.1', 779), 0)
        self._test('getnameinfo', ('127.0.0.1', 779), socket.NI_DGRAM)

    def test_NOFQDN(self):
        if False:
            print('Hello World!')
        self._test('getnameinfo', ('127.0.0.1', 80), socket.NI_NOFQDN)

    def test_NAMEREQD(self):
        if False:
            i = 10
            return i + 15
        self._test('getnameinfo', ('127.0.0.1', 80), socket.NI_NAMEREQD)

class Test_getnameinfo_geventorg(TestCase):

    @unittest.skipIf(RESOLVER_DNSPYTHON, 'dnspython raises an error when multiple results are returned')
    def test_NUMERICHOST(self):
        if False:
            while True:
                i = 10
        self._test('getnameinfo', (TestGeventOrg.HOSTNAME, 80), 0)
        self._test('getnameinfo', (TestGeventOrg.HOSTNAME, 80), socket.NI_NUMERICHOST)

    @unittest.skipIf(RESOLVER_DNSPYTHON, 'dnspython raises an error when multiple results are returned')
    def test_NUMERICSERV(self):
        if False:
            return 10
        self._test('getnameinfo', (TestGeventOrg.HOSTNAME, 80), socket.NI_NUMERICSERV)

    def test_domain1(self):
        if False:
            for i in range(10):
                print('nop')
        self._test('getnameinfo', (TestGeventOrg.HOSTNAME, 80), 0)

    def test_domain2(self):
        if False:
            while True:
                i = 10
        self._test('getnameinfo', ('www.gevent.org', 80), 0)

    def test_port_zero(self):
        if False:
            print('Hello World!')
        self._test('getnameinfo', ('www.gevent.org', 0), 0)

class Test_getnameinfo_fail(TestCase):

    def test_port_string(self):
        if False:
            while True:
                i = 10
        self._test('getnameinfo', ('www.gevent.org', 'http'), 0)

    def test_bad_flags(self):
        if False:
            i = 10
            return i + 15
        self._test('getnameinfo', ('localhost', 80), 55555555)

class TestInvalidPort(TestCase):

    @flaky.reraises_flaky_race_condition()
    def test_overflow_neg_one(self):
        if False:
            while True:
                i = 10
        self._test('getnameinfo', ('www.gevent.org', -1), 0)

    @greentest.skipOnLibuvOnPyPyOnWin('Errors dont match')
    def test_typeerror_none(self):
        if False:
            for i in range(10):
                print('nop')
        self._test('getnameinfo', ('www.gevent.org', None), 0)

    @greentest.skipOnLibuvOnPyPyOnWin("Errors don't match")
    def test_typeerror_str(self):
        if False:
            return 10
        self._test('getnameinfo', ('www.gevent.org', 'x'), 0)

    def test_overflow_port_too_large(self):
        if False:
            print('Hello World!')
        self._test('getnameinfo', ('www.gevent.org', 65536), 0)
if __name__ == '__main__':
    greentest.main()