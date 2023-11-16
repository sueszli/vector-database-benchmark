"""Tests for directory lookup through Launchpad.net"""
import os
import xmlrpclib
import bzrlib
from bzrlib import debug, errors, tests, transport
from bzrlib.branch import Branch
from bzrlib.directory_service import directories
from bzrlib.tests import features, ssl_certs, TestCaseInTempDir, TestCaseWithMemoryTransport
from bzrlib.plugins.launchpad import _register_directory, lp_registration
from bzrlib.plugins.launchpad.lp_directory import LaunchpadDirectory
from bzrlib.plugins.launchpad.account import get_lp_login, set_lp_login
from bzrlib.tests import http_server

def load_tests(standard_tests, module, loader):
    if False:
        return 10
    result = loader.suiteClass()
    (t_tests, remaining_tests) = tests.split_suite_by_condition(standard_tests, tests.condition_isinstance((TestXMLRPCTransport,)))
    transport_scenarios = [('http', dict(server_class=PreCannedHTTPServer))]
    if features.HTTPSServerFeature.available():
        transport_scenarios.append(('https', dict(server_class=PreCannedHTTPSServer)))
    tests.multiply_tests(t_tests, transport_scenarios, result)
    result.addTests(remaining_tests)
    return result

class FakeResolveFactory(object):

    def __init__(self, test, expected_path, result):
        if False:
            i = 10
            return i + 15
        self._test = test
        self._expected_path = expected_path
        self._result = result
        self._submitted = False

    def __call__(self, path):
        if False:
            return 10
        self._test.assertEqual(self._expected_path, path)
        return self

    def submit(self, service):
        if False:
            while True:
                i = 10
        self._service_url = service.service_url
        self._submitted = True
        return self._result

class LocalDirectoryURLTests(TestCaseInTempDir):
    """Tests for branch urls that we try to pass through local resolution."""

    def assertResolve(self, expected, url, submitted=False):
        if False:
            return 10
        path = url[url.index(':') + 1:].lstrip('/')
        factory = FakeResolveFactory(self, path, dict(urls=['bzr+ssh://fake-resolved']))
        directory = LaunchpadDirectory()
        self.assertEqual(expected, directory._resolve(url, factory, _lp_login='user'))
        self.assertEqual(submitted, factory._submitted)

    def test_short_form(self):
        if False:
            print('Hello World!')
        self.assertResolve('bzr+ssh://bazaar.launchpad.net/+branch/apt', 'lp:apt')

    def test_two_part_form(self):
        if False:
            i = 10
            return i + 15
        self.assertResolve('bzr+ssh://bazaar.launchpad.net/+branch/apt/2.2', 'lp:apt/2.2')

    def test_two_part_plus_subdir(self):
        if False:
            return 10
        self.assertResolve('bzr+ssh://bazaar.launchpad.net/+branch/apt/2.2/BUGS', 'lp:apt/2.2/BUGS')

    def test_user_expansion(self):
        if False:
            return 10
        self.assertResolve('bzr+ssh://bazaar.launchpad.net/~user/apt/foo', 'lp:~/apt/foo')

    def test_ubuntu(self):
        if False:
            while True:
                i = 10
        self.assertResolve('bzr+ssh://bazaar.launchpad.net/+branch/ubuntu', 'lp:ubuntu')

    def test_ubuntu_invalid(self):
        if False:
            i = 10
            return i + 15
        "Invalid ubuntu urls don't crash.\n\n        :seealso: http://pad.lv/843900\n        "
        self.assertRaises(errors.InvalidURL, self.assertResolve, '', 'ubuntu:natty/updates/smartpm')

    def test_ubuntu_apt(self):
        if False:
            print('Hello World!')
        self.assertResolve('bzr+ssh://bazaar.launchpad.net/+branch/ubuntu/apt', 'lp:ubuntu/apt')

    def test_ubuntu_natty_apt(self):
        if False:
            print('Hello World!')
        self.assertResolve('bzr+ssh://bazaar.launchpad.net/+branch/ubuntu/natty/apt', 'lp:ubuntu/natty/apt')

    def test_ubuntu_natty_apt_filename(self):
        if False:
            i = 10
            return i + 15
        self.assertResolve('bzr+ssh://bazaar.launchpad.net/+branch/ubuntu/natty/apt/filename', 'lp:ubuntu/natty/apt/filename')

    def test_user_two_part(self):
        if False:
            while True:
                i = 10
        self.assertResolve('bzr+ssh://fake-resolved', 'lp:~jameinel/apt', submitted=True)

    def test_user_three_part(self):
        if False:
            print('Hello World!')
        self.assertResolve('bzr+ssh://bazaar.launchpad.net/~jameinel/apt/foo', 'lp:~jameinel/apt/foo')

    def test_user_three_part_plus_filename(self):
        if False:
            while True:
                i = 10
        self.assertResolve('bzr+ssh://bazaar.launchpad.net/~jameinel/apt/foo/fname', 'lp:~jameinel/apt/foo/fname')

    def test_user_ubuntu_two_part(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertResolve('bzr+ssh://fake-resolved', 'lp:~jameinel/ubuntu', submitted=True)
        self.assertResolve('bzr+ssh://fake-resolved', 'lp:~jameinel/debian', submitted=True)

    def test_user_ubuntu_three_part(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertResolve('bzr+ssh://fake-resolved', 'lp:~jameinel/ubuntu/natty', submitted=True)
        self.assertResolve('bzr+ssh://fake-resolved', 'lp:~jameinel/debian/sid', submitted=True)

    def test_user_ubuntu_four_part(self):
        if False:
            return 10
        self.assertResolve('bzr+ssh://fake-resolved', 'lp:~jameinel/ubuntu/natty/project', submitted=True)
        self.assertResolve('bzr+ssh://fake-resolved', 'lp:~jameinel/debian/sid/project', submitted=True)

    def test_user_ubuntu_five_part(self):
        if False:
            while True:
                i = 10
        self.assertResolve('bzr+ssh://bazaar.launchpad.net/~jameinel/ubuntu/natty/apt/branch', 'lp:~jameinel/ubuntu/natty/apt/branch')
        self.assertResolve('bzr+ssh://bazaar.launchpad.net/~jameinel/debian/sid/apt/branch', 'lp:~jameinel/debian/sid/apt/branch')

    def test_user_ubuntu_five_part_plus_subdir(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertResolve('bzr+ssh://bazaar.launchpad.net/~jameinel/ubuntu/natty/apt/branch/f', 'lp:~jameinel/ubuntu/natty/apt/branch/f')
        self.assertResolve('bzr+ssh://bazaar.launchpad.net/~jameinel/debian/sid/apt/branch/f', 'lp:~jameinel/debian/sid/apt/branch/f')

    def test_handles_special_lp(self):
        if False:
            return 10
        self.assertResolve('bzr+ssh://bazaar.launchpad.net/+branch/apt', 'lp:apt')
        self.assertResolve('bzr+ssh://bazaar.launchpad.net/+branch/apt', 'lp:///apt')
        self.assertResolve('bzr+ssh://bazaar.launchpad.net/+branch/apt', 'lp://production/apt')
        self.assertResolve('bzr+ssh://bazaar.launchpad.dev/+branch/apt', 'lp://dev/apt')
        self.assertResolve('bzr+ssh://bazaar.staging.launchpad.net/+branch/apt', 'lp://staging/apt')
        self.assertResolve('bzr+ssh://bazaar.qastaging.launchpad.net/+branch/apt', 'lp://qastaging/apt')
        self.assertResolve('bzr+ssh://bazaar.demo.launchpad.net/+branch/apt', 'lp://demo/apt')

    def test_debug_launchpad_uses_resolver(self):
        if False:
            print('Hello World!')
        self.assertResolve('bzr+ssh://bazaar.launchpad.net/+branch/bzr', 'lp:bzr', submitted=False)
        debug.debug_flags.add('launchpad')
        self.addCleanup(debug.debug_flags.discard, 'launchpad')
        self.assertResolve('bzr+ssh://fake-resolved', 'lp:bzr', submitted=True)

class DirectoryUrlTests(TestCaseInTempDir):
    """Tests for branch urls through Launchpad.net directory"""

    def test_short_form(self):
        if False:
            return 10
        'A launchpad url should map to a http url'
        factory = FakeResolveFactory(self, 'apt', dict(urls=['http://bazaar.launchpad.net/~apt/apt/devel']))
        directory = LaunchpadDirectory()
        self.assertEqual('http://bazaar.launchpad.net/~apt/apt/devel', directory._resolve('lp:apt', factory))
        self.assertEqual('https://xmlrpc.launchpad.net/bazaar/', factory._service_url)

    def test_qastaging(self):
        if False:
            print('Hello World!')
        'A launchpad url should map to a http url'
        factory = FakeResolveFactory(self, 'apt', dict(urls=['http://bazaar.qastaging.launchpad.net/~apt/apt/devel']))
        url = 'lp://qastaging/apt'
        directory = LaunchpadDirectory()
        self.assertEqual('http://bazaar.qastaging.launchpad.net/~apt/apt/devel', directory._resolve(url, factory))
        self.assertEqual('https://xmlrpc.qastaging.launchpad.net/bazaar/', factory._service_url)

    def test_staging(self):
        if False:
            while True:
                i = 10
        'A launchpad url should map to a http url'
        factory = FakeResolveFactory(self, 'apt', dict(urls=['http://bazaar.staging.launchpad.net/~apt/apt/devel']))
        url = 'lp://staging/apt'
        directory = LaunchpadDirectory()
        self.assertEqual('http://bazaar.staging.launchpad.net/~apt/apt/devel', directory._resolve(url, factory))
        self.assertEqual('https://xmlrpc.staging.launchpad.net/bazaar/', factory._service_url)

    def test_url_from_directory(self):
        if False:
            print('Hello World!')
        'A launchpad url should map to a http url'
        factory = FakeResolveFactory(self, 'apt', dict(urls=['http://bazaar.launchpad.net/~apt/apt/devel']))
        directory = LaunchpadDirectory()
        self.assertEqual('http://bazaar.launchpad.net/~apt/apt/devel', directory._resolve('lp:///apt', factory))

    def test_directory_skip_bad_schemes(self):
        if False:
            return 10
        factory = FakeResolveFactory(self, 'apt', dict(urls=['bad-scheme://bazaar.launchpad.net/~apt/apt/devel', 'http://bazaar.launchpad.net/~apt/apt/devel', 'http://another/location']))
        directory = LaunchpadDirectory()
        self.assertEqual('http://bazaar.launchpad.net/~apt/apt/devel', directory._resolve('lp:///apt', factory))

    def test_directory_no_matching_schemes(self):
        if False:
            while True:
                i = 10
        factory = FakeResolveFactory(self, 'apt', dict(urls=['bad-scheme://bazaar.launchpad.net/~apt/apt/devel']))
        directory = LaunchpadDirectory()
        self.assertRaises(errors.InvalidURL, directory._resolve, 'lp:///apt', factory)

    def test_directory_fault(self):
        if False:
            i = 10
            return i + 15
        factory = FakeResolveFactory(self, 'apt', None)

        def submit(service):
            if False:
                print('Hello World!')
            raise xmlrpclib.Fault(42, 'something went wrong')
        factory.submit = submit
        directory = LaunchpadDirectory()
        self.assertRaises(errors.InvalidURL, directory._resolve, 'lp:///apt', factory)

    def test_skip_bzr_ssh_launchpad_net_when_anonymous(self):
        if False:
            return 10
        self.assertEqual(None, get_lp_login())
        factory = FakeResolveFactory(self, 'apt', dict(urls=['bzr+ssh://bazaar.launchpad.net/~apt/apt/devel', 'http://bazaar.launchpad.net/~apt/apt/devel']))
        directory = LaunchpadDirectory()
        self.assertEqual('http://bazaar.launchpad.net/~apt/apt/devel', directory._resolve('lp:///apt', factory))

    def test_skip_sftp_launchpad_net_when_anonymous(self):
        if False:
            return 10
        self.assertEqual(None, get_lp_login())
        factory = FakeResolveFactory(self, 'apt', dict(urls=['sftp://bazaar.launchpad.net/~apt/apt/devel', 'http://bazaar.launchpad.net/~apt/apt/devel']))
        directory = LaunchpadDirectory()
        self.assertEqual('http://bazaar.launchpad.net/~apt/apt/devel', directory._resolve('lp:///apt', factory))

    def test_with_login_avoid_resolve_factory(self):
        if False:
            while True:
                i = 10
        factory = FakeResolveFactory(self, 'apt', dict(urls=['bzr+ssh://my-super-custom/special/devel', 'http://bazaar.launchpad.net/~apt/apt/devel']))
        directory = LaunchpadDirectory()
        self.assertEqual('bzr+ssh://bazaar.launchpad.net/+branch/apt', directory._resolve('lp:///apt', factory, _lp_login='username'))

    def test_no_rewrite_of_other_bzr_ssh(self):
        if False:
            while True:
                i = 10
        self.assertEqual(None, get_lp_login())
        factory = FakeResolveFactory(self, 'apt', dict(urls=['bzr+ssh://example.com/~apt/apt/devel', 'http://bazaar.launchpad.net/~apt/apt/devel']))
        directory = LaunchpadDirectory()
        self.assertEqual('bzr+ssh://example.com/~apt/apt/devel', directory._resolve('lp:///apt', factory))

    def test_error_for_bad_url(self):
        if False:
            return 10
        directory = LaunchpadDirectory()
        self.assertRaises(errors.InvalidURL, directory._resolve, 'lp://ratotehunoahu')

    def test_resolve_tilde_to_user(self):
        if False:
            while True:
                i = 10
        factory = FakeResolveFactory(self, '~username/apt/test', dict(urls=['bzr+ssh://bazaar.launchpad.net/~username/apt/test']))
        directory = LaunchpadDirectory()
        self.assertEqual('bzr+ssh://bazaar.launchpad.net/~username/apt/test', directory._resolve('lp:~/apt/test', factory, _lp_login='username'))
        set_lp_login('username')
        self.assertEqual('bzr+ssh://bazaar.launchpad.net/~username/apt/test', directory._resolve('lp:~/apt/test', factory))

    def test_tilde_fails_no_login(self):
        if False:
            return 10
        factory = FakeResolveFactory(self, '~username/apt/test', dict(urls=['bzr+ssh://bazaar.launchpad.net/~username/apt/test']))
        self.assertIs(None, get_lp_login())
        directory = LaunchpadDirectory()
        self.assertRaises(errors.InvalidURL, directory._resolve, 'lp:~/apt/test', factory)

class DirectoryOpenBranchTests(TestCaseWithMemoryTransport):

    def test_directory_open_branch(self):
        if False:
            return 10
        target_branch = self.make_branch('target')

        class FooService(object):
            """A directory service that maps the name to a FILE url"""

            def look_up(self, name, url):
                if False:
                    while True:
                        i = 10
                if 'lp:///apt' == url:
                    return target_branch.base.rstrip('/')
                return '!unexpected look_up value!'
        directories.remove('lp:')
        directories.remove('ubuntu:')
        directories.remove('debianlp:')
        directories.register('lp:', FooService, 'Map lp URLs to local urls')
        self.addCleanup(_register_directory)
        self.addCleanup(directories.remove, 'lp:')
        t = transport.get_transport('lp:///apt')
        branch = Branch.open_from_transport(t)
        self.assertEqual(target_branch.base, branch.base)

class PredefinedRequestHandler(http_server.TestingHTTPRequestHandler):
    """Request handler for a unique and pre-defined request.

    The only thing we care about here is that we receive a connection. But
    since we want to dialog with a real http client, we have to send it correct
    responses.

    We expect to receive a *single* request nothing more (and we won't even
    check what request it is), the tests will recognize us from our response.
    """

    def handle_one_request(self):
        if False:
            while True:
                i = 10
        tcs = self.server.test_case_server
        requestline = self.rfile.readline()
        self.MessageClass(self.rfile, 0)
        if requestline.startswith('POST'):
            self.rfile.readline()
        self.wfile.write(tcs.canned_response)

class PreCannedServerMixin(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(PreCannedServerMixin, self).__init__(request_handler=PredefinedRequestHandler)
        self.bytes_read = 0
        self.bytes_written = 0
        self.canned_response = None

class PreCannedHTTPServer(PreCannedServerMixin, http_server.HttpServer):
    pass
if features.HTTPSServerFeature.available():
    from bzrlib.tests import https_server

    class PreCannedHTTPSServer(PreCannedServerMixin, https_server.HTTPSServer):
        pass

class TestXMLRPCTransport(tests.TestCase):
    server_class = None

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestXMLRPCTransport, self).setUp()
        self.server = self.server_class()
        self.server.start_server()
        self.addCleanup(self.server.stop_server)
        self.overrideEnv('BZR_LP_XMLRPC_URL', None)
        bzrlib.global_state.cmdline_overrides._from_cmdline(['ssl.ca_certs=%s' % ssl_certs.build_path('ca.crt')])

    def set_canned_response(self, server, path):
        if False:
            for i in range(10):
                print('nop')
        response_format = 'HTTP/1.1 200 OK\r\nDate: Tue, 11 Jul 2006 04:32:56 GMT\r\nServer: Apache/2.0.54 (Fedora)\r\nLast-Modified: Sun, 23 Apr 2006 19:35:20 GMT\r\nETag: "56691-23-38e9ae00"\r\nAccept-Ranges: bytes\r\nContent-Length: %(length)d\r\nConnection: close\r\nContent-Type: text/plain; charset=UTF-8\r\n\r\n<?xml version=\'1.0\'?>\n<methodResponse>\n<params>\n<param>\n<value><struct>\n<member>\n<name>urls</name>\n<value><array><data>\n<value><string>bzr+ssh://bazaar.launchpad.net/%(path)s</string></value>\n<value><string>http://bazaar.launchpad.net/%(path)s</string></value>\n</data></array></value>\n</member>\n</struct></value>\n</param>\n</params>\n</methodResponse>\n'
        length = 334 + 2 * len(path)
        server.canned_response = response_format % dict(length=length, path=path)

    def do_request(self, server_url):
        if False:
            for i in range(10):
                print('nop')
        os.environ['BZR_LP_XMLRPC_URL'] = self.server.get_url()
        service = lp_registration.LaunchpadService()
        resolve = lp_registration.ResolveLaunchpadPathRequest('bzr')
        result = resolve.submit(service)
        return result

    def test_direct_request(self):
        if False:
            i = 10
            return i + 15
        self.set_canned_response(self.server, '~bzr-pqm/bzr/bzr.dev')
        result = self.do_request(self.server.get_url())
        urls = result.get('urls', None)
        self.assertIsNot(None, urls)
        self.assertEqual(['bzr+ssh://bazaar.launchpad.net/~bzr-pqm/bzr/bzr.dev', 'http://bazaar.launchpad.net/~bzr-pqm/bzr/bzr.dev'], urls)

class TestDebuntuExpansions(TestCaseInTempDir):
    """Test expansions for ubuntu: and debianlp: schemes."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestDebuntuExpansions, self).setUp()
        self.directory = LaunchpadDirectory()

    def _make_factory(self, package='foo', distro='ubuntu', series=None):
        if False:
            for i in range(10):
                print('nop')
        if series is None:
            path = '%s/%s' % (distro, package)
            url_suffix = '~branch/%s/%s' % (distro, package)
        else:
            path = '%s/%s/%s' % (distro, series, package)
            url_suffix = '~branch/%s/%s/%s' % (distro, series, package)
        return FakeResolveFactory(self, path, dict(urls=['http://bazaar.launchpad.net/' + url_suffix]))

    def assertURL(self, expected_url, shortcut, package='foo', distro='ubuntu', series=None):
        if False:
            for i in range(10):
                print('nop')
        factory = self._make_factory(package=package, distro=distro, series=series)
        self.assertEqual('http://bazaar.launchpad.net/~branch/' + expected_url, self.directory._resolve(shortcut, factory))

    def test_bogus_distro(self):
        if False:
            print('Hello World!')
        self.assertRaises(errors.InvalidURL, self.directory._resolve, 'gentoo:foo')

    def test_trick_bogus_distro_u(self):
        if False:
            print('Hello World!')
        self.assertRaises(errors.InvalidURL, self.directory._resolve, 'utube:foo')

    def test_trick_bogus_distro_d(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(errors.InvalidURL, self.directory._resolve, 'debuntu:foo')

    def test_missing_ubuntu_distroseries_without_project(self):
        if False:
            print('Hello World!')
        self.assertURL('ubuntu/intrepid', 'ubuntu:intrepid', package='intrepid')

    def test_missing_ubuntu_distroseries_with_project(self):
        if False:
            i = 10
            return i + 15
        self.assertURL('ubuntu/intrepid/foo', 'ubuntu:intrepid/foo', series='intrepid')

    def test_missing_debian_distroseries(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertURL('debian/sid', 'debianlp:sid', package='sid', distro='debian')

    def test_ubuntu_default_distroseries_expansion(self):
        if False:
            while True:
                i = 10
        self.assertURL('ubuntu/foo', 'ubuntu:foo')

    def test_ubuntu_natty_distroseries_expansion(self):
        if False:
            i = 10
            return i + 15
        self.assertURL('ubuntu/natty/foo', 'ubuntu:natty/foo', series='natty')

    def test_ubuntu_n_distroseries_expansion(self):
        if False:
            return 10
        self.assertURL('ubuntu/natty/foo', 'ubuntu:n/foo', series='natty')

    def test_ubuntu_maverick_distroseries_expansion(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertURL('ubuntu/maverick/foo', 'ubuntu:maverick/foo', series='maverick')

    def test_ubuntu_m_distroseries_expansion(self):
        if False:
            return 10
        self.assertURL('ubuntu/maverick/foo', 'ubuntu:m/foo', series='maverick')

    def test_ubuntu_lucid_distroseries_expansion(self):
        if False:
            i = 10
            return i + 15
        self.assertURL('ubuntu/lucid/foo', 'ubuntu:lucid/foo', series='lucid')

    def test_ubuntu_l_distroseries_expansion(self):
        if False:
            while True:
                i = 10
        self.assertURL('ubuntu/lucid/foo', 'ubuntu:l/foo', series='lucid')

    def test_ubuntu_karmic_distroseries_expansion(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertURL('ubuntu/karmic/foo', 'ubuntu:karmic/foo', series='karmic')

    def test_ubuntu_k_distroseries_expansion(self):
        if False:
            print('Hello World!')
        self.assertURL('ubuntu/karmic/foo', 'ubuntu:k/foo', series='karmic')

    def test_ubuntu_jaunty_distroseries_expansion(self):
        if False:
            return 10
        self.assertURL('ubuntu/jaunty/foo', 'ubuntu:jaunty/foo', series='jaunty')

    def test_ubuntu_j_distroseries_expansion(self):
        if False:
            while True:
                i = 10
        self.assertURL('ubuntu/jaunty/foo', 'ubuntu:j/foo', series='jaunty')

    def test_ubuntu_hardy_distroseries_expansion(self):
        if False:
            while True:
                i = 10
        self.assertURL('ubuntu/hardy/foo', 'ubuntu:hardy/foo', series='hardy')

    def test_ubuntu_h_distroseries_expansion(self):
        if False:
            while True:
                i = 10
        self.assertURL('ubuntu/hardy/foo', 'ubuntu:h/foo', series='hardy')

    def test_ubuntu_dapper_distroseries_expansion(self):
        if False:
            return 10
        self.assertURL('ubuntu/dapper/foo', 'ubuntu:dapper/foo', series='dapper')

    def test_ubuntu_d_distroseries_expansion(self):
        if False:
            while True:
                i = 10
        self.assertURL('ubuntu/dapper/foo', 'ubuntu:d/foo', series='dapper')

    def test_debian_default_distroseries_expansion(self):
        if False:
            while True:
                i = 10
        self.assertURL('debian/foo', 'debianlp:foo', distro='debian')

    def test_debian_squeeze_distroseries_expansion(self):
        if False:
            return 10
        self.assertURL('debian/squeeze/foo', 'debianlp:squeeze/foo', distro='debian', series='squeeze')

    def test_debian_lenny_distroseries_expansion(self):
        if False:
            i = 10
            return i + 15
        self.assertURL('debian/lenny/foo', 'debianlp:lenny/foo', distro='debian', series='lenny')