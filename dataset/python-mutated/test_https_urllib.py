"""Tests for the SSL support in the urllib HTTP transport.

"""
import os
import sys
from bzrlib import config, trace
from bzrlib.errors import ConfigOptionValueError
from bzrlib import tests
from bzrlib.transport.http import _urllib2_wrappers
from bzrlib.transport.http._urllib2_wrappers import ssl

class CaCertsConfigTests(tests.TestCaseInTempDir):

    def get_stack(self, content):
        if False:
            i = 10
            return i + 15
        return config.MemoryStack(content.encode('utf-8'))

    def test_default_exists(self):
        if False:
            while True:
                i = 10
        'Check that the default we provide exists for the tested platform.'
        stack = self.get_stack('')
        self.assertPathExists(stack.get('ssl.ca_certs'))

    def test_specified(self):
        if False:
            i = 10
            return i + 15
        self.build_tree(['cacerts.pem'])
        path = os.path.join(self.test_dir, 'cacerts.pem')
        stack = self.get_stack('ssl.ca_certs = %s\n' % path)
        self.assertEqual(path, stack.get('ssl.ca_certs'))

    def test_specified_doesnt_exist(self):
        if False:
            print('Hello World!')
        stack = self.get_stack('')
        self.overrideAttr(_urllib2_wrappers.opt_ssl_ca_certs, 'default', os.path.join(self.test_dir, u'nonexisting.pem'))
        self.warnings = []

        def warning(*args):
            if False:
                return 10
            self.warnings.append(args[0] % args[1:])
        self.overrideAttr(trace, 'warning', warning)
        self.assertEqual(None, stack.get('ssl.ca_certs'))
        self.assertLength(1, self.warnings)
        self.assertContainsRe(self.warnings[0], 'is not valid for "ssl.ca_certs"')

class CertReqsConfigTests(tests.TestCaseInTempDir):

    def test_default(self):
        if False:
            while True:
                i = 10
        stack = config.MemoryStack('')
        self.assertEqual(ssl.CERT_REQUIRED, stack.get('ssl.cert_reqs'))

    def test_from_string(self):
        if False:
            for i in range(10):
                print('nop')
        stack = config.MemoryStack('ssl.cert_reqs = none\n')
        self.assertEqual(ssl.CERT_NONE, stack.get('ssl.cert_reqs'))
        stack = config.MemoryStack('ssl.cert_reqs = required\n')
        self.assertEqual(ssl.CERT_REQUIRED, stack.get('ssl.cert_reqs'))
        stack = config.MemoryStack('ssl.cert_reqs = invalid\n')
        self.assertRaises(ConfigOptionValueError, stack.get, 'ssl.cert_reqs')

class MatchHostnameTests(tests.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(MatchHostnameTests, self).setUp()
        if sys.version_info < (2, 7, 9):
            raise tests.TestSkipped('python version too old to provide proper https hostname verification')

    def test_no_certificate(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(ValueError, ssl.match_hostname, {}, 'example.com')

    def test_wildcards_in_cert(self):
        if False:
            print('Hello World!')

        def ok(cert, hostname):
            if False:
                while True:
                    i = 10
            ssl.match_hostname(cert, hostname)

        def not_ok(cert, hostname):
            if False:
                for i in range(10):
                    print('nop')
            self.assertRaises(ssl.CertificateError, ssl.match_hostname, cert, hostname)
        ok({'subject': ((('commonName', 'a*b.com'),),)}, 'axxb.com')
        not_ok({'subject': ((('commonName', 'a*b.co*'),),)}, 'axxb.com')
        not_ok({'subject': ((('commonName', 'a*b*.com'),),)}, 'axxbxxc.com')

    def test_no_valid_attributes(self):
        if False:
            while True:
                i = 10
        self.assertRaises(ssl.CertificateError, ssl.match_hostname, {'Problem': 'Solved'}, 'example.com')

    def test_common_name(self):
        if False:
            while True:
                i = 10
        cert = {'subject': ((('commonName', 'example.com'),),)}
        self.assertIs(None, ssl.match_hostname(cert, 'example.com'))
        self.assertRaises(ssl.CertificateError, ssl.match_hostname, cert, 'example.org')